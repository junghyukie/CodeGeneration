import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
import time
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG
from transformers import GenerationConfig

class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        self.generation_config = GenerationConfig(
            do_sample=self.args.do_sample,
            temperature=self.args.temperature if self.args.do_sample else None,
            top_p=self.args.top_p if self.args.do_sample else None,
            repetition_penalty=self.args.repetition_penalty,
        )

    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def _task_eval_from_predictions(self, task, sources_sequences, predicted_sequences, ground_truths):
        if task in ['CodeSearchNet', 'TheVault_Csharp']:
            calc_codebleu = False
        else:
            calc_codebleu = True
        return compute_metrics(predicted_sequences, ground_truths, calc_codebleu=calc_codebleu, language=DATASET_TO_OUTPUT_LANG.get(task, None))

    def _ordered_unique_prediction_rows(self, prediction_rows):
        if prediction_rows and all("__index__" in row for row in prediction_rows):
            rows_by_index = {}
            for row in prediction_rows:
                index = int(row["__index__"])
                if index not in rows_by_index:
                    rows_by_index[index] = row
            prediction_rows = [rows_by_index[index] for index in sorted(rows_by_index)]

        return [
            {key: value for key, value in row.items() if key != "__index__"}
            for row in prediction_rows
        ]

    def _gather_prediction_rows(self, prediction_rows):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered_rows = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_rows, prediction_rows)
            prediction_rows = [
                row
                for rank_rows in gathered_rows
                for row in rank_rows
            ]

        if not prediction_rows:
            return prediction_rows

        if all("__index__" in row for row in prediction_rows):
            return self._ordered_unique_prediction_rows(prediction_rows)

        # No indices available: de-duplicate by row content while preserving order.
        seen = set()
        unique_rows = []
        for row in prediction_rows:
            key = (row.get("source"), row.get("ground-truth"), str(row.get("prediction")))
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        return unique_rows

    def task_generation_evaluation(self, task, test_dataloader, device, max_ans_len=None, return_predictions=False):
        self.model.eval()
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        sample_indices = []

        if max_ans_len is None:
            max_ans_len = getattr(self.args, "max_ans_len", 256)

        is_executable = getattr(self.args, "benchmark", "non-executable") != "non-executable"
        if is_executable:
            return_predictions = True
            num_return_sequences = int(getattr(self.args, "num_return_sequences", 1))
            top_k = int(getattr(self.args, "top_k", 0))
            generation_kwargs = self.generation_config.to_dict()
            generation_kwargs.update({
                "num_return_sequences": num_return_sequences,
                "top_k": top_k,
            })
            generation_config = GenerationConfig(**generation_kwargs)
        else:
            num_return_sequences = 1
            generation_config = self.generation_config

        progress_bar = tqdm(total=len(test_dataloader), leave=True, disable=(self.args.global_rank != 0))
        for step, batch in enumerate(test_dataloader):
            batch_indices = batch.pop('indices', None)
            if batch_indices is not None:
                sample_indices.extend(batch_indices.detach().cpu().tolist())

            sources_sequences += batch['sources']
            if 'gts' in batch:
                ground_truths += batch['gts']
                del batch['gts']
            elif 'labels' in batch:
                label_tensor = batch['labels']
                for row in label_tensor:
                    valid_ids = row[row != -100].detach().cpu().tolist()
                    ground_truths.append(
                        self.tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    )
                del batch['labels']
            else:
                ground_truths += [''] * len(batch['sources'])

            del batch['sources']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]

            with torch.no_grad():
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id

                generate_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=max_ans_len,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    use_cache=True,
                )

            sequences = self.tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            if is_executable and num_return_sequences > 1:
                batch_preds = [
                    sequences[i:i + num_return_sequences]
                    for i in range(0, len(sequences), num_return_sequences)
                ]
                predicted_sequences.extend(batch_preds)
            else:
                predicted_sequences += sequences

            if self.args.global_rank == 0:
                progress_bar.update(1)
                description = f"Test step {step}"
                progress_bar.set_description(description, refresh=False)

        prediction_rows = [
            {
                "source": source,
                "ground-truth": gt,
                "prediction": pred,
            }
            for source, gt, pred in zip(sources_sequences, ground_truths, predicted_sequences)
        ]
        if len(sample_indices) == len(prediction_rows):
            for row, index in zip(prediction_rows, sample_indices):
                row["__index__"] = index

        prediction_rows = self._gather_prediction_rows(prediction_rows)
        sources_sequences = [row["source"] for row in prediction_rows]
        ground_truths = [row["ground-truth"] for row in prediction_rows]
        predicted_sequences = [row["prediction"] for row in prediction_rows]

        metrics = {} if is_executable else self._task_eval_from_predictions(
            task, sources_sequences, predicted_sequences, ground_truths
        )
        if return_predictions:
            return metrics, prediction_rows
        return metrics

    def _resolve_max_ans_len(self, task_idx):
        max_ans_len = getattr(self.args, "max_ans_len", 256)
        if isinstance(max_ans_len, (list, tuple)):
            if len(max_ans_len) == 0:
                return 256
            if len(max_ans_len) == 1:
                return int(max_ans_len[0])
            return int(max_ans_len[task_idx])
        return int(max_ans_len)

    def _save_generation_predictions(self, split_name, task_idx, task, metrics, prediction_rows):
        if self.args.global_rank != 0 or self.args.output_dir is None:
            return
        safe_task_name = str(task).replace("/", "_").replace(":", "_")
        pred_dir = os.path.join(self.args.output_dir, "predictions", split_name)
        os.makedirs(pred_dir, exist_ok=True)
        pred_file = os.path.join(pred_dir, f"{task_idx}_{safe_task_name}.json")
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": prediction_rows}, f, ensure_ascii=False, indent=2)
        print_rank_0(f"Saved {split_name} predictions to {pred_file}", self.args.global_rank)

    def test_all_tasks_and_save_predictions(self):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        for task_idx, (task_name, test_dataloader) in enumerate(self.test_task_list.items()):
            print_rank_0(
                f"***** Final testing on task {task_name} after continual training *****",
                self.args.global_rank,
            )
            test_result, prediction_rows = self.task_generation_evaluation(
                task_name,
                test_dataloader,
                device,
                max_ans_len=self._resolve_max_ans_len(task_idx),
                return_predictions=True,
            )
            self._save_generation_predictions("final-test", task_idx, task_name, test_result, prediction_rows)


    def train_one_task(self, task, i_task, epochs):
        # 在单独某个任务上训练
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        test_dataloader = self.test_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        global_step = 0
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                global_step += 1
                del batch['sources']
                batch.pop('indices', None)
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    logging_steps = getattr(self.args, 'logging_steps', 10)
                    if global_step % logging_steps == 0:
                        print_rank_0(f"task={task} epoch={epoch+1} step={global_step} loss={loss.item():.6f}", self.args.global_rank)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()

            # Validate on eval split after each epoch.
            print_rank_0(
                f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                self.args.global_rank)
            eval_result, eval_predictions = self.task_generation_evaluation(
                task,
                eval_dataloader,
                device,
                max_ans_len=self._resolve_max_ans_len(i_task),
                return_predictions=True,
            )
            print_rank_0(f"[task={task}] validation result: {eval_result}", self.args.global_rank)

            self._save_generation_predictions(f"eval-epoch{epoch+1}", i_task, task, eval_result, eval_predictions)
        
        for seen_idx, (test_task, test_dataset) in enumerate(list(self.test_task_list.items())[:i_task+1]):
            print_rank_0(
                f"***** Testing on current task {test_task} after training {task} on all epochs *****",
                self.args.global_rank)
            test_result, test_predictions = self.task_generation_evaluation(
                test_task,
                test_dataset,
                device,
                max_ans_len=self._resolve_max_ans_len(seen_idx),
                return_predictions=True,
            )
            print_rank_0(f"[task={test_task}] post-train test result: {test_result}", self.args.global_rank)

            self._save_generation_predictions("test-after-task", i_task, test_task, test_result, test_predictions)
    
    
    def train_continual(self):
        start_task_id = int(getattr(self.args, "start_task_id", 0))
        task_items = list(self.train_task_list.items())[start_task_id:]
        for i_task, (task, _) in enumerate(task_items, start=start_task_id):
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            self.save_model(i_task)
        # self.test_all_tasks_and_save_predictions()

    
    def run_calibration_inference(self):
        """Multi-GPU inference on calibration_MBPP using the distributed eval DataLoader.

        Expects:
          - eval_task_list[language] to be a DataLoader wrapping the calibration split
            (built via create_executable_dataset(..., use_calibration_eval=True)).
          - args.calibration_datasets[language]: the original HF Dataset with a 'test' column
            used to retrieve unit-test code by sample index.
        """
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        language = self.args.dataset_name[0]
        calib_ds = getattr(self.args, "calibration_datasets", {}).get(language)
        has_test = calib_ds is not None and "test" in calib_ds.column_names

        eval_loader = self.eval_task_list.get(language)
        if eval_loader is None:
            print_rank_0(
                f"[calibration] No eval DataLoader for language={language}",
                self.args.global_rank,
            )
            return

        print_rank_0(
            f"[calibration] Running inference on calibration_MBPP  language={language}",
            self.args.global_rank,
        )

        do_sample = getattr(self.args, "do_sample", False)
        num_ret = int(getattr(self.args, "num_return_sequences", 1)) if do_sample else 1
        max_ans = int(self.args.max_ans_len[0])
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        gen_kwargs = dict(
            max_new_tokens=max_ans,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=getattr(self.args, "temperature", 0.2),
                top_p=getattr(self.args, "top_p", 0.95),
                num_return_sequences=num_ret,
            )
            top_k = int(getattr(self.args, "top_k", 0))
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        rep_pen = getattr(self.args, "repetition_penalty", 1.0)
        if rep_pen != 1.0:
            gen_kwargs["repetition_penalty"] = rep_pen

        self.model.eval()
        # local_rows: list of (original_index, source, ground_truth, prediction)
        local_rows = []
        progress = tqdm(
            total=len(eval_loader),
            leave=True,
            disable=(self.args.global_rank != 0),
            desc="calibration",
        )

        for batch in eval_loader:
            batch_indices = batch.pop("indices", None)
            sources = batch.pop("sources", [])
            gts = batch.pop("gts", [])
            batch = to_device(batch, device)
            prompt_len = batch["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

            decoded = self.tokenizer.batch_decode(
                output_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            bs = batch["input_ids"].shape[0]
            for i in range(bs):
                seqs = decoded[i * num_ret : (i + 1) * num_ret]
                pred = seqs if num_ret > 1 else seqs[0]
                idx = int(batch_indices[i]) if batch_indices is not None else len(local_rows)
                src = sources[i] if i < len(sources) else ""
                gt = gts[i] if i < len(gts) else ""
                local_rows.append((idx, src, gt, pred))

            if self.args.global_rank == 0:
                progress.update(1)

        progress.close()

        # Ensure all ranks finish generation before the collective.
        # Without this barrier, a fast rank enters all_gather_object while a
        # slow rank is still inside model.generate(), causing an NCCL timeout.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        # Gather across all GPUs
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_rows)
            all_rows = [item for rank_rows in gathered for item in rank_rows]
        else:
            all_rows = local_rows

        # Only rank 0 saves
        if self.args.global_rank != 0:
            return

        n_total = len(calib_ds) if calib_ds is not None else len(all_rows)
        seen = set()
        result_rows = []
        for idx, src, gt, pred in sorted(all_rows, key=lambda x: x[0]):
            if idx in seen or idx >= n_total:
                continue
            seen.add(idx)
            result_rows.append({
                "source": src,
                "ground-truth": gt,
                "prediction": pred,
                "test": calib_ds[idx]["test"] if has_test else "",
            })

        out_dir = getattr(self.args, "inference_output_path", None) or self.args.output_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"calibration_{language}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": {}, "predictions": result_rows}, f, ensure_ascii=False, indent=2)
        print_rank_0(
            f"[calibration] Saved {len(result_rows)} results → {out_file}",
            self.args.global_rank,
        )

    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))

        if self.args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
        print_rank_0('Sucessful saving model after round {}'.format(round), self.args.global_rank)
        
