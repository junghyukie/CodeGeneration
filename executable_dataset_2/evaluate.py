#!/usr/bin/env python3
"""
Evaluate solution snippets against inline tests across multiple languages.

The primary entrypoint is:

    evaluate(language: str, solution: str, test: str) -> dict

The evaluator accepts raw code snippets (optionally wrapped in markdown code
fences), combines the solution with the provided test snippet, and then
executes the result locally or with Docker.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_LANGUAGES = (
    "cpp",
    "python",
    "swift",
    "rust",
    "csharp",
    "java",
    "php",
    "typescript",
    "shell",
)

FILE_NAMES = {
    "cpp": "Main.cpp",
    "python": "main.py",
    "swift": "Main.swift",
    "rust": "main.rs",
    "csharp": "Program.cs",
    "java": "Main.java",
    "php": "main.php",
    "typescript": "main.ts",
    "shell": "main.sh",
}

CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>disable</Nullable>
  </PropertyGroup>
</Project>
"""

DEFAULT_TIMEOUT_SECONDS = 30
THIS_DIR = Path(__file__).resolve().parent
DOCKER_WORKDIR = "/workspace"
DOCKER_JOBS_DIR = "/workspace/jobs"


@dataclass
class PreparedSource:
    language: str
    source: str
    file_name: str
    main_class: str | None = None
    extra_files: dict[str, str] = field(default_factory=dict)


def _empty_result(language: str | None = None) -> dict:
    return {
        "language": language,
        "use_docker": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "execution_time": 0.0,
        "test_passed": False,
    }


def _docker_image() -> str:
    return os.environ.get(
        "EXECUTABLE_DATASET_2_DOCKER_IMAGE",
        "cl4code-executable-dataset-2:latest",
    )


def _docker_container() -> str:
    return os.environ.get(
        "EXECUTABLE_DATASET_2_DOCKER_CONTAINER",
        "cl4code-executable-dataset-2-runner",
    )


def _strip_fenced_code(text: str) -> str:
    stripped = text.strip()
    language_pattern = "|".join(re.escape(lang) for lang in SUPPORTED_LANGUAGES)
    match = re.search(rf"```\s*(?:{language_pattern})\s*\n([\s\S]*?)```", stripped)
    if match:
        return match.group(1).strip()
    match = re.search(r"```[^\n]*\n([\s\S]*?)```", stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _apply_process_result(result: dict, process_result: dict, use_docker: bool, test_passed: bool = False) -> dict:
    result["use_docker"] = use_docker
    result["returncode"] = process_result.get("returncode")
    result["stdout"] = process_result.get("stdout", "")
    result["stderr"] = process_result.get("stderr", "")
    result["execution_time"] = process_result.get("execution_time", 0.0)
    result["test_passed"] = test_passed
    return result


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    lines = text.rstrip().splitlines() or [""]
    return "\n".join(prefix + line if line else prefix for line in lines)


def _ensure_trailing_newline(text: str) -> str:
    return text.rstrip() + "\n"


def _trim_trailing_braces(text: str) -> str:
    balance = 0
    for char in text:
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
    if balance >= 0:
        return text

    lines = text.rstrip().splitlines()
    while lines and balance < 0:
        line = lines[-1].rstrip()
        if line.strip() == "}":
            lines.pop()
            balance += 1
            continue
        if line.endswith("}"):
            lines[-1] = line[:-1].rstrip()
            balance += 1
            if not lines[-1]:
                lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


def _python_source(solution: str, test: str) -> PreparedSource:
    combined = _ensure_trailing_newline(solution) + "\n" + _ensure_trailing_newline(test)
    if re.search(r"\bmath\.", combined) and not re.search(
        r"^\s*(?:import\s+math|from\s+math\s+import\s+.+)\s*$",
        combined,
        re.MULTILINE,
    ):
        future_block = re.match(r"^(?:\s*from\s+__future__\s+import\s+.+\n)+", combined)
        if future_block:
            combined = (
                future_block.group(0)
                + "import math\n"
                + combined[future_block.end():]
            )
        else:
            combined = "import math\n" + combined
    return PreparedSource(language="python", source=combined, file_name=FILE_NAMES["python"])


def _wrap_cpp_test(test: str) -> str:
    if re.search(r"\bmain\s*\(", test):
        return _ensure_trailing_newline(test)
    return "int main()\n{\n" + _indent(test, 4) + "\n    return 0;\n}\n"


def _cpp_source(solution: str, test: str) -> PreparedSource:
    prefix = "#include <bits/stdc++.h>\n#include <cassert>\n#include <cmath>\nusing namespace std;\n\n"
    source = prefix + _ensure_trailing_newline(solution) + "\n" + _wrap_cpp_test(test)
    return PreparedSource(language="cpp", source=source, file_name=FILE_NAMES["cpp"])


def _swift_source(solution: str, test: str) -> PreparedSource:
    combined = _ensure_trailing_newline(solution) + "\n" + _ensure_trailing_newline(test)
    if not re.search(r"\bimport\s+Foundation\b", combined):
        combined = "import Foundation\n\n" + combined
    return PreparedSource(language="swift", source=combined, file_name=FILE_NAMES["swift"])


def _rust_source(solution: str, test: str) -> PreparedSource:
    if not re.search(r"\bfn\s+main\s*\(", test):
        test = "fn main() {\n" + _indent(test, 4) + "\n}\n"
    source = "#![allow(dead_code)]\n\n" + _ensure_trailing_newline(solution) + "\n" + _ensure_trailing_newline(test)
    return PreparedSource(language="rust", source=source, file_name=FILE_NAMES["rust"])


def _merge_into_single_class(source_members: str, class_source: str) -> str | None:
    match = re.match(
        r"^\s*((?:public\s+)?class\s+\w+\s*\{)([\s\S]*)(\}\s*)$",
        class_source.strip(),
    )
    if not match:
        return None
    header, body, closing = match.groups()
    merged_body = source_members.strip()
    if body.strip():
        merged_body = merged_body + "\n\n" + body.strip() if merged_body else body.strip()
    return header + "\n" + _indent(merged_body, 4) + "\n" + closing.strip() + "\n"


def _find_public_class(source: str) -> str | None:
    match = re.search(r"\bpublic\s+class\s+(\w+)\b", source)
    return match.group(1) if match else None


def _find_java_main_class(source: str) -> str | None:
    class_matches = list(re.finditer(r"(?:public\s+)?class\s+(\w+)\b", source))
    if not class_matches:
        return "Main" if re.search(r"\bstatic\s+void\s+main\s*\(", source) else None

    for index, match in enumerate(class_matches):
        end = class_matches[index + 1].start() if index + 1 < len(class_matches) else len(source)
        class_chunk = source[match.start():end]
        if re.search(r"\bstatic\s+void\s+main\s*\(", class_chunk):
            return match.group(1)

    return None


def _java_source(solution: str, test: str) -> PreparedSource:
    solution_has_class = bool(re.search(r"\bclass\s+\w+\b", solution))
    public_class_name = _find_public_class(solution)

    if not solution_has_class:
        if not re.search(r"\bclass\s+\w+\b", test):
            test = _trim_trailing_braces(test)
        merged = _merge_into_single_class(solution, test)
        if merged:
            source = merged
        else:
            if re.search(r"\bstatic\s+void\s+main\s*\(", test):
                test_members = test
            else:
                test_members = "public static void main(String[] args) throws Exception {\n" + _indent(test, 8) + "\n    }"
            body = solution.strip()
            if body:
                body += "\n\n"
            body += test_members.strip()
            source = "public class Main {\n" + _indent(body, 4) + "\n}\n"
    else:
        if re.search(r"\bclass\s+\w+\b", test):
            source = _ensure_trailing_newline(solution) + "\n" + _ensure_trailing_newline(test)
        else:
            test = _trim_trailing_braces(test)
            wrapper_class = "class Main" if public_class_name else "public class Main"
            if re.search(r"\bstatic\s+void\s+main\s*\(", test):
                main_members = test
            else:
                main_members = "public static void main(String[] args) throws Exception {\n" + _indent(test, 8) + "\n    }"
            source = (
                _ensure_trailing_newline(solution)
                + "\n"
                + wrapper_class
                + " {\n"
                + _indent(main_members.strip(), 4)
                + "\n}\n"
            )

    file_stem = _find_public_class(source) or "Main"
    main_class = _find_java_main_class(source) or "Main"
    return PreparedSource(
        language="java",
        source=source,
        file_name=f"{file_stem}.java",
        main_class=main_class,
    )


def _csharp_source(solution: str, test: str) -> PreparedSource:
    prefix = "using System;\nusing System.Collections.Generic;\nusing System.Diagnostics;\nusing System.Linq;\n\n"
    solution_has_container = bool(re.search(r"\b(?:class|namespace|record|struct)\s+\w+\b", solution))

    if not solution_has_container:
        if not re.search(r"\b(?:class|namespace|record|struct)\s+\w+\b", test):
            test = _trim_trailing_braces(test)
        merged = _merge_into_single_class(solution, test)
        if merged:
            source = prefix + merged
        else:
            if re.search(r"\bstatic\s+void\s+Main\s*\(", test):
                test_members = test
            else:
                test_members = "public static void Main(string[] args) {\n" + _indent(test, 8) + "\n    }"
            body = solution.strip()
            if body:
                body += "\n\n"
            body += test_members.strip()
            source = prefix + "public class Program {\n" + _indent(body, 4) + "\n}\n"
    else:
        if re.search(r"\b(?:class|namespace|record|struct)\s+\w+\b", test):
            source = prefix + _ensure_trailing_newline(solution) + "\n" + _ensure_trailing_newline(test)
        else:
            test = _trim_trailing_braces(test)
            if re.search(r"\bstatic\s+void\s+Main\s*\(", test):
                program_body = test.strip()
            else:
                program_body = "public static void Main(string[] args) {\n" + _indent(test, 8) + "\n    }"
            source = (
                prefix
                + _ensure_trailing_newline(solution)
                + "\npublic class Program {\n"
                + _indent(program_body, 4)
                + "\n}\n"
            )

    return PreparedSource(
        language="csharp",
        source=source,
        file_name=FILE_NAMES["csharp"],
        extra_files={"App.csproj": CSPROJ},
    )


def _php_source(solution: str, test: str) -> PreparedSource:
    combined = _ensure_trailing_newline(solution) + "\n" + _ensure_trailing_newline(test)
    if not combined.lstrip().startswith("<?php"):
        combined = "<?php\n\n" + combined
    return PreparedSource(language="php", source=combined, file_name=FILE_NAMES["php"])


def _typescript_source(solution: str, test: str) -> PreparedSource:
    source = _ensure_trailing_newline(solution) + "\n" + _ensure_trailing_newline(test)
    return PreparedSource(language="typescript", source=source, file_name=FILE_NAMES["typescript"])


def _shell_source(solution: str, test: str) -> PreparedSource:
    combined = _ensure_trailing_newline(solution) + "\n" + _ensure_trailing_newline(test)
    if combined.lstrip().startswith("#!"):
        lines = combined.splitlines()
        shebang = lines[0]
        body = "\n".join(lines[1:]).lstrip("\n")
        if re.search(r"^\s*set\s+-[^\n]*e", body, re.MULTILINE):
            source = combined
        elif body:
            source = shebang + "\nset -e\n\n" + body + "\n"
        else:
            source = shebang + "\nset -e\n"
    else:
        source = "#!/usr/bin/env bash\nset -e\n\n" + combined
    return PreparedSource(language="shell", source=source, file_name=FILE_NAMES["shell"])


def _prepare_source(language: str, solution: str, test: str) -> PreparedSource:
    if language == "cpp":
        return _cpp_source(solution, test)
    if language == "python":
        return _python_source(solution, test)
    if language == "swift":
        return _swift_source(solution, test)
    if language == "rust":
        return _rust_source(solution, test)
    if language == "csharp":
        return _csharp_source(solution, test)
    if language == "java":
        return _java_source(solution, test)
    if language == "php":
        return _php_source(solution, test)
    if language == "typescript":
        return _typescript_source(solution, test)
    if language == "shell":
        return _shell_source(solution, test)
    raise ValueError(f"Unsupported language: {language}")


def _write_prepared_source(work_dir: Path, prepared: PreparedSource) -> Path:
    source_path = work_dir / prepared.file_name
    source_path.write_text(prepared.source, encoding="utf-8")
    for relative_name, content in prepared.extra_files.items():
        (work_dir / relative_name).write_text(content, encoding="utf-8")
    return source_path


def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def _docker_available() -> bool:
    if not _command_exists("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _docker_image_available(image_name: str) -> bool:
    result = _run_process(
        ["docker", "image", "inspect", image_name],
        cwd=None,
        timeout=10,
    )
    return result["success"]


def _docker_container_status(container_name: str) -> str | None:
    result = _run_process(
        ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
        cwd=None,
        timeout=10,
    )
    if not result["success"]:
        return None
    status = result["stdout"].strip()
    return status or None


def start_docker_container(
    image_name: str | None = None,
    container_name: str | None = None,
    timeout: int = 120,
) -> dict:
    image_name = image_name or _docker_image()
    container_name = container_name or _docker_container()

    if not _docker_available():
        return {
            "success": False,
            "stdout": "",
            "stderr": "Docker is not available.",
            "returncode": None,
            "execution_time": 0.0,
        }

    status = _docker_container_status(container_name)
    if status == "running":
        _run_process(
            ["docker", "exec", container_name, "mkdir", "-p", DOCKER_JOBS_DIR],
            cwd=None,
            timeout=timeout,
        )
        return {
            "success": True,
            "stdout": f"{container_name} is already running.\n",
            "stderr": "",
            "returncode": 0,
            "execution_time": 0.0,
        }

    if status in {"created", "exited"}:
        result = _run_process(["docker", "start", container_name], cwd=None, timeout=timeout)
        if result["success"]:
            _run_process(
                ["docker", "exec", container_name, "mkdir", "-p", DOCKER_JOBS_DIR],
                cwd=None,
                timeout=timeout,
            )
        return result

    if not _docker_image_available(image_name):
        return {
            "success": False,
            "stdout": "",
            "stderr": (
                f"Persistent Docker image '{image_name}' does not exist. "
                f"Build it first with: docker build -t {image_name} -f {THIS_DIR / 'Dockerfile'} {THIS_DIR}"
            ),
            "returncode": None,
            "execution_time": 0.0,
        }

    result = _run_process(
        [
            "docker",
            "run",
            "-d",
            "--init",
            "--name",
            container_name,
            "--memory", "2g",
            "--pids-limit", "256",
            "-w",
            DOCKER_WORKDIR,
            image_name,
            "tail",
            "-f",
            "/dev/null",
        ],
        cwd=None,
        timeout=timeout,
    )
    if result["success"]:
        _run_process(
            ["docker", "exec", container_name, "mkdir", "-p", DOCKER_JOBS_DIR],
            cwd=None,
            timeout=timeout,
        )
    return result


def stop_docker_container(container_name: str | None = None, timeout: int = 60) -> dict:
    container_name = container_name or _docker_container()
    status = _docker_container_status(container_name)
    if status is None:
        return {
            "success": True,
            "stdout": f"{container_name} does not exist.\n",
            "stderr": "",
            "returncode": 0,
            "execution_time": 0.0,
        }
    if status != "running":
        return {
            "success": True,
            "stdout": f"{container_name} is already stopped.\n",
            "stderr": "",
            "returncode": 0,
            "execution_time": 0.0,
        }
    return _run_process(["docker", "stop", container_name], cwd=None, timeout=timeout)


def _ensure_docker_container(timeout: int = 120) -> dict:
    return start_docker_container(
        image_name=_docker_image(),
        container_name=_docker_container(),
        timeout=timeout,
    )


def _python_command() -> str | None:
    for command in ("python3", "python"):
        if _command_exists(command):
            return command
    return None


def _typescript_local_runner(source: str) -> tuple[str, list[str]] | None:
    if _command_exists("tsx"):
        return "tsx", ["tsx", FILE_NAMES["typescript"]]
    if _command_exists("ts-node"):
        return "ts-node", ["ts-node", FILE_NAMES["typescript"]]
    if _command_exists("bun"):
        return "bun", ["bun", FILE_NAMES["typescript"]]
    if _command_exists("deno"):
        return "deno", ["deno", "run", "--quiet", FILE_NAMES["typescript"]]
    if _command_exists("tsc") and _command_exists("node"):
        return "tsc", ["node", str(Path("dist") / "main.js")]

    requires_typescript_runtime = bool(
        re.search(r"\binterface\s+\w+|:\s*(?:number|string|boolean|void|unknown|any|never)\b", source)
    )
    if not requires_typescript_runtime and _command_exists("node"):
        return "node-js-compatible", ["node", "main.js"]
    return None


def _run_process(command: list[str], cwd: Path | None, timeout: int) -> dict:
    start = time.time()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group so we can kill the whole tree
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            return {
                "success": False,
                "stdout": "",
                "stderr": "Time Limit Exceeded",
                "returncode": None,
                "execution_time": float(timeout),
            }
        elapsed = time.time() - start
        return {
            "success": process.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": process.returncode,
            "execution_time": elapsed,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": None,
            "execution_time": time.time() - start,
        }


def _local_commands(language: str, prepared: PreparedSource) -> tuple[list[str] | None, list[str] | None, str | None]:
    if language == "cpp" and _command_exists("g++"):
        return (
            ["g++", "-O2", "-std=c++17", prepared.file_name, "-o", "main"],
            ["./main"],
            None,
        )
    if language == "python":
        python_cmd = _python_command()
        if python_cmd:
            return (None, [python_cmd, prepared.file_name], None)
    if language == "swift" and _command_exists("swiftc"):
        return (
            ["swiftc", prepared.file_name, "-o", "main"],
            ["./main"],
            None,
        )
    if language == "rust" and _command_exists("rustc"):
        return (
            ["rustc", "--edition=2021", prepared.file_name, "-o", "main"],
            ["./main"],
            None,
        )
    if language == "csharp":
        if _command_exists("mcs") and _command_exists("mono"):
            return (
                ["mcs", "-define:DEBUG", prepared.file_name, "-out:program.exe"],
                ["mono", "program.exe"],
                None,
            )
        if _command_exists("dotnet"):
            return (
                ["dotnet", "build", "App.csproj", "-nologo", "/clp:ErrorsOnly"],
                ["dotnet", "run", "--no-build", "--project", "App.csproj", "-nologo"],
                None,
            )
    if language == "java" and _command_exists("javac") and _command_exists("java"):
        return (
            ["javac", prepared.file_name],
            ["java", "-ea", "-cp", ".", prepared.main_class or "Main"],
            None,
        )
    if language == "php" and _command_exists("php"):
        return (
            None,
            ["php", "-d", "zend.assertions=1", "-d", "assert.exception=1", prepared.file_name],
            None,
        )
    if language == "typescript":
        runner = _typescript_local_runner(prepared.source)
        if runner is not None:
            runner_name, run_command = runner
            if runner_name == "tsc":
                return (
                    [
                        "tsc",
                        prepared.file_name,
                        "--target",
                        "es2020",
                        "--module",
                        "commonjs",
                        "--outDir",
                        "dist",
                    ],
                    run_command,
                    None,
                )
            if runner_name == "node-js-compatible":
                return (
                    None,
                    ["node", prepared.file_name],
                    None,
                )
            return (None, run_command, None)
    if language == "shell":
        shell_cmd = "bash" if _command_exists("bash") else "sh" if _command_exists("sh") else None
        if shell_cmd:
            # Set -u relative to current user process count so $(…) subshells work but
            # fork bombs are capped. 100 extra slots covers parallel workers + test subshells.
            try:
                uid = os.getuid()
                # ulimit -u counts threads (LWPs) on Linux, not just processes
                ps_result = subprocess.run(
                    ["ps", "-u", str(uid), "-L", "--no-header", "-o", "tid"],
                    capture_output=True, text=True, check=False, timeout=5,
                )
                current_threads = len(ps_result.stdout.strip().splitlines())
            except Exception:
                current_threads = 300
            proc_limit = current_threads + 100
            return (None, [shell_cmd, "-c", f"ulimit -v 524288 -u {proc_limit}; exec {shell_cmd} {prepared.file_name}"], None)
    return (None, None, f"No local toolchain found for {language}")


def _docker_commands(language: str, prepared: PreparedSource) -> tuple[list[str] | None, list[str]]:
    if language == "cpp":
        return (
            ["sh", "-lc", f"g++ -O2 -std=c++17 {prepared.file_name} -o main"],
            ["sh", "-lc", "./main"],
        )
    if language == "python":
        return (None, ["sh", "-lc", f"python3 {prepared.file_name}"])
    if language == "swift":
        return (
            ["sh", "-lc", f"swiftc {prepared.file_name} -o main"],
            ["sh", "-lc", "./main"],
        )
    if language == "rust":
        return (
            ["sh", "-lc", f"rustc --edition=2021 {prepared.file_name} -o main"],
            ["sh", "-lc", "./main"],
        )
    if language == "csharp":
        return (
            ["sh", "-lc", "mcs -define:DEBUG Program.cs -out:program.exe"],
            ["sh", "-lc", "mono program.exe"],
        )
    if language == "java":
        return (
            ["sh", "-lc", f"javac {prepared.file_name}"],
            ["sh", "-lc", f"java -ea -cp . {prepared.main_class or 'Main'}"],
        )
    if language == "php":
        return (
            None,
            ["sh", "-lc", f"php -d zend.assertions=1 -d assert.exception=1 {prepared.file_name}"],
        )
    if language == "typescript":
        return (
            None,
            ["sh", "-lc", f"bun {prepared.file_name}"],
        )
    if language == "shell":
        return (
            None,
            ["sh", "-lc", f"ulimit -v 524288 -u 64; exec bash {prepared.file_name}"],
        )
    raise ValueError(f"Unsupported language: {language}")


def _run_docker_process(container_command: list[str], container_workdir: str, timeout: int) -> dict:
    docker_command = [
        "docker",
        "exec",
        "-w",
        container_workdir,
        _docker_container(),
        *container_command,
    ]
    return _run_process(docker_command, cwd=None, timeout=timeout)


def _copy_into_docker_container(source_dir: Path, container_workdir: str, timeout: int) -> dict:
    create_result = _run_process(
        ["docker", "exec", _docker_container(), "mkdir", "-p", container_workdir],
        cwd=None,
        timeout=timeout,
    )
    if not create_result["success"]:
        return create_result
    return _run_process(
        ["docker", "cp", f"{source_dir}/.", f"{_docker_container()}:{container_workdir}/"],
        cwd=None,
        timeout=timeout,
    )


def _evaluate_with_local_toolchain(prepared: PreparedSource, timeout: int) -> dict:
    result = _empty_result(prepared.language)

    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        _write_prepared_source(work_dir, prepared)
        compile_command, run_command, error_message = _local_commands(prepared.language, prepared)

        if error_message:
            result["stderr"] = error_message
            return result

        if compile_command is not None:
            compile_result = _run_process(compile_command, cwd=work_dir, timeout=timeout)
            if not compile_result["success"]:
                return _apply_process_result(result, compile_result, use_docker=False)

        run_result = _run_process(run_command or [], cwd=work_dir, timeout=timeout)
        return _apply_process_result(
            result,
            run_result,
            use_docker=False,
            test_passed=run_result["success"],
        )


def _evaluate_with_docker(prepared: PreparedSource, timeout: int) -> dict:
    result = _empty_result(prepared.language)

    ensure_result = _ensure_docker_container(timeout=120)
    if not ensure_result["success"]:
        result["stderr"] = ensure_result["stderr"] or "Unable to start the Docker container."
        return result

    with tempfile.TemporaryDirectory() as temp_dir:
        host_work_dir = Path(temp_dir)
        _write_prepared_source(host_work_dir, prepared)
        compile_command, run_command = _docker_commands(prepared.language, prepared)
        job_id = uuid.uuid4().hex
        container_workdir = f"{DOCKER_JOBS_DIR}/{job_id}"

        try:
            copy_result = _copy_into_docker_container(host_work_dir, container_workdir, timeout=30)
            if not copy_result["success"]:
                result["stderr"] = copy_result["stderr"] or "Failed to copy files into the Docker container."
                return result

            if compile_command is not None:
                compile_result = _run_docker_process(
                    compile_command,
                    container_workdir=container_workdir,
                    timeout=timeout,
                )
                if not compile_result["success"]:
                    return _apply_process_result(result, compile_result, use_docker=True)

            run_result = _run_docker_process(
                run_command,
                container_workdir=container_workdir,
                timeout=timeout,
            )
            return _apply_process_result(
                result,
                run_result,
                use_docker=True,
                test_passed=run_result["success"],
            )
        finally:
            _run_process(
                ["docker", "exec", _docker_container(), "rm", "-rf", container_workdir],
                cwd=None,
                timeout=15,
            )


def evaluate(language: str, solution: str, test: str) -> dict:
    """
    Evaluate a generated solution snippet against an inline test snippet.

    The evaluator supports:
    `cpp`, `python`, `swift`, `rust`, `csharp`, `java`, `php`,
    `typescript`, and `shell`.
    """
    clean_language = language.strip().lower()
    result = _empty_result(clean_language or None)

    if clean_language not in SUPPORTED_LANGUAGES:
        result["stderr"] = (
            f"Unsupported language '{language}'. "
            f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}."
        )
        return result

    clean_solution = _strip_fenced_code(solution)
    clean_test = _strip_fenced_code(test)

    try:
        prepared = _prepare_source(clean_language, clean_solution, clean_test)
    except Exception as exc:
        result["stderr"] = f"Failed to prepare source: {exc}"
        return result

    _, run_command, error_message = _local_commands(clean_language, prepared)
    if run_command is not None and error_message is None:
        return _evaluate_with_local_toolchain(prepared, DEFAULT_TIMEOUT_SECONDS)

    if not _docker_available():
        result["stderr"] = error_message or f"No local toolchain or Docker runtime is available for {clean_language}."
        return result

    return _evaluate_with_docker(prepared, DEFAULT_TIMEOUT_SECONDS)


__all__ = [
    "SUPPORTED_LANGUAGES",
    "evaluate",
    "start_docker_container",
    "stop_docker_container",
]
