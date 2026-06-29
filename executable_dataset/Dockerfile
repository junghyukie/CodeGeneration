FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV SWIFT_VERSION=6.3.1
ENV SWIFT_PLATFORM=ubuntu24.04
ENV SWIFT_BRANCH=swift-6.3.1-release
ENV SWIFT_DIR=swift-6.3.1-RELEASE
ENV PATH="/opt/swift/usr/bin:/root/.cargo/bin:/root/.bun/bin:${PATH}"
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
ENV DOTNET_ROOT="/usr/lib/dotnet"

RUN apt-get update && apt-get install -y --no-install-recommends \
    binutils \
    build-essential \
    ca-certificates \
    ca-certificates-java \
    clang \
    curl \
    gcc \
    g++ \
    libcurl4-openssl-dev \
    libicu-dev \
    libncurses-dev \
    libssl-dev \
    libxml2-dev \
    libz3-dev \
    lld \
    mono-devel \
    openjdk-17-jdk \
    php-cli \
    pkg-config \
    python3 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o ${SWIFT_DIR}-${SWIFT_PLATFORM}.tar.gz \
        https://download.swift.org/${SWIFT_BRANCH}/ubuntu2404/${SWIFT_DIR}/${SWIFT_DIR}-${SWIFT_PLATFORM}.tar.gz \
    && mkdir -p /opt/swift \
    && tar -xzf ${SWIFT_DIR}-${SWIFT_PLATFORM}.tar.gz --strip-components=1 -C /opt/swift \
    && rm ${SWIFT_DIR}-${SWIFT_PLATFORM}.tar.gz

RUN curl -fsSL -o packages-microsoft-prod.deb \
        https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends dotnet-sdk-8.0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --profile minimal
RUN curl -fsSL https://bun.sh/install | bash
    
WORKDIR /workspace
RUN mkdir -p /workspace/jobs

CMD ["tail", "-f", "/dev/null"]
