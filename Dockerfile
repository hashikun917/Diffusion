FROM nvidia/cuda:12.4.0-base-ubuntu22.04
# GPUがいらないなら ubuntu:22.04 でもいい

# デフォルトのpythonのバージョンを指定
ARG PYTHON=3.11

# Install base packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON} python3-pip python3-venv \
        build-essential git curl ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

# uvのパスを通す
ENV PATH="/root/.local/bin:$PATH"

# Install dependencies
WORKDIR /workspace
COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml --system 
# RUN uv sync でもいい。ただし.venvが自動で作られてしまう