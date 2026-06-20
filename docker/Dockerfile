# syntax=docker/dockerfile:1

# CUDA_VERSION can be overridden at build time:
#   docker build --build-arg CUDA_VERSION=13.2.0 ...
ARG CUDA_VERSION=12.9.0

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    zsh \
    vim \
    sudo \
    ca-certificates \
    gnupg \
    locales \
    python3 \
    python3-pip \
    python3-venv \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# ── uv (Astral) ───────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/zsh \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# ── Persistent shell history ──────────────────────────────────────────────────
RUN mkdir /commandhistory && chown ${USERNAME}:${USERNAME} /commandhistory

USER ${USERNAME}

# ── ty (Astral type checker) ──────────────────────────────────────────────────
# Installed as a global uv tool so `ty` is available on PATH everywhere
RUN uv tool install ty

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
