#!/bin/bash
# Build the riskzero-clean devcontainer image if it doesn't already exist.
#
# Usage:
#   ./docker/build.sh                        # CUDA 12.9.0 (default)
#   ./docker/build.sh 13.2.0                 # CUDA 13.2.0 (Blackwell)
#   ./docker/build.sh 12.9.0 --no-cache      # Force rebuild

set -euo pipefail
export DOCKER_BUILDKIT=1
export PROGRESS_NO_TRUNC=1

CUDA_VERSION="${1:-12.9.0}"
shift || true  # consume first arg if present; pass remaining args to docker build

IMAGE="riskzero-clean:latest"

if docker image inspect "$IMAGE" &>/dev/null; then
    echo "Image '${IMAGE}' already exists — skipping build."
    echo "  CUDA version baked in: $(docker inspect --format '{{.Config.Labels.cuda_version}}' "$IMAGE" 2>/dev/null || echo 'unknown')"
    echo "  To rebuild: docker rmi ${IMAGE} && ./docker/build.sh [cuda_version]"
    exit 0
fi

echo "Building '${IMAGE}' with CUDA ${CUDA_VERSION}..."
docker build \
    --tag "${IMAGE}" \
    --label "cuda_version=${CUDA_VERSION}" \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "USERNAME=vscode" \
    --build-arg "USER_UID=1000" \
    --build-arg "USER_GID=1000" \
    -f docker/Dockerfile \
    "$@" \
    .

echo "Done. Image '${IMAGE}' is ready."
