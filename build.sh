#!/usr/bin/env bash
set -euo pipefail

REGISTRY=${REGISTRY:-docker.io/myleslandais}
TAG=${TAG:-latest}
IMAGE="$REGISTRY/vllm-omni-tts:$TAG"

echo "Building $IMAGE"
docker build -t "$IMAGE" infra/vllm-omni/

echo "Pushing $IMAGE"
docker push "$IMAGE"

echo "Done: $IMAGE"
