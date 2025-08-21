#!/bin/bash
set -e

IMAGE_TAGS=$1

echo "Building and pushing image with tags: $IMAGE_TAGS"

IFS=',' read -ra TAGS <<< "$IMAGE_TAGS"
TAG_ARGS=""
for tag in "${TAGS[@]}"; do
    TAG_ARGS+=" -t $tag"
done

docker buildx build \
    --platform linux/amd64 \
    $TAG_ARGS \
    --push
    .

echo "Image pushed successfully"