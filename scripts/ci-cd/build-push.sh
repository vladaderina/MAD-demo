#!/bin/bash
set -e

IMAGE_TAGS=$1

echo "Setting up Docker Buildx"
docker buildx create --use

echo "Logging in to GitHub Container Registry"
echo "${{ secrets.CICD_PAT }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin

echo "Building and pushing image with tags: $IMAGE_TAGS"

IFS=',' read -ra TAGS <<< "$IMAGE_TAGS"
TAG_ARGS=""
for tag in "${TAGS[@]}"; do
    TAG_ARGS+=" -t $tag"
done

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    $TAG_ARGS \
    --push \
    --cache-from type=gha \
    --cache-to type=gha,mode=max \
    .

echo "Image pushed successfully"