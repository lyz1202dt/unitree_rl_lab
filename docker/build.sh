#!/usr/bin/env bash
set -euo pipefail

docker_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${docker_dir}/.." && pwd)"
source "${docker_dir}/.env.base"

if (( $# > 1 )); then
    echo "Usage: $0 [--proxy]" >&2
    exit 2
fi

build_options=()
case "${1:-}" in
    "") ;;
    --proxy)
        proxy_url="http://127.0.0.1:7890"
        build_options+=(--network host --allow network.host)
        for name in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
            build_options+=(--build-arg "${name}=${proxy_url}")
        done
        ;;
    -h|--help)
        echo "Usage: $0 [--proxy]"
        exit 0
        ;;
    *)
        echo "Usage: $0 [--proxy]" >&2
        exit 2
        ;;
esac

exec docker buildx build \
    "${build_options[@]}" \
    --load \
    --tag unitree-rl-lab:latest \
    --file "${docker_dir}/Dockerfile" \
    --build-arg "ISAACLAB_BASE_IMAGE_ARG=${ISAACLAB_BASE_IMAGE}" \
    --build-arg "UNITREE_RL_LAB_PATH_ARG=${DOCKER_UNITREE_RL_LAB_PATH}" \
    "${project_dir}"
