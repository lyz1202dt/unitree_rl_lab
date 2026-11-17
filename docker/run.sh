#!/usr/bin/env bash
set -euo pipefail

docker_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! docker image inspect unitree-rl-lab:latest >/dev/null 2>&1; then
    echo "Image unitree-rl-lab:latest is missing. Run ${docker_dir}/build.sh first." >&2
    exit 1
fi

exec docker compose \
    --env-file "${docker_dir}/.env.base" \
    -f "${docker_dir}/docker-compose.yaml" \
    run --rm --no-deps unitree-rl-lab "$@"
