#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Some Isaac Lab Conda installations need CUDA's CUPTI directory to import torch.
for cupti in /usr/local/cuda*/extras/CUPTI/lib64/libcupti.so.12; do
    if [[ -f "$cupti" ]]; then
        export LD_LIBRARY_PATH="$(dirname "$cupti"):${LD_LIBRARY_PATH:-}"
        break
    fi
done

if ! command -v python >/dev/null 2>&1; then
    echo "Activate env_robotlab (or another environment with mujoco, numpy and torch) first." >&2
    exit 1
fi

exec python "$script_dir/simulate.py" "$@"
