#!/bin/bash

set -e

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export HF_XET_CACHE="$PWD/.cache/huggingface/xet"

cd "$(dirname "$0")"
exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 "$@"
