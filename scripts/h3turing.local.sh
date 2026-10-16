#!/usr/bin/env bash
# 包装项目自带 CLI（h3turing 包是纯标准库，用系统 python3 即可，不必装进任何 venv）。
# 用法与 h3turing 完全一致：
#   scripts/h3turing.local.sh doctor --models-dir /home/a/ai-work/models
#   scripts/h3turing.local.sh demo   --tier t2v-final --out ./h3_demo_output
#   scripts/h3turing.local.sh tiers
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE/../python/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PY:-python3}" -c 'import sys; from h3turing.cli import main; sys.exit(main(sys.argv[1:]))' "$@"