#!/bin/zsh
# Build the isolated venv for the harness.
#
# Keep benchmark dependencies isolated from unrelated local projects.
set -eu
cd "$(dirname "$0")"
PY="${RLM_PYTHON:-$(command -v python3.11 || true)}"
[ -n "$PY" ] || {
  echo "Python 3.11 is required for the recorded evaluation environment" >&2
  exit 2
}
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
"$PY" -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -e ".[dev,comparison]"
./.venv/bin/python -c "import rlm, importlib.metadata as m; print('rlms', m.version('rlms'))"
