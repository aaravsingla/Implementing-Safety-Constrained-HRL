#!/usr/bin/env bash
# usage: LOG_DIR=logs/v2 _run_one.sh <script.py> [args...]  (args must include --name NAME); logs to $LOG_DIR/NAME.log
cd "$(dirname "$0")/.."
name=$(echo "$@" | sed -E 's/.*--name ([^ ]+).*/\1/')
LOG_DIR=${LOG_DIR:-logs/v2}
mkdir -p "$LOG_DIR"
exec .venv/bin/python -W ignore "$@" > "$LOG_DIR/$name.log" 2>&1
