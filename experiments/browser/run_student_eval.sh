#!/bin/bash
# Extract the ONNX student on both splits, then evaluate it with the shipped models and refitted ones.
#   bash experiments/browser/run_student_eval.sh <onnx file in /mnt/t7/openpronounce-browser/onnx> <variant>
set -e
cd "$(dirname "$0")"
PY=../../.venv-exp/bin/python
LOG=/mnt/t7/openpronounce-browser
for s in test train; do $PY extract_onnx.py --model "$1" --variant "$2" --split $s --threads ${THREADS:-8}; done
$PY eval_variant.py --variant "$2" > $LOG/eval_$2_norefit.log 2>&1
$PY eval_variant.py --variant "$2" --refit > $LOG/eval_$2_refit.log 2>&1
