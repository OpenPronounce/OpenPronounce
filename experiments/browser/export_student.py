"""Export the distilled student to ONNX: fp32, int8 (dynamic, MatMul only), fp16.

    HF_HOME=~/.cache/huggingface PYTHONPATH=/mnt/t7/openpronounce-browser/pylib \
        .venv-exp/bin/python experiments/browser/export_student.py --ckpt /mnt/t7/openpronounce-browser/student/final.pt

Outputs, same order as phone.onnx so extract_onnx.py reads it unchanged: ``logits`` (1, T, 392)
phone logits, ``hidden20`` (1, T, 1024) projection of the student layer meant to stand for the
teacher's layer 20, then ``asr_logits`` (1, T, 32) char CTC logits of the 960h vocabulary.
"""

import argparse
import os

import torch

from distill import Student
from export_onnx import OUT, convert_fp16, quantize_int8


class Wrapper(torch.nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, x):
        phone, asr, hidden = self.student(x)
        return phone, hidden, asr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--name", default="student")
    args = parser.parse_args()
    student = Student()
    student.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    student.eval()
    path = os.path.join(OUT, f"{args.name}.fp32.onnx")
    names = ["logits", "hidden20", "asr_logits"]
    with torch.no_grad():
        torch.onnx.export(Wrapper(student), (torch.randn(1, 64000),), path, input_names=["input_values"], output_names=names,
                          dynamic_axes={"input_values": {1: "samples"}, **{n: {1: "frames"} for n in names}},
                          opset_version=17, dynamo=False)
    print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB")
    quantize_int8(path, os.path.join(OUT, f"{args.name}.int8.onnx"))
    convert_fp16(path, os.path.join(OUT, f"{args.name}.fp16.onnx"))


if __name__ == "__main__":
    main()
