"""Write the distilled student as a transformers folder the library can load (and the Hub can host).

    HF_HOME=~/.cache/huggingface .venv-exp/bin/python experiments/browser/export_hub.py \
        --ckpt /mnt/t7/openpronounce-browser/student/final.pt

Layout of the folder (``--out``):
  config.json, model.safetensors      Wav2Vec2ForCTC: wav2vec2-base body + the phone head (392 tokens)
  preprocessor_config.json, vocab.json, tokenizer_config.json, special_tokens_map.json   (from the teacher)
  hidden_projection.safetensors       768 -> 1024 linear map of hidden_states[10], the stand-in for
                                      the teacher's hidden_states[20] (``weight``, ``bias``)
  char_head.safetensors               768 -> 32 char CTC head (wav2vec2-large-960h vocabulary), not used
                                      by the library, kept for the browser graph
  onnx/model.onnx, onnx/model_int8.onnx   the graphs of export_student.py (outputs: logits, hidden20, asr_logits)

``config.json`` carries an ``openpronounce`` block that openpronounce.phones reads.
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch

from distill import HID_LAYER, PHONE, TEACHER_LAYER, Student

OUT = "/mnt/t7/openpronounce-browser/hub/wav2vec2-base-phones-en"
ONNX = "/mnt/t7/openpronounce-browser/onnx"


def _teacher_file(name):
    """A file of the phone teacher, from the cache of $HF_HOME or from the one on /mnt/t7."""
    from transformers.utils import cached_file

    path = cached_file(PHONE, name, _raise_exceptions_for_missing_entries=False)
    if path is None:
        import glob

        found = glob.glob(f"/mnt/t7/ai-models/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/snapshots/*/{name}")
        path = found[0]
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args()
    from safetensors.torch import save_file
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
    student = Student()
    student.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    student.eval()

    teacher = Wav2Vec2Config.from_pretrained(PHONE)
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
    config.architectures = ["Wav2Vec2ForCTC"]
    config.vocab_size = teacher.vocab_size
    for key in ("pad_token_id", "bos_token_id", "eos_token_id", "ctc_loss_reduction", "ctc_zero_infinity"):
        setattr(config, key, getattr(teacher, key))
    config.mask_time_prob = 0.0
    config.layerdrop = 0.0
    config.openpronounce = {
        "models": "student",
        "hidden_projection": {"file": "hidden_projection.safetensors", "source_layer": HID_LAYER,
                              "layer": TEACHER_LAYER, "teacher": PHONE},
        "distilled_from": [PHONE, "facebook/wav2vec2-large-960h"],
    }
    model = Wav2Vec2ForCTC(config)
    state = {f"wav2vec2.{k}": v for k, v in student.body.state_dict().items()}
    state["lm_head.weight"] = student.phone_head.weight.detach().clone()
    state["lm_head.bias"] = student.phone_head.bias.detach().clone()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    save_file({"weight": student.hid_proj.weight.detach().contiguous(), "bias": student.hid_proj.bias.detach().contiguous()},
              os.path.join(args.out, "hidden_projection.safetensors"))
    save_file({"weight": student.asr_head.weight.detach().contiguous(), "bias": student.asr_head.bias.detach().contiguous()},
              os.path.join(args.out, "char_head.safetensors"))
    for name in ("preprocessor_config.json", "vocab.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(_teacher_file(name), os.path.join(args.out, name))

    # same outputs as the training module, on a fixed random input
    x = torch.randn(1, 48000, generator=torch.Generator().manual_seed(0))
    x = (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)
    with torch.no_grad():
        phone, _, hidden = student(x)
        out = model(x, output_hidden_states=True)
        projected = student.hid_proj(out.hidden_states[HID_LAYER])
    print("phone logits max |diff|", float((phone - out.logits).abs().max()),
          "projection max |diff|", float((hidden - projected).abs().max()))

    if not args.no_onnx:
        os.makedirs(os.path.join(args.out, "onnx"), exist_ok=True)
        for source, target in (("student.fp32.onnx", "model.onnx"), ("student.int8.onnx", "model_int8.onnx")):
            shutil.copy(os.path.join(ONNX, source), os.path.join(args.out, "onnx", target))
    for name in sorted(os.listdir(args.out)):
        path = os.path.join(args.out, name)
        if os.path.isfile(path):
            print(f"{name:34s} {os.path.getsize(path) / 1e6:9.1f} MB")


if __name__ == "__main__":
    main()
