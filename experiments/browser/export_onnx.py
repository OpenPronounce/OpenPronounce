"""Export the two English wav2vec2 recognizers to ONNX (fp32), then int8 (dynamic) and fp16.

    PYTHONPATH=/mnt/t7/openpronounce-browser/pylib HF_HOME=~/.cache/huggingface \
        .venv-exp/bin/python experiments/browser/export_onnx.py [--only phone|asr] [--out DIR]

The graphs take the *normalized* waveform (zero mean, unit variance, what the
Wav2Vec2FeatureExtractor does with do_normalize=True), shape (1, samples), float32.

phone.onnx (facebook/wav2vec2-lv-60-espeak-cv-ft): outputs ``logits`` (1, T, 392) and
``hidden20`` (1, T, 1024), hidden_states[20] as used by the word model.
asr.onnx (facebook/wav2vec2-large-960h): outputs ``logits`` (1, T, 32) and ``last_hidden``
(1, T, 1024), what speech.transcribe and speech.extract_embeddings read.
"""

import argparse
import os
import time

import torch

OUT = "/mnt/t7/openpronounce-browser/onnx"
MODELS = {"phone": "facebook/wav2vec2-lv-60-espeak-cv-ft", "asr": "facebook/wav2vec2-large-960h"}


class PhoneWrapper(torch.nn.Module):
    def __init__(self, model, layer=20):
        super().__init__()
        self.model, self.layer = model, layer

    def forward(self, x):
        out = self.model(x, output_hidden_states=True)
        return out.logits, out.hidden_states[self.layer]


class AsrWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        hidden = self.model.wav2vec2(x).last_hidden_state
        return self.model.lm_head(hidden), hidden


def export(kind, out_dir):
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(MODELS[kind], attn_implementation="eager").eval()
    wrapper = PhoneWrapper(model) if kind == "phone" else AsrWrapper(model)
    names = ["logits", "hidden20"] if kind == "phone" else ["logits", "last_hidden"]
    path = os.path.join(out_dir, f"{kind}.fp32.onnx")
    x = torch.randn(1, 16000 * 4)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(wrapper, (x,), path, input_names=["input_values"], output_names=names,
                          dynamic_axes={"input_values": {1: "samples"}, names[0]: {1: "frames"}, names[1]: {1: "frames"}},
                          opset_version=17, dynamo=False, do_constant_folding=True)
    print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t0:.0f}s")
    del model, wrapper
    return path


def quantize_int8(fp32, out, per_channel=False):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32.replace(".fp32.onnx", ".pre.onnx")
    quant_pre_process(fp32, pre, skip_symbolic_shape=True)
    # MatMul only: the conv feature encoder and the positional conv stay fp32 (ConvInteger is
    # slow or missing in onnxruntime-web and hurts accuracy for little size gain).
    quantize_dynamic(pre, out, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
                     per_channel=per_channel, reduce_range=False)
    os.remove(pre)
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB")


def convert_fp16(fp32, out):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(fp32)
    model = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model, out)
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(MODELS))
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--per-channel-only", action="store_true", help="only write <kind>.int8pc.onnx (per-channel int8)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for kind in [args.only] if args.only else list(MODELS):
        fp32 = os.path.join(args.out, f"{kind}.fp32.onnx")
        if not args.skip_export or not os.path.exists(fp32):
            fp32 = export(kind, args.out)
        if args.per_channel_only:
            quantize_int8(fp32, os.path.join(args.out, f"{kind}.int8pc.onnx"), per_channel=True)
            continue
        quantize_int8(fp32, os.path.join(args.out, f"{kind}.int8.onnx"))
        convert_fp16(fp32, os.path.join(args.out, f"{kind}.fp16.onnx"))


if __name__ == "__main__":
    main()
