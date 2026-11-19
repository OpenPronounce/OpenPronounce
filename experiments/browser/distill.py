"""Distil the two English wav2vec2-large teachers into one wav2vec2-base student (GPU).

Targets, computed on the fly by the teachers (fp16, no grad) on the same audio crop:
  - lv-60-espeak phone logits  -> student phone head (768 -> 392), frame KL
  - lv-60-espeak hidden_states[20] -> projection of student hidden_states[HID_LAYER] (768 -> 1024),
    L1 (scaled) + (1 - cosine)
  - 960h char CTC logits       -> student asr head (768 -> 32), frame KL
Both teachers and the student share the conv feature encoder geometry (stride 320), so frames match.

    HF_HOME=~/.cache/huggingface ~/.venvs/openpronounce-gpu/bin/python experiments/browser/distill.py \
        --hours 2.5 --out /mnt/t7/openpronounce-browser/student
Audio: predecode.py output (Common Voice 27 en subset on /mnt/t7 + speechocean762 train).
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

AUDIO = "/mnt/t7/openpronounce-browser/distill_audio"
PHONE = "facebook/wav2vec2-lv-60-espeak-cv-ft"
ASR = "facebook/wav2vec2-large-960h"
BASE = "facebook/wav2vec2-base"
HID_LAYER = 10  # student layer (of 12) projected onto teacher layer 20 (of 24)
TEACHER_LAYER = 20


class Student(torch.nn.Module):
    def __init__(self, base=BASE, hid_layer=HID_LAYER):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        config = Wav2Vec2Config.from_pretrained(base)
        config.mask_time_prob = 0.0
        config.layerdrop = 0.0
        self.body = Wav2Vec2Model.from_pretrained(base, config=config)
        self.hid_layer = hid_layer
        dim = config.hidden_size
        self.phone_head = torch.nn.Linear(dim, 392)
        self.asr_head = torch.nn.Linear(dim, 32)
        self.hid_proj = torch.nn.Linear(dim, 1024)

    def forward(self, x):
        out = self.body(x, output_hidden_states=True)
        last = out.last_hidden_state
        return self.phone_head(last), self.asr_head(last), self.hid_proj(out.hidden_states[self.hid_layer])


def normalize(batch):
    mean = batch.mean(dim=1, keepdim=True)
    var = batch.var(dim=1, keepdim=True, unbiased=False)
    return (batch - mean) / torch.sqrt(var + 1e-7)


class Batches:
    """Length-bucketed batches of random crops, all cropped to the shortest clip (no padding)."""

    def __init__(self, items, audio, seconds_per_batch=96.0, max_crop=8.0, seed=0):
        self.items, self.audio = items, audio
        self.budget, self.max_crop = seconds_per_batch * 16000, int(max_crop * 16000)
        self.rng = random.Random(seed)

    def __iter__(self):
        while True:
            order = sorted(self.items, key=lambda it: it["length"] + self.rng.random() * 8000)
            chunks, chunk = [], []
            for it in order:
                chunk.append(it)
                crop = min(min(c["length"] for c in chunk), self.max_crop)
                if crop * len(chunk) >= self.budget:
                    chunks.append(chunk)
                    chunk = []
            self.rng.shuffle(chunks)
            for chunk in chunks:
                crop = min(min(c["length"] for c in chunk), self.max_crop)
                waves = []
                for c in chunk:
                    start = c["offset"] + self.rng.randint(0, c["length"] - crop)
                    waves.append(self.audio[start:start + crop])
                yield torch.from_numpy(np.stack(waves).astype(np.float32) / 32767.0)


def kl(student_logits, teacher_logits):
    t = F.log_softmax(teacher_logits.float(), dim=-1)
    s = F.log_softmax(student_logits.float(), dim=-1)
    return (t.exp() * (t - s)).sum(-1).mean()


@torch.no_grad()
def evaluate(student, teachers, val, device):
    student.eval()
    agree_p = agree_a = cos = n = 0.0
    for wave in val:
        x = normalize(wave.to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tp, th, ta = teachers(x)
            sp, sa, sh = student(x)
        T = min(tp.shape[1], sp.shape[1])
        agree_p += (sp[:, :T].argmax(-1) == tp[:, :T].argmax(-1)).float().mean().item()
        agree_a += (sa[:, :T].argmax(-1) == ta[:, :T].argmax(-1)).float().mean().item()
        cos += F.cosine_similarity(sh[:, :T].float(), th[:, :T].float(), dim=-1).mean().item()
        n += 1
    student.train()
    return {"phone_argmax_agree": agree_p / n, "asr_argmax_agree": agree_a / n, "hidden_cos": cos / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/mnt/t7/openpronounce-browser/student")
    parser.add_argument("--hours", type=float, default=2.5, help="wall-clock budget")
    parser.add_argument("--steps", type=int, default=0, help="planned steps (0: from a throughput probe)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-seconds", type=float, default=96.0)
    parser.add_argument("--w-asr", type=float, default=0.5)
    parser.add_argument("--w-hid", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    from transformers import Wav2Vec2ForCTC

    with open(os.path.join(AUDIO, "index.json")) as f:
        index = json.load(f)
    audio = np.memmap(os.path.join(AUDIO, "audio.int16"), dtype=np.int16, mode="r")
    rng = random.Random(0)
    cv = [it for it in index if it["name"].startswith("cv:")]
    rng.shuffle(cv)
    val_items, train_items = cv[:200], cv[200:] + [it for it in index if it["name"].startswith("so762:")]
    val = []
    for it in val_items[:200]:
        crop = min(it["length"], 8 * 16000)
        val.append(torch.from_numpy(audio[it["offset"]:it["offset"] + crop].astype(np.float32)[None] / 32767.0))
    print(f"train {len(train_items)} clips {sum(i['length'] for i in train_items) / 16000 / 3600:.1f} h, val {len(val)} (CV)")

    phone_t = Wav2Vec2ForCTC.from_pretrained(PHONE, torch_dtype=torch.float16).to(device).eval()
    asr_t = Wav2Vec2ForCTC.from_pretrained(ASR, torch_dtype=torch.float16).to(device).eval()

    def teachers(x):
        with torch.no_grad():
            xh = x.half()
            out = phone_t(xh, output_hidden_states=True)
            return out.logits, out.hidden_states[TEACHER_LAYER], asr_t(xh).logits

    student = Student().to(device)
    student.body.feature_extractor._freeze_parameters()
    heads = [p for n, p in student.named_parameters() if not n.startswith("body.")]
    body = [p for n, p in student.named_parameters() if n.startswith("body.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": heads, "lr": args.lr * 10}], weight_decay=0.01)
    step, ckpt = 0, os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        student.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        args.steps = state["planned_steps"]
    batches = iter(Batches(train_items, audio, args.batch_seconds, seed=step))
    hid_scale = None
    t0, log = time.time(), open(os.path.join(args.out, "log.jsonl"), "a")
    planned = args.steps or 10 ** 9
    warmup = 500
    while step < planned:
        wave = next(batches).to(device, non_blocking=True)
        x = normalize(wave)
        tp, th, ta = teachers(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sp, sa, sh = student(x)
        T = min(tp.shape[1], sp.shape[1])
        if hid_scale is None:
            hid_scale = th.float().abs().mean().item()
        l_phone = kl(sp[:, :T], tp[:, :T])
        l_asr = kl(sa[:, :T], ta[:, :T])
        l_hid = F.l1_loss(sh[:, :T].float(), th[:, :T].float()) / hid_scale + \
            (1 - F.cosine_similarity(sh[:, :T].float(), th[:, :T].float(), dim=-1)).mean()
        loss = l_phone + args.w_asr * l_asr + args.w_hid * l_hid
        lr_scale = min(1.0, (step + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / planned))) if args.steps else 1.0)
        for g, base_lr in zip(opt.param_groups, (args.lr, args.lr * 10)):
            g["lr"] = base_lr * lr_scale
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()
        step += 1
        if step == 50 and not args.steps:
            # throughput probe: plan the cosine schedule to fit the wall-clock budget
            per_step = (time.time() - t0) / 50
            planned = args.steps = int(args.hours * 3600 / per_step)
            print(f"{per_step:.2f}s/step -> {planned} steps in {args.hours} h", flush=True)
        if step % 100 == 0:
            rec = {"step": step, "loss": loss.item(), "phone_kl": l_phone.item(), "asr_kl": l_asr.item(),
                   "hid": l_hid.item(), "lr": opt.param_groups[0]["lr"], "elapsed": time.time() - t0}
            if step % 2000 == 0 or step == planned:
                rec.update(evaluate(student, teachers, val, device))
                torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "step": step,
                            "planned_steps": planned}, ckpt)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    rec = {"step": step, "final": True, **evaluate(student, teachers, val, device)}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    torch.save({"model": student.state_dict(), "step": step, "planned_steps": planned}, os.path.join(args.out, "final.pt"))


if __name__ == "__main__":
    main()
