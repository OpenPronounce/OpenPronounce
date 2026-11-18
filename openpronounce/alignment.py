"""CTC forced alignment and sequence likelihood on frame log posteriors, in numpy.

Given the ``(frames, classes)`` log posteriors of a CTC model and the label sequence
that *should* have been said, :func:`forced_align` finds the most likely frame span of
every label (Viterbi over the blank-interleaved label sequence) and
:func:`log_likelihood` sums the probability of every alignment (the CTC forward pass).
Both are exact and need no torchaudio.
"""

import numpy as np

NEG = -1e30


def _extend(labels, blank):
    """Interleave ``labels`` with blanks: ``[b, l1, b, l2, ..., ln, b]``."""
    ext = np.full(2 * len(labels) + 1, blank, dtype=np.int64)
    ext[1::2] = labels
    return ext


def _skips(ext, blank):
    """Which extended states can be entered from two states back (a label that differs from the previous one)."""
    skip = np.zeros(len(ext), dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    return skip


def forced_align(log_probs, labels, blank=0):
    """Viterbi alignment of ``labels`` on ``log_probs`` (``(frames, classes)`` log posteriors).

    Returns ``(spans, log_prob)``: one ``(start, end)`` frame range per label (``end``
    exclusive) and the log probability of the best path. A label cannot be aligned when
    ``frames < 2 * len(labels) + 1`` in the worst case (repeated labels need a blank in
    between); the caller should check :func:`fits`.
    """
    log_probs = np.asarray(log_probs, dtype=np.float32)
    n_frames = log_probs.shape[0]
    ext = _extend(labels, blank)
    n_states = len(ext)
    lp = log_probs[:, ext]
    skip = _skips(ext, blank)
    dp = np.full(n_states, NEG, dtype=np.float64)
    dp[0] = lp[0, 0]
    if n_states > 1:
        dp[1] = lp[0, 1]
    backpointers = np.zeros((n_frames, n_states), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, n_frames):
        from1 = np.concatenate(([NEG], dp[:-1]))
        from2 = np.where(skip, np.concatenate(([NEG, NEG], dp[:-2])), NEG)
        best = np.maximum(np.maximum(dp, from1), from2)
        backpointers[t] = np.where(best == from2, 2, np.where(best == from1, 1, 0))
        dp = best + lp[t]
    if n_states == 1 or dp[n_states - 1] >= dp[n_states - 2]:
        state = n_states - 1
    else:
        state = n_states - 2
    total = float(dp[state])
    path = np.zeros(n_frames, dtype=np.int64)
    for t in range(n_frames - 1, -1, -1):
        path[t] = state
        state -= int(backpointers[t, state])
    spans = []
    for i in range(len(labels)):
        frames = np.flatnonzero(path == 2 * i + 1)
        spans.append((int(frames[0]), int(frames[-1]) + 1) if len(frames) else (0, 0))
    return spans, total


def fits(n_frames, labels):
    """Whether ``labels`` can be aligned on ``n_frames`` frames (every repeated label needs a blank in between)."""
    needed = len(labels) + sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    return n_frames >= needed


def log_likelihood(log_probs, labels, blank=0):
    """CTC forward pass: ``log P(labels | log_probs)`` summed over every alignment."""
    log_probs = np.asarray(log_probs, dtype=np.float32)
    n_frames = log_probs.shape[0]
    if len(labels) == 0:
        return float(log_probs[:, blank].sum())
    ext = _extend(labels, blank)
    n_states = len(ext)
    lp = log_probs[:, ext]
    skip = _skips(ext, blank)
    alpha = np.full(n_states, NEG)
    alpha[0] = lp[0, 0]
    if n_states > 1:
        alpha[1] = lp[0, 1]
    for t in range(1, n_frames):
        a1 = np.concatenate(([NEG], alpha[:-1]))
        a2 = np.where(skip, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        m = np.maximum(np.maximum(alpha, a1), a2)
        alpha = m + np.log(np.exp(alpha - m) + np.exp(a1 - m) + np.exp(a2 - m)) + lp[t]
    if n_states == 1:
        return float(alpha[0])
    m = max(alpha[-1], alpha[-2])
    return float(m + np.log(np.exp(alpha[-1] - m) + np.exp(alpha[-2] - m)))
