import unittest

import numpy as np

from openpronounce import alignment


def _log_probs(ids, n_classes=4):
    """Near one-hot log posteriors following the frame-by-frame ``ids`` (0 = blank)."""
    probs = np.full((len(ids), n_classes), 0.01)
    for t, i in enumerate(ids):
        probs[t, i] = 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


class TestForcedAlign(unittest.TestCase):

    def test_spans_follow_the_frames(self):
        lp = _log_probs([0, 1, 1, 0, 2, 2, 2, 0, 3, 0])
        spans, total = alignment.forced_align(lp, [1, 2, 3])
        self.assertEqual(spans, [(1, 3), (4, 7), (8, 9)])
        self.assertLess(total, 0)

    def test_repeated_label_needs_a_blank(self):
        lp = _log_probs([1, 1, 0, 1, 1])
        spans, _ = alignment.forced_align(lp, [1, 1])
        self.assertEqual(spans, [(0, 2), (3, 5)])
        self.assertTrue(alignment.fits(5, [1, 1]))
        self.assertFalse(alignment.fits(2, [1, 1]))

    def test_missing_label_is_squeezed(self):
        # label 2 never shows up: it still gets a (short) span, the path log prob drops
        lp = _log_probs([1, 1, 1, 3, 3, 3])
        spans, total_wrong = alignment.forced_align(lp, [1, 2, 3])
        _, total_right = alignment.forced_align(lp, [1, 3])
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[1][1] - spans[1][0], 1)
        self.assertLess(total_wrong, total_right)


class TestLogLikelihood(unittest.TestCase):

    def test_sums_to_one_over_all_sequences(self):
        # two frames, blank + two labels: every label sequence the model can emit
        rng = np.random.default_rng(0)
        lp = np.log(rng.dirichlet(np.ones(3), size=2))
        sequences = [[], [1], [2], [1, 2], [2, 1], [1, 1], [2, 2]]
        total = sum(np.exp(alignment.log_likelihood(lp, s)) for s in sequences)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_right_sequence_is_more_likely(self):
        lp = _log_probs([0, 1, 1, 0, 2, 2, 2, 0, 3, 0])
        self.assertGreater(alignment.log_likelihood(lp, [1, 2, 3]), alignment.log_likelihood(lp, [1, 3]))
        self.assertGreater(alignment.log_likelihood(lp, [1, 2, 3]), alignment.log_likelihood(lp, [1, 3, 2]))

    def test_viterbi_bounded_by_forward(self):
        rng = np.random.default_rng(1)
        lp = np.log(rng.dirichlet(np.ones(5), size=12))
        _, best_path = alignment.forced_align(lp, [1, 2, 2, 4])
        self.assertLessEqual(best_path, alignment.log_likelihood(lp, [1, 2, 2, 4]) + 1e-9)
