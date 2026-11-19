"""Another phone recognizer than the default one: its config block, its hidden projection, its own models."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from openpronounce import gop, phones, scorer

VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "h": 4, "ə": 5}


def _tiny_recognizer(directory, models="tiny", source_layer=1, layer=20, out_dim=6):
    """A random two-layer Wav2Vec2ForCTC saved like the distilled student, with a hidden projection."""
    from safetensors.torch import save_file
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    torch.manual_seed(0)
    config = Wav2Vec2Config(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                            conv_dim=(8,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                            vocab_size=len(VOCAB), pad_token_id=0)
    config.openpronounce = {"models": models, "hidden_projection": {
        "file": "hidden_projection.safetensors", "source_layer": source_layer, "layer": layer}}
    Wav2Vec2ForCTC(config).eval().save_pretrained(directory)
    Wav2Vec2FeatureExtractor(do_normalize=True).save_pretrained(directory)
    with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(VOCAB, f)
    linear = torch.nn.Linear(16, out_dim)
    save_file({"weight": linear.weight.detach().contiguous(), "bias": linear.bias.detach().contiguous()},
              os.path.join(directory, "hidden_projection.safetensors"))
    return linear


def _clear():
    for cached in (phones._load_model, phones._model_info, phones._vocab_ids, phones.phone_vocab,
                   gop.load_model, gop.load_missing_model, scorer.load_scorer):
        cached.cache_clear()


class _Selected(unittest.TestCase):
    """Runs every test with ``phones.PHONE_MODEL_NAME`` pointing at a tiny recognizer in a temp folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.linear = _tiny_recognizer(self.tmp.name)
        patcher = mock.patch.object(phones, "PHONE_MODEL_NAME", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"OPENPRONOUNCE_DEVICE": "cpu"})
        patcher.start()
        self.addCleanup(patcher.stop)
        from openpronounce import device

        device.get_device.cache_clear()
        self.addCleanup(device.get_device.cache_clear)
        _clear()
        self.addCleanup(_clear)


class TestModelInfo(unittest.TestCase):

    def test_default_model_has_no_block_and_is_not_read(self):
        with mock.patch.object(phones, "PHONE_MODEL_NAME", phones.DEFAULT_PHONE_MODEL), \
                mock.patch("transformers.utils.cached_file", side_effect=AssertionError("no download")):
            phones._model_info.cache_clear()
            self.assertEqual(phones.model_info(), {})
            self.assertIsNone(gop.recognizer_suffix())

    def test_disabled_recognizer_has_no_block(self):
        with mock.patch.object(phones, "PHONE_MODEL_NAME", "off"):
            self.assertEqual(phones.model_info(), {})


class TestHiddenProjection(_Selected):

    def test_block_is_read_from_the_config(self):
        self.assertEqual(phones.model_info()["models"], "tiny")
        self.assertEqual(phones.model_info()["hidden_projection"]["layer"], 20)

    def test_the_layer_it_stands_for_goes_through_the_projection(self):
        wave = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
        log_posteriors, hidden = phones.phone_log_posteriors(wave, hidden_layer=20)
        self.assertEqual(log_posteriors.shape[1], len(VOCAB))
        self.assertEqual(hidden.shape, (log_posteriors.shape[0], 6))
        _, model, _ = phones._load_model()
        inputs = phones._load_model()[0](wave, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            expected = self.linear(model(inputs.input_values, output_hidden_states=True).hidden_states[1][0]).numpy()
        np.testing.assert_allclose(hidden, expected, atol=1e-5)

    def test_another_layer_is_refused(self):
        with self.assertRaises(ValueError):
            phones.phone_log_posteriors(np.zeros(16000, dtype=np.float32), hidden_layer=12)

    def test_posteriors_alone_need_no_projection(self):
        log_posteriors = phones.phone_log_posteriors(np.zeros(16000, dtype=np.float32))
        np.testing.assert_allclose(np.exp(log_posteriors).sum(axis=1), 1.0, rtol=1e-5)

    def test_recognition_keeps_the_projected_states(self):
        recognition = phones.recognize_phones(np.zeros(16000, dtype=np.float32), hidden_layer=20)
        self.assertEqual(recognition.vocab, tuple(VOCAB))
        self.assertEqual(recognition.hidden.shape[1], 6)


class TestModelsOfTheRecognizer(_Selected):

    def setUp(self):
        super().setUp()
        self.models = tempfile.TemporaryDirectory()
        self.addCleanup(self.models.cleanup)
        patcher = mock.patch.dict(os.environ, {gop.MODELS_ENV: self.models.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, name, data):
        with open(os.path.join(self.models.name, name), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_names_put_the_recognizer_first(self):
        self.assertEqual(gop.model_names("scorer", "en"),
                         ["scorer_en_tiny.json", "scorer_en.json", "scorer_generic_tiny.json", "scorer_generic.json"])
        self.assertEqual(gop.model_names("scorer", "generic"), ["scorer_generic_tiny.json", "scorer_generic.json"])

    def test_names_without_a_suffix(self):
        with mock.patch.object(phones, "PHONE_MODEL_NAME", phones.DEFAULT_PHONE_MODEL):
            self.assertEqual(gop.model_names("word_mdd", "fr"), ["word_mdd_fr.json", "word_mdd_generic.json"])

    def test_its_own_scorer_wins(self):
        self._write("scorer_en_tiny.json", {"lang": "en", "features": [], "models": {}, "trained_on": "tiny"})
        self.assertEqual(scorer.load_scorer("en").data["trained_on"], "tiny")

    def test_without_its_own_model_the_default_ones_load(self):
        self.assertTrue(gop.find_model("word_mdd", "en").endswith("word_mdd_en.json"))
        self.assertTrue(gop.find_model("word_mdd", "fr").endswith("word_mdd_generic.json"))

    def test_listed_with_their_recognizer(self):
        self._write("scorer_en_tiny.json", {"lang": "en", "features": [], "models": {}, "trained_on": "tiny"})
        models = {(m["lang"], m["kind"], m["recognizer"]): m for m in gop.installed_models()}
        self.assertFalse(models["en", "scorer", "tiny"]["bundled"])
        self.assertTrue(models["en", "scorer", None]["bundled"])


class TestShippedStudentModels(unittest.TestCase):

    def test_they_read_the_layer_the_student_projects_onto(self):
        path = gop.model_path("word_mdd_en_student.json")
        if path is None:
            self.skipTest("student models not installed")
        self.assertEqual(gop.WordModel.load(path).hidden_layer, 20)
        self.assertIsNotNone(gop.model_path("scorer_en_student.json"))


if __name__ == "__main__":
    unittest.main()
