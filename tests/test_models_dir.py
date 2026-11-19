import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from openpronounce import cli, gop, scorer


def _scorer(lang, **extra):
    return {"lang": lang, "features": [], "models": {}, "trained_on": "unit test", "metrics": {"total": {"pearson": 0.5}},
            **extra}


class TestModelsDir(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {gop.MODELS_ENV: self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        for cached in (scorer.load_scorer, gop.load_model):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)

    def _write(self, name, data):
        with open(os.path.join(self.tmp.name, name), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_adds_a_language(self):
        self.assertEqual(scorer.load_scorer("xx").data["lang"], "generic")
        self.assertEqual(scorer.load_scorer("xx", "free_scorer").data["lang"], "generic")
        scorer.load_scorer.cache_clear()
        self._write("scorer_xx.json", _scorer("xx"))
        self.assertEqual(scorer.load_scorer("xx").data["trained_on"], "unit test")

    def test_overrides_the_shipped_model(self):
        self._write("scorer_en.json", _scorer("en"))
        self.assertEqual(scorer.load_scorer("en").data["trained_on"], "unit test")

    def test_shipped_model_without_the_variable(self):
        with mock.patch.dict(os.environ, {gop.MODELS_ENV: ""}):
            self.assertEqual(gop.models_dirs(), [gop.MODELS_DIR])
            self.assertTrue(gop.model_path("scorer_en.json").startswith(gop.MODELS_DIR))

    def test_several_directories_first_wins(self):
        with tempfile.TemporaryDirectory() as second:
            with open(os.path.join(second, "scorer_xx.json"), "w", encoding="utf-8") as f:
                json.dump(_scorer("xx", trained_on="second"), f)
            self._write("scorer_xx.json", _scorer("xx"))
            with mock.patch.dict(os.environ, {gop.MODELS_ENV: os.pathsep.join([self.tmp.name, second])}):
                self.assertEqual(scorer.load_scorer("xx").data["trained_on"], "unit test")

    def test_missing_directory_is_ignored(self):
        with mock.patch.dict(os.environ, {gop.MODELS_ENV: os.path.join(self.tmp.name, "nope")}):
            self.assertIsNotNone(scorer.load_scorer("en"))
            self.assertTrue(all(m["bundled"] for m in gop.installed_models()))

    def test_installed_models(self):
        self._write("scorer_xx.json", _scorer("xx", license="Commercial"))
        self._write("scorer_en.json", _scorer("en"))
        self._write("notes.json", {})
        models = {(m["lang"], m["kind"]): m for m in gop.installed_models() if m["recognizer"] is None}
        self.assertEqual(models["xx", "scorer"]["license"], "Commercial")
        self.assertEqual(models["en", "scorer"]["license"], "unspecified")
        self.assertFalse(models["en", "scorer"]["bundled"])
        self.assertEqual(models["en", "word_mdd"]["license"], gop.PACKAGE_LICENSE)
        self.assertTrue(models["en", "word_mdd"]["bundled"])
        self.assertTrue(models["en", "free_scorer"]["bundled"])
        self.assertTrue(models["generic", "word_missing"]["bundled"])
        self.assertTrue(models["generic", "free_scorer"]["bundled"])
        self.assertTrue(models["nl", "native"]["bundled"])
        self.assertEqual(len(models), 14)
        student = [m for m in gop.installed_models() if m["recognizer"] == "student"]
        self.assertEqual(sorted(m["kind"] for m in student), ["scorer", "word_mdd"])
        self.assertTrue(all(m["bundled"] and m["lang"] == "en" for m in student))

    def test_cli_lists_models(self):
        self._write("scorer_xx.json", _scorer("xx", license="Commercial"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["--models"]), 0)
        self.assertIn("xx  utterance scorer Commercial, total pearson 0.500", out.getvalue())
        self.assertIn("trained on unit test", out.getvalue())

    def test_cli_still_requires_audio(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli.main([])


if __name__ == "__main__":
    unittest.main()


class TestGenericBuild(unittest.TestCase):
    """What the benchmarks need to rebuild the generic scorers and their native reference."""

    def setUp(self):
        import sys

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {gop.MODELS_ENV: self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        for cached in (scorer.load_scorer, gop.load_model, scorer.native_shift):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)
        patcher = mock.patch.object(sys, "path", ["benchmarks"] + sys.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_english_runs_on_the_generic_models(self):
        import shutil

        import word_model

        directory = word_model.english_on_generic_models()
        self.addCleanup(shutil.rmtree, directory)
        self.assertEqual(os.environ[gop.MODELS_ENV].split(os.pathsep), [directory, self.tmp.name])
        self.assertEqual(scorer.load_scorer("en").data["lang"], "generic")
        self.assertEqual(scorer.load_scorer("en", "free_scorer").data["lang"], "generic")
        self.assertEqual(os.path.realpath(gop.model_path("word_mdd_en.json")),
                         os.path.realpath(os.path.join(gop.MODELS_DIR, "word_mdd_generic.json")))

    def test_reference_is_written_into_the_generic_scorers(self):
        import native_calibration

        for kind in ("scorer", "free_scorer"):
            with open(os.path.join(self.tmp.name, f"{kind}_generic.json"), "w", encoding="utf-8") as f:
                json.dump(_scorer("generic"), f)
        with self.assertRaises(ValueError):
            scorer.native_offsets([{name: 0.0 for name in scorer.ALIGNMENT_FEATURES}])
        rows = [{name: value for name in scorer.FREE_FEATURES} for value in (1.0, 3.0)]
        written = native_calibration.write_reference({"scorer": rows, "free_scorer": []}, "english.csv")
        self.assertEqual(list(written), ["scorer"])
        data = scorer.load_scorer("generic").data
        self.assertEqual(data["native_reference"], {name: 2.0 for name in scorer.ALIGNMENT_FEATURES})
        self.assertIn("english.csv, 2 recordings", data["native_reference_source"])
        self.assertNotIn("native_reference", scorer.load_scorer("generic", "free_scorer").data)
        self.assertEqual(scorer.native_offsets([{name: 0.5 for name in scorer.ALIGNMENT_FEATURES}])["gm_mean"], 1.5)
