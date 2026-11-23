import io
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import server


def _wav_upload(name="rec.wav"):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
    buf.seek(0)
    return {"file": (name, buf, "audio/wav")}


class TestServer(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(server.app)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_home_serves_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<html", response.text)

    def test_phonemes(self):
        response = self.client.post("/phonemes", data={"text": "hello world"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreater(len(body["phonemes"]), 0)
        self.assertEqual(len(body["phonemes"]), len(body["words"]))

    @patch("server.speech.transcribe", return_value="HELLO")
    def test_speech2text(self, _):
        import io
        import numpy as np
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
        buf.seek(0)
        response = self.client.post("/speech2text", files={"file": ("rec.wav", buf, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"transcript": "HELLO"})

    def test_ui_assets_and_languages(self):
        for path in ("/static/ui.js", "/static/audio.js", "/static/viseme.js", "/static/assets/logo.svg"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        home = self.client.get("/").text
        for element in ("record-btn", "language-select", "expected-text", "word-chips", "score-ring"):
            self.assertIn(f'id="{element}"', home)
        languages = self.client.get("/languages").json()
        self.assertEqual(languages["default"], "en")
        self.assertIn({"code": "en", "name": "English"}, languages["languages"])


class TestFreeSpeechRoute(unittest.TestCase):

    @patch("server.free.assess", return_value={"mode": "free", "score": 70.0})
    def test_free(self, assess):
        import io
        import numpy as np
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
        buf.seek(0)
        response = TestClient(server.app).post("/free", files={"file": ("rec.wav", buf, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"mode": "free", "score": 70.0})
        self.assertEqual(assess.call_args.kwargs, {"lang": "en"})


class TestUploadsLeaveNothingOnDisk(unittest.TestCase):
    """A recording is decoded in memory: neither the upload nor a converted copy stays in the temp dir."""

    ROUTES = (
        ("/pronunciation", "server.speech.compare_audio_with_text", {"expected_text": "hello"}),
        ("/free", "server.free.assess", {}),
        ("/speech2text", "server.speech.transcribe", {}),
    )

    def setUp(self):
        self.client = TestClient(server.app)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        patcher = patch.object(tempfile, "tempdir", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_temp_dir_empty(self):
        self.assertEqual(os.listdir(self.tmp), [])

    def test_success_leaves_nothing(self):
        for route, target, data in self.ROUTES:
            with self.subTest(route=route), patch(target, return_value={}):
                response = self.client.post(route, data=data, files=_wav_upload())
                self.assertEqual(response.status_code, 200)
                self.assert_temp_dir_empty()

    def test_failure_leaves_nothing(self):
        for route, target, data in self.ROUTES:
            with self.subTest(route=route), patch(target, side_effect=RuntimeError("boom")):
                response = self.client.post(route, data=data, files=_wav_upload())
                self.assertEqual(response.status_code, 500)
                self.assert_temp_dir_empty()

    def test_undecodable_upload_leaves_nothing(self):
        files = {"file": ("rec.webm", io.BytesIO(b"not audio"), "audio/webm")}
        response = self.client.post("/speech2text", files=files)
        self.assertEqual(response.status_code, 500)
        self.assert_temp_dir_empty()
