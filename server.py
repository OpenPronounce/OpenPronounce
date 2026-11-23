"""FastAPI server exposing the OpenPronounce web UI and JSON API.

Run with: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from openpronounce import __version__, audio, free, speech
from openpronounce.languages import DEFAULT_LANGUAGE, LANGUAGES, get_language

logger = logging.getLogger("openpronounce.server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="OpenPronounce",
    description="Phoneme-level pronunciation assessment (Wav2Vec2 + DTW). English by default, see /languages.",
    version=__version__,
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def _load_upload(upload: UploadFile):
    """Decode an uploaded recording to a 16 kHz mono waveform, leaving nothing on disk.

    ``audio.load`` decodes webm, m4a and the rest through ffmpeg and resamples, so the
    upload is read once from a temporary file that is removed whatever happens.
    """
    suffix = os.path.splitext(upload.filename or "")[1] or ".webm"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="openpronounce-upload-")
    try:
        with os.fdopen(fd, "wb") as buffer:
            buffer.write(upload.file.read())
        return audio.load(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _validate_lang(lang: str) -> str:
    try:
        return get_language(lang).code
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/pronunciation")
async def api_analyze_pronunciation(file: UploadFile = File(...), expected_text: str = Form(...),
                                    lang: str = Form(DEFAULT_LANGUAGE)):
    """Score ``file`` against ``expected_text`` in ``lang``. Returns the full analysis (score, errors, prosody)."""
    lang = _validate_lang(lang)
    try:
        sound = _load_upload(file)
        return speech.compare_audio_with_text(sound, expected_text, lang=lang)
    except Exception:
        logger.exception("pronunciation analysis failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/free")
async def api_assess_free_speech(file: UploadFile = File(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Score ``file`` without an expected text: transcript, scores and unclear words (see ``openpronounce.free``)."""
    lang = _validate_lang(lang)
    try:
        sound = _load_upload(file)
        return free.assess(sound, lang=lang)
    except Exception:
        logger.exception("free speech analysis failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/speech2text")
async def api_speech2text(file: UploadFile = File(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Transcribe ``file`` with the Wav2Vec2 model of ``lang``."""
    lang = _validate_lang(lang)
    try:
        sound = _load_upload(file)
        return {"transcript": speech.transcribe(sound, lang)}
    except Exception:
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/phonemes")
async def api_phonemes(text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Return the IPA phonemes of ``text`` in ``lang`` and the word each phoneme belongs to."""
    lang = _validate_lang(lang)
    try:
        phonemes, words = speech.get_phonemes_with_word_mapping(text, lang)
        return {"phonemes": phonemes, "words": list(words.values())}
    except Exception:
        logger.exception("phonemization failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/tts")
async def api_tts(text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
    """Return a 16 kHz wav reference pronunciation of ``text`` in ``lang``."""
    lang = _validate_lang(lang)
    try:
        return FileResponse(audio.text2speech(text, lang=lang), media_type="audio/wav")
    except Exception:
        logger.exception("tts failed")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.get("/languages")
async def api_languages():
    """List the supported languages (``code`` is the value of the ``lang`` form field)."""
    languages = [{"code": language.code, "name": language.name} for language in LANGUAGES.values()]
    return {"default": DEFAULT_LANGUAGE, "languages": languages}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})
