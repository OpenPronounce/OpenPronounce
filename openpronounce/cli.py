"""Command-line entry point: ``openpronounce <audio> "<expected text>"``."""

import argparse
import json
import sys

from .languages import DEFAULT_LANGUAGE, LANGUAGES


def _print_models():
    from . import gop

    for model in gop.installed_models():
        metrics = model["metrics"]
        headline = next(((key, metrics[key]) for key in ("total", "word_auc_lt5") if key in metrics),
                        next(((key, value) for key, value in metrics.items() if not isinstance(value, str)), None))
        if headline and isinstance(headline[1], dict):
            headline = (f"{headline[0]} {next(iter(headline[1]))}", next(iter(headline[1].values())))
        metric = f", {headline[0]} {headline[1]:.3f}" if headline else ""
        lang = model["lang"] + (f" ({model['recognizer']})" if model.get("recognizer") else "")
        print(f"{lang}  {gop.MODEL_KINDS[model['kind']]:16s} {model['license']}{metric}")
        print(f"    {model['path']}")
        if model["trained_on"]:
            print(f"    trained on {model['trained_on']}")
    return 0


def _quiet():
    """Keep the model-loading chatter of transformers out of the terminal (downloads still show their progress)."""
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()


def _free(args):
    from . import audio, free

    result = free.assess(audio.load(args.audio), lang=args.lang)
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    if result["scores"]:
        scores = result["scores"]
        print(f"Score        : {scores['total']}/100")
        print(f"               accuracy {scores['accuracy']}, fluency {scores['fluency']}, prosodic {scores['prosodic']}")
    else:
        print("Score        : none")
    if not result["reliable"]:
        print(f"Warning      : {'; '.join(result['reasons'])} (scores unreliable)")
    print(f"Transcription: {result['transcribe']}")
    print(f"Speech       : {result['speech_duration']:.1f} s in {len(result['segments'])} segment(s)")
    if result["unclear_words"]:
        print("Unclear words: " + ", ".join(result["unclear_words"]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="openpronounce",
        description="Score the pronunciation of a recording, against the sentence it should contain or as free speech.",
    )
    parser.add_argument("audio", nargs="?", help="path to a wav/mp3/flac/ogg/webm recording")
    parser.add_argument("text", nargs="?", help="the sentence the speaker was supposed to say; leave it out for free speech")
    parser.add_argument("--models", action="store_true",
                        help="list the scoring models in use (shipped ones and $OPENPRONOUNCE_MODELS_DIR) and exit")
    parser.add_argument("--lang", default=DEFAULT_LANGUAGE, choices=sorted(LANGUAGES),
                        help="language of the sentence (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="print the full JSON result instead of a summary")
    parser.add_argument("--no-prosody", action="store_true", help="omit prosody contours from the JSON output")
    parser.add_argument("--fast", action="store_true",
                        help="phone recognizer only: no word transcription, no synthetic reference, no prosody contours")
    args = parser.parse_args(argv)

    if args.models:
        return _print_models()
    if args.audio is None:
        parser.error("the following arguments are required: audio")
    _quiet()
    if args.text is None:
        return _free(args)

    from . import audio, speech

    sound = audio.load(args.audio)
    options = {"acoustic": False, "transcription": False, "prosody": False} if args.fast else {}
    result = speech.compare_audio_with_text(sound, args.text, lang=args.lang, **options)

    if args.no_prosody:
        result.pop("prosody", None)

    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"Score        : {result['score']}/100")
    if result.get("scores"):
        scores = result["scores"]
        print(f"               accuracy {scores['accuracy']}, fluency {scores['fluency']}, prosodic {scores['prosodic']}")
    if not result["reliable"]:
        print(f"Warning      : {'; '.join(result['reasons'])} (scores unreliable)")
    if result["transcribe"] is not None:
        print(f"Transcription: {result['transcribe']}")
    if "heard_phones" in result["differences"]:
        print(f"Heard phones : /{' '.join(result['differences']['heard_phones'])}/")
    errors = result["differences"]["errors"]
    if errors:
        print("Mispronounced:")
        for err in errors:
            if err.get("missing"):
                print(f"  - {err['word']}: not said")
                continue
            actual = f"/{err['actual']}/" if err["actual"] else "(missing)"
            confidence = f" (confidence {err['confidence']:.0%})" if "confidence" in err else ""
            print(f"  - {err['word']}: expected /{err['expected']}/, heard {actual}{confidence}")
    elif result["reliable"]:
        print("No mispronounced word detected.")
    words = {w["position"]: w["word"] for w in result["differences"].get("words", [])}
    for extra in result["differences"].get("extra", []):
        if extra["after"] is not None and extra["before"] is not None:
            where = f'between "{words[extra["after"]]}" and "{words[extra["before"]]}"'
        elif extra["after"] is not None:
            where = f'after "{words[extra["after"]]}"'
        else:
            where = f'before "{words[extra["before"]]}"'
        print(f"Not expected : /{' '.join(extra['heard'])}/ {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
