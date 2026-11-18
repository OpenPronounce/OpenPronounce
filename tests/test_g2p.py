import os
import unittest
from unittest import mock

from openpronounce import g2p, phones


class TestLexiconBackend(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {g2p.BACKEND_ENV: ""})
        patcher.start()
        self.addCleanup(patcher.stop)
        g2p.clear_cache()
        self.addCleanup(g2p.clear_cache)

    def test_english_does_not_need_espeak(self):
        self.assertEqual(g2p.backend("en"), "lexicon")
        self.assertEqual(g2p.backend("fr"), "espeak")
        with mock.patch.object(g2p, "_espeak", side_effect=AssertionError("espeak called")):
            words, groups = g2p.phones_by_word("Hello, how are you?", "en")
        self.assertEqual(words, ("hello", "how", "are", "you"))
        self.assertEqual(groups, (("h", "ə", "l", "oʊ"), ("h", "aʊ"), ("ɑːɹ",), ("j", "uː")))

    def test_symbols_follow_the_recognizer(self):
        self.assertEqual(g2p.arpabet_to_phones(("B", "ER1", "D")), ("b", "ɜː", "d"))
        self.assertEqual(g2p.arpabet_to_phones(("B", "AH1", "T", "ER0")), ("b", "ʌ", "ɾ", "ɚ"))
        self.assertEqual(g2p.arpabet_to_phones(("P", "IY1", "P", "AH0", "L")), ("p", "iː", "p", "əl"))
        self.assertEqual(g2p.arpabet_to_phones(("HH", "AE1", "P", "IY0")), ("h", "æ", "p", "i"))
        self.assertEqual(g2p.arpabet_to_phones(("S", "T", "AA1", "R", "T")), ("s", "t", "ɑːɹ", "t"))
        # an r that opens the next syllable stays a consonant
        self.assertEqual(g2p.arpabet_to_phones(("S", "AA1", "R", "IY0")), ("s", "ɑː", "ɹ", "i"))

    def test_upper_case_and_punctuation(self):
        self.assertEqual(g2p.phones_by_word("IT'S OK.", "en")[0], ("it's", "ok"))

    def test_words_derived_from_a_known_word(self):
        cases = {"dora's": "d ɔː ɹ ə z", "wolfs": "w ʊ l f s", "stoped": "s t ɑː p t", "jumpped": "dʒ ʌ m p t",
                 "birdbath": "b ɜː d b æ θ"}
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(" ".join(g2p.phones_by_word(word, "en")[1][0]), expected)
                self.assertFalse(g2p.is_guessed(word, "en"))

    def test_unknown_word_is_a_guess(self):
        self.assertTrue(g2p.is_guessed("Quirt", "en"))
        self.assertEqual(g2p.phones_by_word("quirt", "en")[1][0], ("k", "w", "ɜː", "t"))
        self.assertFalse(g2p.is_guessed("hello", "en"))

    def test_numbers(self):
        self.assertEqual(g2p.number_words(0), ["zero"])
        self.assertEqual(g2p.number_words(42), ["forty", "two"])
        self.assertEqual(g2p.number_words(2026), ["two", "thousand", "twenty", "six"])
        self.assertEqual(g2p.number_words(1200300), ["one", "million", "two", "hundred", "thousand", "three", "hundred"])
        words, groups = g2p.phones_by_word("I have 3 cats", "en")
        self.assertEqual(words, ("i", "have", "3", "cats"))
        self.assertEqual(groups[2], ("θ", "ɹ", "iː"))
        self.assertFalse(g2p.is_guessed("3", "en"))

    def test_empty_text(self):
        self.assertEqual(g2p.phones_by_word("...", "en"), ((), ()))

    def test_forced_backend(self):
        with mock.patch.dict(os.environ, {g2p.BACKEND_ENV: "espeak"}):
            self.assertEqual(g2p.backend("en"), "espeak")
            self.assertFalse(g2p.is_guessed("quirt", "en"))
        with mock.patch.dict(os.environ, {g2p.BACKEND_ENV: "lexicon"}):
            with self.assertRaises(ValueError):
                g2p.backend("fr")
        with mock.patch.dict(os.environ, {g2p.BACKEND_ENV: "festival"}):
            with self.assertRaises(ValueError):
                g2p.backend("en")

    def test_missing_espeak_says_how_to_install(self):
        with mock.patch.dict(os.environ, {g2p.ESPEAK_ENV: ""}), mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "apt install espeak-ng"):
                g2p.phones_by_word("bonjour", "fr")

    def test_espeak_output_is_split_into_words_and_phones(self):
        printed = "b_ɔ̃_ʒ_ˈu_ʁ ʒ_ə- s_y_ˌi\n(en)d_ɪ_v_ˈɛ_l_ə_p_ɚ(fr)\n"
        with mock.patch.object(g2p, "espeak_binary", return_value="espeak-ng"), \
                mock.patch("subprocess.run", return_value=mock.Mock(stdout=printed)) as run:
            groups = g2p._espeak("bonjour je suis developer", "fr-fr")
        self.assertEqual(groups, [("b", "ɔ̃", "ʒ", "u", "ʁ"), ("ʒ", "ə"), ("s", "y", "i"),
                                  ("d", "ɪ", "v", "ɛ", "l", "ə", "p", "ɚ")])
        self.assertEqual(run.call_args[0][0][-3:], ["-v", "fr-fr", "bonjour je suis developer"])

    def test_merged_words_are_phonemized_one_by_one(self):
        answers = {"would have to": [("w", "ʊ", "d", "h", "æ", "f", "t", "ə")], "would": [("w", "ʊ", "d")],
                   "have": [("h", "æ", "v")], "to": [("t", "ə")]}
        with mock.patch.object(g2p, "_espeak", side_effect=lambda text, voice: answers[text]):
            self.assertEqual(g2p._espeak_words(("would", "have", "to"), "en"),
                             (("w", "ʊ", "d"), ("h", "æ", "v"), ("t", "ə")))


class TestGuessedWordsAreNotBlamed(unittest.TestCase):

    def setUp(self):
        g2p.clear_cache()

    def test_rule_path_skips_a_guessed_word(self):
        # "quirt" is not in the dictionary: whatever was heard, it is not reported
        result = phones.compare_phones(["h", "ə", "l", "oʊ", "b", "ɑ", "b"], "hello quirt", "en", use_model=False)
        self.assertNotIn("quirt", result["words_with_errors"])
        result = phones.compare_phones(["b", "ɑ", "b", "k", "w", "ɚ", "t"], "hello quirt", "en", use_model=False)
        self.assertEqual(result["words_with_errors"], ["hello"])


if __name__ == "__main__":
    unittest.main()
