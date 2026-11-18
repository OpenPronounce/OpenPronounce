# What OpenPronounce does not do

Read this before you put a score in front of a learner.

**The models learned from one population.** Mandarin-speaking learners of English, half of them children, reading sentences of 2 to 12 words. The scores follow their raters: a clean adult reading gets about 80, not 100, and an accent of another origin is less well covered. `openpronounce-evaluate` tells you what that costs on your own learners.

**Children are scored less well than adults.** Wav2Vec2 was trained on adults. With the sentence, the score follows the raters at 0.85 for adults and 0.67 for children of 12 and under. Without it, 0.82 and 0.42: do not use free speech with children.

**Free speech was measured on read sentences with the text hidden.** No rated corpus of spontaneous speech can be used here. A long recording is cut at its pauses and the scores are the average of its parts; on 22-second recordings of adults they follow the raters at about 0.75. Without the text there is no telling which sounds went wrong: words are reported as unclear, never as mispronounced.

**Quick native speech is not rewarded.** Italians reading a text at their own pace scored 2 points under French speakers reading the same Italian text slowly and clearly. The sound recognizer is less sure of fast speech, and the scorer takes that for a lack of clarity. A careful intermediate learner can outscore a native.

**Hesitation costs points, whoever hesitates.** The scorer learned from read sentences, where a pause or a restart is a fault. Natives talking on the radio or telling their life get 66 to 67 where natives of the same languages get 72 when they read, and they lose it all in fluency and prosody. Expect the same for a fluent learner who thinks while speaking.

**A missing word is found more surely than an extra one.** On sentences where a word was added to the text, 83 % of those words are reported as not said, and 0.7 % of the words really read are too. On sentences where a word was removed from the text, the speech in excess is located 63 times in 100, and reported by mistake in 3 sentences in 100. A single extra sound ("an developer" for "a developer") goes unnoticed.

**Only English was measured against human ratings.** French, Spanish, German, Italian, Portuguese and Dutch use models trained on the English learners without anything specific to English sounds. They were checked on native speakers (220 recordings per language): natives score like English ones (72 to 77, English 74); about one of their words in a hundred is reported; a word replaced by another is found three to four times in five. Free speech works for them too: natives get 70 to 73 without their text, English ones 71. Nobody rated learners of these languages, so how well the scores follow a teacher's judgement there is unknown. All that could be checked is whether natives come out above learners, in French, Italian and Spanish, on groups of 8 to 18 speakers. They do in short spoken answers and read sentences, by 1.5 to 4.6 points. They do not on a text read aloud in Italian.

**The scores of the other languages rest on an assumption.** German, Portuguese and Dutch natives used to score 4 to 12 points under English ones, because the sound recognizer and the pronunciation dictionary agree less well in these languages. Each language now has a native reference that cancels that gap. It assumes a learner strays from the natives of the language the way a learner of English strays from English natives, and no learner of German, Portuguese or Dutch was there to check.

**Long sentences are outside what the scorer learned.** It never saw more than 10 words. English natives get 77 on sentences of 12 words at most and 73 from 20 words up, and on long readings the fluency and prosody scores follow the raters less well than the total.

**A foreign name is read with its foreign sounds.** Outside English, the expected sounds of "Aerosmith" are the English ones, and a speaker who says it the local way gets it reported.

**The word model is lenient where the raters are.** Function words, word-final consonants, accent traits the raters would not mark below 5. The phone recognizer itself mishears about one sound in ten on a clean native reading.

**A word missing from the English dictionary is never blamed.** It is derived from a known word or pronounced by rule, marked `guessed`, and left out of the errors.

**A phone line moves the score.** The models learned from 16 kHz recordings. Through a simulated land line (8 kHz, narrow band) the scores still follow the raters (0.78 against 0.81 with the sentence, 0.64 against 0.68 without), but one recording's score moves by 3 to 4 points on average, and a real call adds noise that was not simulated. Do not compare a score taken on the phone with one taken on a microphone.

**A bad recording is scored anyway.** Too quiet (under -60 dBFS, where a 16-bit file starts to lose points), clipped, noisy or too short: `audio_quality.issues` says so and `reliable` turns false. Showing the score or not is your decision.

**The synthetic reference voice needs the network** with gTTS, on the first analysis of a sentence. `--fast`, `acoustic=False`, Piper or Kokoro avoid it.
