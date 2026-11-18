# Licensing

OpenPronounce is free software under the [GNU Affero General Public License](LICENSE), version 3 or later. You can use it, study it, change it and share it. If you distribute it, or let people use a product built on it over a network, the AGPL asks you to publish the source of that product under the same licence.

Releases 0.1 to 0.4 were published under the MIT licence and stay under it ([LICENSE-MIT-0.x](LICENSE-MIT-0.x)).

## Commercial licence

If the AGPL does not fit your product, a commercial licence replaces it: you ship OpenPronounce inside closed software, on a server, in a mobile app or in a browser, without publishing your own code. It covers the code of this repository and the models shipped with it, for every language.

You do not need it to find out whether OpenPronounce is good enough for you: the AGPL lets you run it on your own recordings, and `openpronounce-evaluate` tells you how closely its scores follow your assessors'. When the figures suit you, the plans and prices are at <https://openpronounce.github.io/pricing.html>, and the terms at <https://openpronounce.github.io/docs/commercial-licence.html>.

## What a commercial licence does not cover

Nothing OpenPronounce imports is under a copyleft licence. Three things around it are, and they keep their own licence whatever yours is:

- espeak-ng (GPL-3), which every language other than English needs. It is a separate program: OpenPronounce runs it and reads what it prints, without importing or linking any of it. If you ship espeak-ng with your product, its licence applies to that program and to it alone. English does not use it;
- the Piper voices (`openpronounce[tts-piper]`, GPL-3);
- the mouth drawings of the web demo, taken from Papagayo-NG.

[NOTICE](NOTICE) lists every third-party component and its licence.

## Models

A model file states its own licence in its `license` key; `openpronounce --models` prints it. The shipped models follow the licence of the code. They were trained on speechocean762 (CC BY 4.0).
