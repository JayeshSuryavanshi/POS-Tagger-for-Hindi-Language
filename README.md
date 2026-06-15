# POS Tagger for Hindi (NLTK)

A Part-of-Speech (POS) tagger for the Hindi language built with [NLTK](https://www.nltk.org/).
It trains a sequence of n-gram taggers with back-off on the
[Universal Dependencies Hindi-HDTB treebank](https://universaldependencies.org/treebanks/hi_hdtb/)
and evaluates tagging accuracy on a held-out test split.

## Overview

POS tagging assigns a grammatical category (noun, verb, adjective, …) to each word
in a sentence. It is a foundational step for downstream NLP tasks such as parsing,
machine translation, and named-entity recognition.

This project implements POS tagging two ways:

| Script | Approach | Data |
|--------|----------|------|
| `NLP_Project.py` | DefaultTagger → Unigram → Bigram → Trigram taggers chained with back-off | UD Hindi-HDTB (`.conllu`) |
| `importnltk.py`  | TnT (Trigrams'n'Tags) statistical tagger | NLTK `indian` corpus (`hindi.pos`) |

The primary, up-to-date implementation is **`NLP_Project.py`**.

## How it works (`NLP_Project.py`)

1. Parse the CoNLL-U dataset files, extracting `(word, UPOS)` pairs per sentence.
2. Build a back-off chain so rarer n-grams fall back to simpler models:
   `TrigramTagger → BigramTagger → UnigramTagger → DefaultTagger('NN')`.
3. Evaluate the trigram tagger's accuracy on the test split.
4. Tokenize a sample Hindi sentence with `indic-nlp-library` and print its tags.

## Dataset

The repository includes the UD Hindi-HDTB treebank splits in CoNLL-U format:

- `hi_hdtb-ud-train.conllu` — training sentences
- `hi_hdtb-ud-test.conllu` — test sentences

Each token line is tab-separated; column 2 is the word form and column 4 is the
universal POS tag.

## Setup

```bash
# Python 3.8+
pip install -r requirements.txt
```

The script downloads the required NLTK data packages (`punkt`, `punkt_tab`,
`indian`, `nonbreaking_prefixes`) automatically on first run.

## Usage

```bash
python NLP_Project.py
```

Example output:

```
Model Accuracy: 88.xx%

Tagged Sentence:
दो/NUM आदमी/NOUN आए/VERB ।/PUNCT
```

(The exact accuracy depends on the installed NLTK version and data.)

To try the alternative TnT-based tagger:

```bash
python importnltk.py
```

## Project structure

```
.
├── NLP_Project.py            # Main n-gram back-off tagger (UD Hindi-HDTB)
├── importnltk.py             # Alternative TnT tagger (NLTK indian corpus)
├── hi_hdtb-ud-train.conllu   # Training data
├── hi_hdtb-ud-test.conllu    # Test data
├── Hindipos.pdf              # Project report
└── requirements.txt
```

## Tech stack

- Python 3.8+
- NLTK 3.8.1
- indic-nlp-library
