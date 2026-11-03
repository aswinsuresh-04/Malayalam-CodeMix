# Malayalam-English Code-Mix Word Identification

Word-level language identification for Manglish (Malayalam written in English
letters) mixed with English. Given a sentence, the system labels each word as
Malayalam, English, Name, or Other.

Example:

    game start avan pokuva
    game    -> English
    start   -> English
    avan    -> Malayalam
    pokuva  -> Malayalam

## Why this is hard
Manglish has no fixed spelling (ippo / ipo / ippoo), and some words exist in
both languages, so a plain dictionary does not work.

## Method
1. Data: DravidianCodeMix Malayalam (16,010 YouTube comments); 10,314 comments
   in Roman letters were kept.
2. Rough training labels from rules: English words that people actually use
   (checked by word frequency), Malayalam suffixes (-ettan, -kka), a small name
   list, and short slang (da, di). Labels: ML, EN, NE (names), OTHER.
3. Test set: 50 sentences (439 words) labeled by hand (data/gold_fixed.csv).
4. Model: character n-grams (1-4 letters) + logistic regression, with spelling
   normalization (ippo / ipo look alike).
5. Extra English training data: about 6,000 common English words (by frequency)
   plus fake misspellings of English words, so misspelled English is still
   recognised as English.
6. predict.py also handles Malayalam script (always Malayalam), numbers, and the
   single-letter English words "a" and "i".

## Results (439 hand-labeled words)
| Version | Accuracy |
|---|---|
| Char n-gram baseline | 94.1% |
| + corrected test labels | 94.5% |
| + noisy English training | 95.0% |
| v2: frequency-based labels + common English words (final) | 94.5% |

v2 scores slightly lower than the previous version on this test set (about 2
words), but the test set is all movie comments. On 16 informal office/tech
sentences, v2 fixed 8 mislabeled words (aa, aam, deadline, prioritize, database,
rescheduled) and broke none.

- Malayalam spelling variants: 21 / 21 correct
- Misspelled English variants: 16 / 19 correct
- Names: precision 1.00, recall 0.59 (weakest class)

## Experiments that did not help
- Capital-letter rule for names: accuracy dropped to 86%.
- 5,000 random dictionary words with typos: dropped to 94.3%.
- Previous/next word as features: test accuracy rose to 95.2%, but English words
  between Malayalam words were mislabeled, and names did not improve.

## Limitations
- The final model looks at one word at a time, so it ignores context.
- Names it has not seen are often labeled Malayalam or English (for example
  "Shyam").
- Training labels come from rules and contain some errors.
- Small test set (439 words), all from movie comments.

## Future work
Better name labels, then a pretrained transformer (XLM-R / MuRIL) that reads
the whole sentence.

## How to run
Python 3.10.11

    pip install -r requirements.txt
    python -c "import nltk; nltk.download('words'); nltk.download('stopwords')"
    mkdir results
    python src/load_data.py
    python src/filter_roman.py
    python src/auto_label.py
    python src/train_v2.py
    python src/predict.py

data/gold_fixed.csv is included. Other data files and trained models are rebuilt
by the scripts above (they are not stored in the repo).