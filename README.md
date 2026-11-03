# Malayalam-English Code-Mix Word Identification

Word-level language identification for Manglish (Malayalam written in English
letters) mixed with English. Given a sentence, the system labels each word as
Malayalam, English, Name, or Other.

Example:

    Mammootty padam trailer kidu aanu
    Mammootty -> Name
    padam     -> Malayalam
    trailer   -> English
    kidu      -> Malayalam
    aanu      -> Malayalam

## Why this is hard
Manglish has no fixed spelling (ippo / ipo / ippoo), and some words exist in
both languages, so a plain dictionary does not work.

## Method
1. Data: DravidianCodeMix Malayalam (16,010 YouTube comments); 10,314 comments
   in Roman letters were kept.
2. Rough training labels from rules (English dictionary, name list, Malayalam
   suffixes such as -ettan / -kka). Labels: ML, EN, NE (names), OTHER.
3. Test set: 50 sentences (439 words) labeled by hand (data/gold_fixed.csv).
4. Model: character n-grams (1-4 letters) + logistic regression, with
   spelling normalization (ippo / ipo look alike).
5. Noisy English: about 1,300 fake misspellings of English words added to
   training, so misspelled English is still recognised as English.

## Results (439 hand-labeled words)
| Version | Accuracy |
|---|---|
| Char n-gram baseline | 94.1% |
| + corrected test labels | 94.5% |
| + noisy English training | 95.0% |

- Malayalam spelling variants: 21 / 21 correct
- Misspelled English variants: 16 / 19 correct
- Names: precision 1.00, recall 0.59 (weakest class)

Also tried: capital-letter rule for names (accuracy dropped to 86%) and a
5,000-word English dictionary with typos (dropped to 94.3%).

## Limitations
- The model looks at one word at a time, so it ignores context.
- Names it has not seen are often labeled Malayalam.
- Training labels come from rules and contain some errors.
- Small test set (439 words).

## Future work
Context features (neighbouring words), better name labels, a transformer
model (XLM-R / MuRIL).

## How to run
Python 3.10.11

    pip install -r requirements.txt
    python src/load_data.py
    python src/filter_roman.py
    python src/auto_label.py
    python src/train_noisy.py
    python src/predict.py

Note: data/gold_fixed.csv is included; other data files and trained models are
rebuilt by the scripts above.