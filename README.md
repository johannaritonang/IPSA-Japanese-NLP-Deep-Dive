# Water That Sells Itself · Japanese NLP deep dive

Python companion to the Power BI report *Water That Sells Itself* (IPSA The Time R Aqua, Japan social listening, Jul–Dec 2020).
The dashboard answers the brief with a keyword dictionary. This notebook tests and extends it with Japanese NLP.

| Question | Method |
|---|---|
| Did the dictionary miss anything? | Japanese tokenisation (janome + skincare user dictionary, compound nouns), TF-IDF + NMF topics, k chosen by UMass coherence, stability across random restarts (Hungarian matching), topic × theme heatmap, rival-brand counts |
| Which words separate positive posts? | Weighted log-odds with an informative Dirichlet prior (Monroe et al., 2008) |
| Can we trust the vendor's sentiment? | 5-fold cross-validated char n-gram logistic regression + manual audit of every vendor-negative consumer post, with English translations of all 17 complaints |
| Which themes travel together? | Co-occurrence network (lift) with networkx |
| Is the timeline real? | Platform-level daily series to separate collection windows from behaviour |

## Headline results
* Vendor sentiment overstates negativity about 4×: only 8 of 62 "negative" consumer posts criticise the product, and at least 17 posts (1.5%) are critical, not 5.6%.
* Real complaints centre on the core promise: no visible effect, breakouts or irritation, not moisturising enough.
* NMF (9 of 10 topics stable across random restarts) finds a *moisture power vs price* gap, and a direct count shows 7% of consumer posts name a rival brand, which the dictionary does not cover.
* @cosme reviews arrive in three collection windows. Twitter is the honest trend line and still peaks in December (award season).

## Run it
```bash
pip install -r requirements.txt
# put the IPSA export (.xlsx, any name containing "ipsa") in the data folder
jupyter notebook "IPSA NLP Deep Dive.ipynb"
```
The first cell finds the code, the data and the images folders by itself, whether they sit inside this folder or one level up.
If the workbook lives elsewhere, paste its full path into `DATA_OVERRIDE`. Every chart is also saved to the images folder.
A Japanese font is needed for chart labels. Windows (Yu Gothic, Meiryo) and macOS (Hiragino) already have one, and on Linux you can install Noto Sans CJK JP.

## Files
```
IPSA NLP Deep Dive/
├── IPSA NLP Deep Dive.ipynb      the analysis, with outputs
├── IPSA NLP Deep Dive.html       the same notebook as a static page
├── ipsanlp.py                    loading, tokenisation, statistics, chart helpers, audit labels and translations
├── themes.py                     the 33-theme keyword dictionary shared with the Power BI model
├── requirements.txt
├── data/
│   └── manual sentiment audit.csv   hand-labelled audit (row = position in the raw export)
├── images/                       key charts as PNG (rewritten each time the notebook runs)
└── powerbi/
    └── IPSA Water That Sells Itself.pbit   companion Power BI report, insight text aligned with this notebook
```
The raw social-listening export is not included; it belongs to the original exercise.
