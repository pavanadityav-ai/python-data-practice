# Python Data Practice

Small Python and NumPy exercises I wrote while learning data analysis for machine learning.

## What the script does

`evaluation.py` works on a small sample of speech-recognition style results (Telugu words). For each audio clip it has the duration, the predicted word and the actual word.

The script:

1. Reads the data from a CSV file into a list of dictionaries.
2. Calculates **accuracy**: the share of predictions that match the actual word.
3. Does **error analysis**: lists which clips were predicted wrongly.
4. Calculates **duration statistics**: the average, the longest clip, and the clips longer than 4 seconds.

## Example output

```
Accuracy: 75.0%
Wrong: ['tel_002']
Average: 4.0
Longest: tel_003
Longer than 4: ['tel_001' 'tel_003'] (2)
```

## Concepts used

- Functions, loops and running totals
- Reading CSV files with `csv.DictReader`
- Converting text to numbers with `float()`
- NumPy arrays, boolean masks, `mean()` and `argmax()`

## How to run

```
pip install numpy
python evaluation.py
```
