# Assignment 4
### StringMatchScorers for name-alias pairs
<br/>

# Tasks
Tasks & Instructions on **Assignment 4.pdf**.

Given dataset returned by **SPARQL** queries, which contains name-alias pairs from wikipedia, we want to evaluate performances of different scoring methods for string matching.

# Methods:
**Exact Match**: 1.0 if total match, 0.0 otherwise.

**Jaccard similarity**: (Num of intersection character set) / (Num of union character set).

**Levenshtein distance**: Edit distance(insertions, deletions or substitutions).

**Tf-idf match**: Fit a tf-idf character-based n-gram vectorizer on the name set; fit a KNN Classifier with the tf-idf name vectors. Given Aliases, use the vectorizer to transform them into vectors; use the KNN classifier to predict the probability of each class. Specifically, use the probability of the paired name as the score.

**Jaro distance**: Edit distance(insertions or deletions).

# Data Preprocessing:
## Data Cleaning & Balancing
### Problems:
1. The dataset given by **SPARQL** queries contains lines with names and empty aliases.
2. The dataset given by **SPARQL** queries only contains correct matches, without any wrong pair-matching.
### Solutions:
1. Delete all the rows that contain empty strings of aliases.
2. Traverse the dataset from start in the order, and from end in the reversed order meanwhile; compose these corresponding elements as pairs; add them as new rows.
### Mention:
These two procedures also work for large dataset.
<br/>

## Data Splitting
### Problems:
1. A procedure to split original dataset into training and testing dataset (0.8:0.2), with no exploding complexity for large datasets. (The proportion of 0.8:0.2 is for names, which means all rows that contain the same name must be in the same set.)
### Solution:
1. We want to complete the process in one time of traverse on the original dataset. We use a training-recording set and a testing-recording set to record the unique names we have traversed. Each time we run into a new row, check whether the name is in training-record set or testing-recording set: if so, directly add them to the corresponding dataset; if not, roll a random number to decide whether it should be in training or testing dataset, add the name to the corresponding recording set, add the row to the corresponding dataset.
### Mention:
This procedure works for large datasets.

## Data Parsing
### Problems:
1. A procedure to parse the original tsv file as files that contain comparison objects.

### Solutions:
1. Use **pickle** to parse tsv files as pkl files. Ensure future implementation of multiprocessing.
### Mention:
The procedure of data parsing could be omitted, since the useful part of the parsed objects only contain names, aliases and matching labels, which could be read from original tsv files. (But it's required by the pdf instructions.)

# Evaluation:
We apply each model to the training dataset to output scores. Then we decide the best threshold according to the F-1 score. With the best threshold, we evaluate on the testing dataset.

A precision-recall plot on the training dataset is saved; best_threshold, precision, recall, f1-score on the testing dataset is (default) printed.

# Explanations of Python Files

main.py: the entrance;

name_matcher.py: implementations of different string matching scorers;

parse_tsv.py: implementations of classes for data cleaning, data balancing, data splitting, data parsing;

eval.py: find the best threshold and output evaluations on the testing dataset;

test_matcher.py: test the scorers on simple examples.

# Usage
```
python main.py -f <path_to_dataset> -s <scoring_algorithm> -e <flag_whether_to_evaluate> -p <flag_whether_to_print_results>
-m <flag_whether_to_use_multiprocessing>
```
path_to_dataset: dataset file path, string, required

scoring_algorithm: must be in ['exact_match', 'jaccard_similarity', 'levenshtein', 'tfidf_match', 'jaro_match'], string, required

flag_whether_to_evaluate: default: True

flag_whether_to_print_results: default: True

flag_whether_to_use_multiprocessing: default: False

**A quick Example:**
```
python main.py -f ../data/names.tsv -s score_match
```

# Discussions:
More discussions in **discussion.md**
