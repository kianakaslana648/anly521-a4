### test scorers on the first 100 records of the original file

import pandas as pd
import numpy as np
from matcher.name_matcher import NameMatchScorer, JaccardScorer, LevenshteinScorer, TfidfScorer, JaroScorer

# read the original file

df = pd.read_csv('data/', encoding='utf-8', sep='\t')
names = df['personLabel'].tolist()
aliases = df['aliasLabel'].tolist()
labels = df['label'].tolist()

### exact match
exact_match_scores = [NameMatchScorer(name, alias).score()
                        for i, (name, alias) in enumerate(zip(names, aliases))]
print('### exact match')
print(exact_match_scores)

### jaccard similarity
jaccard_scores = [JaccardScorer(name, alias).score() 
                  for i, (name, alias) in enumerate(zip(names, aliases))]
print('### jaccard similarity')
print(exact_match_scores)

### levenshtein ratio
Levenshtein_scores = [LevenshteinScorer(name, alias).score() 
                      for i, (name, alias) in enumerate(zip(names, aliases))]
print('### levenshtein ratio')
print(Levenshtein_scores)

#### jaro ratio
jaro_scores = [JaroScorer(name, alias).score() 
               for i, (name, alias) in enumerate(zip(names, aliases))]
print('### jaro ratio')
print(jaro_scores)

### tfidf match
tfidf_scorer = TfidfScorer()
tfidf_scorer.fit(names)
tfidf_scores = tfidf_scorer.score(names, aliases)
print('### tfidf match')
print(tfidf_scores)