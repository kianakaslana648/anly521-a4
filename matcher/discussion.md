# Discussion
## Results:
| Methods | Best Threshold | Precision |Recall | F1 Score|
| :---: | :---: | :---: | :---: | :---: |
| Exact Match | 0 | 0.5 | 1 | 0.667 |
| Jaccard Similarity | 0.454 | 0.930 | 0.852 | 0.889 |
| Levenshtein | 0.393 | 0.961 | 0.889 | 0.924
| Tfidf-KNN | 2.23e-16 | 0.998 | 0.871 | 0.930 |
| Jaro | 0.582 | 0.946 | 0.835 | 0.887 |

Precision-recall plots on the training dataset for each algorithm are saved in the **imgs** folder.

The best method according to F1 score is the tfidf-KNN method. The most flexible method is also the tfidf-KNN method. The tfidf-KNN method takes the longest time (1min).

## Influence of Threshold (Part D)
Larger threshold means we have a stricter standard to make a decition that the match exists. For example, in the setting of Jaccard Similarity, a threshold of 1.0 means we need the character set of two strings to be the same to decide a match; a threshold of 0.01 means (Num of intersection character set) must be above 10% of (Num of union character set) to decide a match.

## Comment on Results of Jaccard Similarity (Part E)
According the **Results** table, the optimal threshold for Jaccard Similarity is 0.454, which leads the largest f1-score on the training dataset. The corresponding metrics of precision, recall and f1-score on the testing dataset are: 0.930, 0.852, 0.889. Thus, in the setting of name-alias pair, Jaccard Similarity can perform properly.

## Comment on the Best Thresholds for Different Methods (Part F)
Surely the best thresholds for different methods are most likely to be different, since the modeling methods and thus the result scores are different. 

In this setting, we cannot borrow the best threshold of Jaccard Similarity for the Levenshtein.

## English-Russian Name-Alias Matching (Part K)
1. The names in two languages could be totally not related, in different character sets and different backgrounds. For example, my Chinese name is 蔡明磊(Minglei Cai), but I used to have a English name of Tony, without any special concerns but simply accidentally selecting that name. There are no useful intersection characters and even no inner relationship in the sense of meaning between the two names. So methods related to edit distance, tf-idf, wouldn't be useful in this case.
2. My suggestion for the name-alias match in two languages is using information retrieval methods: simply make the name-alias pair related and make the IR search fast & efficient, to ensure good performances for name-alias pairs without any inner relationship in the literal sense.

## New Approach (Part L Bonus [a])
Jaro distance is kind of distance different from the Levenshtein distance in sense that: Levenshtein distance includes insertions, deletions and substitutions; but Jaro distance only includes insertions and deletions.

By using the method of **Leveshtein.jaro** in the Python package of **Leveshtein**, we are able to gain the Jaro distance of two strings.

The corresponding scorer is already implemented and the results are recorded.


## Multiprocessing (Part L Bonus [b])
By applying **multiprocessing.Pool**, we use multi-threading for the program.

To evaluate the time difference between programs using multi-threading and not using multi-threading, we record the elapsed time for the scoring process.

Results are below: (Number of threads = 4) (CPU frequency: 2.80GHz)

| Methods | Multi-threading Enabled | Multi-threading Disabled|
| :---: | :---: | :---: |
| Exact Match | 1.815s | 0.037s |
| Jaccard Similarity | 1.855s | 0.200s |
| Levenshtein | 1.769s | 0.041s |
| Tfidf-KNN | Much More time than 42s| 42.620s |
| Jaro | 1.839s | 0.0393s |

Python has limited parallelism when using threads. More details on this post:
https://pythonspeed.com/articles/faster-multiprocessing-pickle/

A better choice for multi-procssing would be hadoop or C++.