from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import pickle
import numpy as np
from name_matcher import NameMatchScorer, JaccardScorer, LevenshteinScorer, TfidfScorer, JaroScorer
from name_matcher import exact_match_score, jaccard_score, levenshtein_score, jaro_score, tfidf_init_worker, tfidf_score
import os
import matplotlib.pyplot as plt
from typing import List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from multiprocessing.pool import Pool

def evaluationPrint(best_threshold: float, precision: float, recall: float, f1: float, algo: str):
    print('------------------------------------')
    print('Evaluation for {} algorithm:'.format(algo))
    print('Best threshold: {}'.format(best_threshold))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1: {}'.format(f1))
    print('------------------------------------')

def bestThreshold(predicted_scores: List[float], train_labels: List[int], algo: str):
    precision, recall, thresholds = precision_recall_curve(train_labels, predicted_scores)
    f1 = [(2*pre*rec)/(pre+rec) for pre, rec in zip(precision, recall)]
    best_threshold = thresholds[np.argmax(f1)]
    
    plt.plot(precision, recall)
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.xlim([0,1.02])
    plt.ylim([0,1.02])
    plt.title('{}: precision-recall'.format(algo))
    plt.savefig('imgs/{}_pc_curve.png'.format(algo))
    return best_threshold

def evaluate(train_path: str, test_path: str, algo: str, if_print: bool = True, multi: bool = False):
    '''
    train_path: path to the parsed train tsv file
    test_path: path to the parsed test tsv file
    method: scoring algorithm

    returned results: best_threshold, precision, recall, f1 (all on testing dataset)
    best_threshold decided by f1 on training dataset
    '''
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    ### load train data
    train_names = []
    train_aliases = []
    train_labels = []
    train_pkl = open(train_path, 'rb')

    print('start loading train data')
    try:
        while True:
            object_file = pickle.load(train_pkl)
            train_names.append(object_file.name1)
            train_aliases.append(object_file.name2)
            train_labels.append(int(object_file.true_label=='True'))
    except:
        print('complete')
    
    ### load test data
    print('start loading test data')
    test_names = []
    test_aliases = []
    test_labels = []
    test_pkl = open(test_path, 'rb')
    try:
        while True:
            object_file = pickle.load(test_pkl)
            test_names.append(object_file.name1)
            test_aliases.append(object_file.name2)
            test_labels.append(int(object_file.true_label=='True'))
    except:
        print('complete')
    
    if multi:
        num_threads = 4
        num_gap = len(train_names) // num_threads
        train_names_list = [train_names[i:i+num_gap] for i in range(0, len(train_names), num_gap)]
        train_aliases_list = [train_aliases[i:i+num_gap] for i in range(0, len(train_aliases), num_gap)]
        #print([len(i) for i in train_names_list])
    print('###')
    print('algo: {}'.format(algo))
    ### exact match
    if algo == 'exact_match':
        ### best threshold according to f1 score on training dataset
        print('### Scoring Start')
        start_time = time.time()
        if multi:
            print('Multi Threading Enabled')
            with Pool(num_threads) as p:
                predicted_scores = p.map(exact_match_score
                                         , zip(train_names_list, train_aliases_list))
                predicted_scores = [item for sublist in predicted_scores for item in sublist]
                #print(len(predicted_scores), predicted_scores)
        else:
            print('Multi Threading Disabled')
            predicted_scores = [NameMatchScorer(name, alias).score()
                            for i, (name, alias) in enumerate(zip(train_names, train_aliases))]
        end_time = time.time()
        print('Time Elapsed: {}'.format(end_time-start_time))
        print('### Scoring Complete')
        
        best_threshold = bestThreshold(predicted_scores, train_labels, algo)

        ### evaluate on testing dataset
        predicted_labels = [int(NameMatchScorer(name, alias).score() >= best_threshold)
                        for i, (name, alias) in enumerate(zip(test_names, test_aliases))]
        precision = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)
        evaluationPrint(best_threshold, precision, recall, f1, algo)
        return best_threshold, precision, recall, f1
    
    ### jaccard similarity
    elif algo == 'jaccard_similarity':
        ### best threshold according to f1 score on training dataset
        print('### Scoring Start')
        start_time = time.time()
        if multi:
            print('Multi Threading Enabled')
            with Pool(num_threads) as p:
                predicted_scores = p.map(jaccard_score
                                         , zip(train_names_list, train_aliases_list))

        else:
            print('Multi Threading Disabled')
            predicted_scores = [JaccardScorer(name, alias).score()
                        for i, (name, alias) in enumerate(zip(train_names, train_aliases))]
        end_time = time.time()
        print('Time Elapsed: {}'.format(end_time-start_time))
        print('### Scoring Complete')

        best_threshold = bestThreshold(predicted_scores, train_labels, algo)

        ### evaluate on testing dataset
        predicted_labels = [int(JaccardScorer(name, alias).score() >= best_threshold)
                        for i, (name, alias) in enumerate(zip(test_names, test_aliases))]
        precision = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)
        evaluationPrint(best_threshold, precision, recall, f1, algo)
        return best_threshold, precision, recall, f1
    
    ### levenshtein ratio
    elif algo == 'levenshtein':
        ### best threshold according to f1 score on training dataset
        print('### Scoring Start')
        start_time = time.time()
        if multi:
            print('Multi Threading Enabled')
            with Pool(num_threads) as p:
                predicted_scores = p.map(levenshtein_score
                                         , zip(train_names_list, train_aliases_list))
                predicted_scores = [item for sublist in predicted_scores for item in sublist]
        else:
            print('Multi Threading Disabled')
            predicted_scores = [LevenshteinScorer(name, alias).score()
                            for i, (name, alias) in enumerate(zip(train_names, train_aliases))]
        end_time = time.time()
        print('Time Elapsed: {}'.format(end_time-start_time))
        print('### Scoring Complete')

        best_threshold = bestThreshold(predicted_scores, train_labels, algo)

        ### evaluate on testing dataset
        predicted_labels = [int(LevenshteinScorer(name, alias).score() >= best_threshold)
                        for i, (name, alias) in enumerate(zip(test_names, test_aliases))]
        precision = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)
        evaluationPrint(best_threshold, precision, recall, f1, algo)
        return best_threshold, precision, recall, f1
    
    ### tfidf match
    elif algo == 'tfidf_match':
        ### best threshold according to f1 score on training dataset
        print('### Scoring Start')
        start_time = time.time()
        tfidf_scorer = TfidfScorer()
        tfidf_scorer.fit(train_names)
        if multi:
            print('Multi Threading Enabled')
            with Pool(num_threads, initializer=tfidf_init_worker, 
                      initargs=(tfidf_scorer.vectorizer,tfidf_scorer.neigh,
                                 tfidf_scorer.vocab_index)) as p:
                predicted_scores = p.map(tfidf_score
                                         , zip(train_names_list, train_aliases_list))
                predicted_scores = [item for sublist in predicted_scores for item in sublist]
        else:
            print('Multi Threading Disabled')
            predicted_scores = tfidf_scorer.score(train_names, train_aliases)
        end_time = time.time()
        print('Time Elapsed: {}'.format(end_time-start_time))
        print('### Scoring Complete')

        best_threshold = bestThreshold(predicted_scores, train_labels, algo)

        ### evaluate on testing dataset
        tfidf_scorer = TfidfScorer()
        tfidf_scorer.fit(test_names)
        predicted_scores = tfidf_scorer.score(test_names, test_aliases)
        predicted_labels = [int(score >= best_threshold) for score in predicted_scores]
        precision = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)
        if if_print:
            evaluationPrint(best_threshold, precision, recall, f1, algo)
        return best_threshold, precision, recall, f1

    ### jaro match
    elif algo == 'jaro_match':
        ### best threshold according to f1 score on training dataset
        print('### Scoring Start')
        start_time = time.time()
        if multi:
            print('Multi Threading Enabled')
            with Pool(num_threads) as p:
                predicted_scores = p.map(jaro_score
                                         , zip(train_names_list, train_aliases_list))
                predicted_scores = [item for sublist in predicted_scores for item in sublist]
        else:
            print('Multi Threading Disabled')
            predicted_scores = [JaroScorer(name, alias).score()
                        for i, (name, alias) in enumerate(zip(train_names, train_aliases))]
        end_time = time.time()
        print('Time Elapsed: {}'.format(end_time-start_time))
        print('### Scoring Complete')

        best_threshold = bestThreshold(predicted_scores, train_labels, algo)

        ### evaluate on testing dataset
        predicted_labels = [int(JaroScorer(name, alias).score() >= best_threshold)
                        for i, (name, alias) in enumerate(zip(test_names, test_aliases))]
        precision = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)
        evaluationPrint(best_threshold, precision, recall, f1, algo)
        return best_threshold, precision, recall, f1
    
    else:
        raise ValueError("Invalid algorithm")