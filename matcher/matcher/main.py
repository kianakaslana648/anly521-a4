
from utils.parse_tsv import TsvIterator, Splitter, DataCleaner, DataBalancer
from eval.eval import evaluate
import argparse
import os
import time 
def main_process(filepath: str, algo: str, eval: bool, if_print: bool, multi: bool):
    ### argument checking
    if not os.path.exists(filepath):
        raise FileNotFoundError("File not found")
    if algo not in ['exact_match', 'jaccard_similarity', 'levenshtein',
                    'tfidf_match', 'jaro_match']:
        raise ValueError("Invalid algorithm")
    
    ### file path
    file_dirpath = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)
    clean_file_name = file_name.split('.')[0] + '-processed.tsv'
    clean_filepath = os.path.join(file_dirpath, clean_file_name)

    ### original file cleaning
    print('start cleaning')
    cleaner = DataCleaner(filepath, clean_filepath)
    orig_count = cleaner.clean()
    print('cleaning complete')

    ### original file balancing (adding wrong matches)
    print('start balancing')
    balancer = DataBalancer(clean_filepath)
    balancer.balance()
    print('balancing complete')
    
    ### train-test splitting
    print('start splitting')
    splitter = Splitter(clean_filepath, orig_count)
    train_count, test_count = splitter.split()
    print('splitting complete')

    ### tsv parser
    print('start tsv-parsing')
    train_file_name = clean_file_name.split('.')[0] + '-train.tsv'
    train_filepath = os.path.join(file_dirpath, train_file_name)
    test_file_name = clean_file_name.split('.')[0] + '-test.tsv'
    test_filepath = os.path.join(file_dirpath, test_file_name)
    train_pkl_dirpath = 'train-tmp'
    test_pkl_dirpath = 'test-tmp'
    ##### train tsv parser
    train_tsv_parser = TsvIterator(train_filepath, train_pkl_dirpath)
    try:
        while True:
            next(train_tsv_parser)
    except:
        print('tsv-parsing complete')
    ##### test tsv parser
    test_tsv_parser = TsvIterator(test_filepath, test_pkl_dirpath)
    try:
        while True:
            next(test_tsv_parser)
    except:
        print('tsv-parsing complete')

    ### evaluate
    train_pkl_path = os.path.join(train_pkl_dirpath, 'comparison.pkl')
    test_pkl_path = os.path.join(test_pkl_dirpath, 'comparison.pkl')
    best_threshold, precision, recall, f1 = evaluate(train_pkl_path, test_pkl_path, 
                                                     algo, if_print, multi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to make corpus "
                                                 "for String Matcher")
    parser.add_argument("-f", "--filepath", required=True, help="Data directory")
    parser.add_argument("-s", "--algo", required=True, help='Score algorithm')
    parser.add_argument("-e", "--eval", default=True, 
                        help="Flag to evaluate")
    parser.add_argument("-p", "--print", default=True,
                         help='Flag to print results')
    parser.add_argument("-m", "--multi", default=False,
                        help='Flag to use multiprocessing')
    args = parser.parse_args()
    args.multi = True if args.multi == 'True' else False
    main_process(args.filepath, args.algo, args.eval, args.print, args.multi)
    #print("Multiprocessing used: {}".format(args.multi))
    #print("Time taken: {} seconds".format(end_time - start_time))