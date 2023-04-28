import abc
import csv
import collections.abc
import dataclasses
import pickle
import os
import random
from name_matcher import NameMatchScorer
import shutil
from file_read_backwards import FileReadBackwards

@dataclasses.dataclass
class Comparison:
    name1: str
    name2: str
    true_label: bool
    score: float = 0.0
    def __init__(self, name1, name2, true_label, score):
        self.name1 = name1
        self.name2 = name2
        self.score = score
        self.true_label = true_label

class DocIterator(abc.ABC, collections.abc.Iterator):
    def __str__(self):
        return self.__class__.__name__
class TsvIterator(DocIterator):
    """Iterator to iterate over tsv-formatted documents"""
    def __init__(self, path, outpath='tmp'):
        self.path = path
        self.fp = open(self.path, 'r', encoding='utf-8')
        self.reader = csv.reader(self.fp, delimiter='\t')
        next(self.reader) # skip first row
        self.outpath = outpath
        if not os.path.exists(self.outpath):
            os.makedirs(outpath)
        self.outp = open(os.path.join(self.outpath, 'comparison.pkl'), 'wb')
    def __iter__(self):
        return self
    def __next__(self):
        try:
            row = next(self.reader)
            pickle.dump(Comparison(row[0], row[1], row[2], NameMatchScorer(row[0], row[1]).score()), self.outp)
            return Comparison(row[0], row[1], row[2], NameMatchScorer(row[0], row[1]).score())
        except StopIteration:
            self.fp.close()
            self.outp.close()
            raise

class Splitter:
    def __init__(self, path, orig_count):
        self.orig_count = orig_count
        self.count = 0
        self.train_count = 0
        self.test_count = 0
        self.path = path
        file_dirpath = os.path.dirname(path)
        file_name = os.path.basename(path)
        train_file_name = file_name.split('.')[0] + '-train.tsv'
        test_file_name = file_name.split('.')[0] + '-test.tsv'
        self.train_path = os.path.join(file_dirpath, train_file_name)
        self.test_path = os.path.join(file_dirpath, test_file_name)
        self.fp = open(self.path, 'r', encoding='utf-8')
        self.train_fp = open(self.train_path, 'w', encoding='utf-8', newline='')
        self.test_fp = open(self.test_path, 'w', encoding='utf-8', newline='')
        self.reader = csv.reader(self.fp, delimiter='\t')
        self.train_writer = csv.writer(self.train_fp, delimiter='\t')
        self.test_writer = csv.writer(self.test_fp, delimiter='\t')
        self.train_writer.writerow(['personLabel', 'aliasLabel'])
        self.test_writer.writerow(['personLabel', 'aliasLabel'])
        next(self.reader)
        self.train_names = set()
        self.test_names = set()
    def split(self):
        while True:
            try:
                ### add label
                self.count += 1
                row = next(self.reader)
                if self.count <= self.orig_count:
                    row.append(True)
                else:
                    row.append(False)
                    
                ### train-test split
                if (row[0] not in self.train_names) and (row[0] not in self.test_names):
                    choice = random.choices([0,1], weights=[0.8, 0.2])[0]
                    if choice == 0:
                        self.train_names.add(row[0])
                        self.train_writer.writerow(row)
                        self.train_count += 1
                    else:
                        self.test_names.add(row[0])
                        self.test_writer.writerow(row)
                        self.test_count += 1
                elif row[0] in self.train_names:
                    self.train_writer.writerow(row)
                    self.train_count += 1
                else:
                    self.test_writer.writerow(row)
                    self.test_count += 1
            except StopIteration:
                self.fp.close()
                self.train_fp.close()
                self.test_fp.close()
                break
        print("-----")
        print("train data: {}".format(self.train_count))
        print("test data: {}".format(self.test_count))
        print("-----")
        return (self.train_count, self.test_count)
    
class DataCleaner:
    def __init__(self, path, clean_path):
        self.count = 0
        self.path = path
        self.clean_path = clean_path
        file_dirpath = os.path.dirname(path)
        file_name = os.path.basename(path)
        tmp_file_name = file_name.split('.')[0] + '_tmp.tsv'
        tmp_file_path = os.path.join(file_dirpath, tmp_file_name)
        self.tmp_file_path = tmp_file_path
        
        self.fp = open(self.path, 'r', encoding='utf-8')
        self.tmp_fp = open(tmp_file_path, 'w', encoding='utf-8', newline='')
        
        self.fp.readline()
        self.writer = csv.writer(self.tmp_fp, delimiter='\t')
        self.writer.writerow(['personLabel', 'aliasLabel'])
    def clean(self):
        while True:
            line = self.fp.readline()
            if not line:
                break
            if len(line.split('\t')) != 2:
                break
            row = [line.split('\t')[0].strip(), line.split('\t')[1].strip()]
            if row[0]!='' and row[1]!='':
                self.writer.writerow(row)
                self.count += 1
        self.fp.close()
        self.tmp_fp.close()
        shutil.move(self.tmp_file_path, self.clean_path)
        return self.count
    
class DataBalancer:
    def __init__(self, path):
        self.path = path
        file_dirpath = os.path.dirname(path)
        file_name = os.path.basename(path)
        tmp_file_name = file_name.split('.')[0] + '_tmp.tsv'
        tmp_file_path = os.path.join(file_dirpath, tmp_file_name)
        self.tmp_file_path = tmp_file_path
        
        shutil.copy(self.path, self.tmp_file_path)
        
        self.tmp_fp = open(self.tmp_file_path, 'a', encoding='utf-8', newline='')
        self.ordered_fp = open(self.path, 'r', encoding='utf-8')
        self.reversed_fp = FileReadBackwards(self.tmp_file_path, encoding="utf-8")
        self.writer = csv.writer(self.tmp_fp, delimiter='\t')
        self.ordered_fp.readline()
    def balance(self):
        while True:
            ordered_line = self.ordered_fp.readline()
            reversed_line = self.reversed_fp.readline()
            if (not reversed_line) or (not ordered_line):
                self.ordered_fp.close()
                self.reversed_fp.close()
                self.tmp_fp.close()
                break
            row = [ordered_line.split('\t')[0].strip(), reversed_line.split('\t')[1].strip()]
            self.writer.writerow(row)
        os.remove(self.path)
        shutil.move(self.tmp_file_path, self.path)