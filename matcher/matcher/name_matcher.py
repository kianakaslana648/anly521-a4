import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

class NameMatchScorer:
    def __init__(self, name1, name2, threshold=0.5):
        self.name1 = name1
        self.name2 = name2
        self.threshold = threshold
    def __str__(self):
        rep = 'Name1: ' + self.name1 + '\t' + 'Name2:' + self.name2
        return rep
    def __repr__(self):
        rep = 'NameMatchScorer(' + self.name1 + ', ' + self.name2 + ')'
        return rep
    def score(self):
        if self.name1 == self.name2:
            return 1.0
        else:
            return 0.0
        
class JaccardScorer(NameMatchScorer):
    def score(self):
        set1 = set([*self.name1])
        set2 = set([*self.name2])
        inter_num = len(set1.intersection(set2)) * 1.0
        union_num = len(set1.union(set2)) * 1.0
        return inter_num / union_num

class LevenshteinScorer(NameMatchScorer):
    def score(self):
        return Levenshtein.ratio(self.name1, self.name2)

class TfidfScorer(NameMatchScorer):
    def __init__(self, ngram_range=(1,4), n_neighbors=5):
        ### type check
        if not (isinstance(ngram_range, tuple) and len(ngram_range)==2):
            raise TypeError('ngram_range must be a 2-tuple')
        if not (isinstance(n_neighbors, int) and n_neighbors > 0):
            raise ValueError('n_neighbors must be a positive integer')
        
        ### initialize vectorizer and knn classifier
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine', weights='distance')
    
    def fit(self, train_names):
        self.vocab_index = {}
        num = 0
        for name in train_names:
            if name not in self.vocab_index.keys():
                self.vocab_index[name] = num
                num += 1
        name_keys = self.vocab_index.keys()
        self.vectorizer.fit(name_keys)
        self.neigh.fit(self.vectorizer.transform(name_keys), list(self.vocab_index.values()))

    def score(self, train_names, aliases):
        alias_vectors = self.vectorizer.transform(aliases)
        proba = self.neigh.predict_proba(alias_vectors)
        predicted_scores = []
        for i in range(proba.shape[0]):
            predicted_scores.append(proba[i, self.vocab_index[train_names[i]]])
        return predicted_scores

class JaroScorer(NameMatchScorer):
    def score(self):
        return Levenshtein.jaro(self.name1, self.name2)
    def score_for_mp(self, name, alias):
        return Levenshtein.jaro(name, alias)


### multi-threading version
def exact_match_score(pair):
    predicted_scores = []
    for name, alias in zip(pair[0], pair[1]):
        if name == alias:
            predicted_scores.append(1.0)
        else:
            predicted_scores.append(0.0)
    return predicted_scores
    
def jaccard_score(pair):
    predicted_scores = []
    for name, alias in zip(pair[0], pair[1]):
        set1 = set([*name])
        set2 = set([*alias])
        inter_num = len(set1.intersection(set2)) * 1.0
        union_num = len(set1.union(set2)) * 1.0
        predicted_scores.append(inter_num / union_num)
    return predicted_scores

def levenshtein_score(pair):
    predicted_scores = []
    for name, alias in zip(pair[0], pair[1]):
        predicted_scores.append(Levenshtein.ratio(name, alias))
    return predicted_scores

def jaro_score(pair):
    predicted_scores = []
    for name, alias in zip(pair[0], pair[1]):
        predicted_scores.append(Levenshtein.jaro(name, alias))
    return predicted_scores

def tfidf_init_worker(global_vectorizer, global_neigh, global_vocab_index):
    global vectorizer
    global neigh
    global vocab_index
    vectorizer = global_vectorizer
    neigh = global_neigh
    vocab_index = global_vocab_index

def tfidf_score(pair):
    predicted_scores = []
    global vectorizer
    global neigh
    global vocab_index
    for name, alias in zip(pair[0], pair[1]):
        predicted_scores.append(neigh.predict_proba(vectorizer.transform([alias]))[0, vocab_index[name]])
    return predicted_scores