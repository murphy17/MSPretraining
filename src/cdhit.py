import re
import os
from uuid import uuid4

bash = lambda s: os.popen(s).read().rstrip().split('\n')

from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from sklearn.model_selection import train_test_split

class CDHIT(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(self, threshold, word_length):
        self.threshold = threshold
        self.word_length = word_length
        self.labels_ = None
        self._working_dir = '/dev/shm'
        self._cdhit_path = os.path.dirname(os.path.abspath(__file__)) + '/../bin'

    def fit(self, X):
        sequences = sorted(set(X))
        fn = str(uuid4())
        
        with open(f'{self._working_dir}/{fn}.fasta','w') as f:
            for sequence in sequences:
                f.write(f'>{sequence}\n{sequence}\n')

        cdhit_params = f'-i {self._working_dir}/{fn}.fasta -o {self._working_dir}/{fn} '
        cdhit_params += f'-M 0 -T 0 -c {self.threshold} -d 0 -n {self.word_length} -l {self.word_length}'
#         if not self.verbose:
#             cdhit_params += '> /dev/null '
        bash(f'cd {self._cdhit_path} && ./cd-hit {cdhit_params}')

        with open(f'{self._working_dir}/{fn}.clstr','r') as f:
            labels = {}
            for line in f:
                line = line.strip()
                if line[0] == '>':
                    cluster = int(line.split(' ')[1])
                else:
                    sequence = re.search(r'[^^]>([^\.]+)',line)[1]
                    labels[sequence] = cluster

        self.labels_ = [labels[s] for s in X]
        
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
def cdhit_split(
    sequences,
    *args,
    split=None,
    threshold=None,
    word_length=None,
    random_state=0
):
    cdhit = CDHIT(
        threshold=threshold,
        word_length=word_length
    )
    clusters = cdhit.fit_predict(sequences)
    train_clusters, test_clusters = train_test_split(
        list(set(clusters)),
        train_size=split,
        random_state=random_state,
        shuffle=True
    )
    train_clusters = set(train_clusters)
    test_clusters = set(test_clusters)
    train_sequences = []
    test_sequences = []
    train_args = [[] for _ in args]
    test_args = [[] for _ in args]
    for i, (s,c) in enumerate(zip(sequences,clusters)):
        if c in train_clusters:
            train_sequences.append(s)
            for a,arg in enumerate(args):
                train_args[a].append(arg[i])
        elif c in test_clusters:
            test_sequences.append(s)
            for a,arg in enumerate(args):
                test_args[a].append(arg[i])
    return train_sequences, test_sequences, *train_args, *test_args