import re
import os
from uuid import uuid4

bash = lambda s: os.popen(s).read().rstrip().split('\n')

from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator

class CDHIT(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(self, threshold, word_length):
        self.threshold = threshold
        self.word_length = word_length
        self.labels_ = None
        self._working_dir = '/dev/shm'
        self._cdhit_path = os.path.dirname(os.path.abspath(__file__)) + '/../bin'

    def fit(self, X):
        sequences = X
        fn = str(uuid4())
        
        with open(f'{working_dir}/{fn}.fasta','w') as f:
            for sequence in sequences:
                f.write(f'>{sequence}\n{sequence}\n')

        cdhit_params = f'-i {self.working_dir}/{fn}.fasta -o {self.working_dir}/{fn}'
        cdhit_params += f'-M 0 -c {threshold} -d 0 -l {word_length} -g 1
        bash(f'cd {self._cdhit_path} && ./cd-hit {cdhit_params}')

        with open(f'{working_dir}/{fn}.clstr','r') as f:
            labels = {}
            for line in f:
                line = line.strip()
                if line[0] == '>':
                    cluster = line[1:]
                else:
                    sequence = re.search(r'[^^]>([^\.]+)',line)[1]
                    labels[sequence] = cluster

        self.labels_ = [labels[s] for s in sequences]
        
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_