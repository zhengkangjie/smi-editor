# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
from functools import lru_cache
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ProLMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = lmdb.open(
            self.db_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        self.is_protein = 'seq' in pickle.loads(env.begin().get(f"{0}".encode("ascii")))

        self.aa_len  = None
        self.build_aa_len = False
        if self.is_protein and not os.path.isfile(self.db_path+'.aa_len'):
            logger.info(self.db_path+'.aa_len' + ' does not exist! Rebuild amino acids length array.')
            self.build_aa_len = True

        self.build_sizes = not os.path.isfile(self.db_path+'.sizes')
        if self.build_sizes:
            logger.info(self.db_path+'.sizes' + ' does not exist! Rebuild sizes array.')
        if self.build_sizes or self.build_aa_len:
            self._sizes = np.zeros((len(self._keys),), dtype=np.int32)
            if self.is_protein:
                self.aa_len = [None for _ in range(len(self._keys))]
            with env.begin() as txn:
                with txn.cursor() as cursor:
                    for key, value in cursor.iternext():
                        n = int(key.decode())
                        if self.is_protein:
                            item = pickle.loads(value)
                            self._sizes[n] = len(item['seq'])
                            atoms = item['atoms']
                            self.aa_len[n] = np.array([len(aa) for aa in atoms])
                        else:
                            self._sizes[n] = len(pickle.loads(value)['atoms'])
            with open(self.db_path+'.sizes', 'wb') as fout:
                pickle.dump(self._sizes, fout)
            if self.is_protein:
                with open(self.db_path+'.aa_len', 'wb') as fout:
                    pickle.dump(self.aa_len, fout)
        else:
            with open(self.db_path+'.sizes', 'rb') as fin:
                self._sizes = pickle.load(fin)
            if self.is_protein:
                with open(self.db_path+'.aa_len', 'rb') as fin:
                    self.aa_len = pickle.load(fin)
        env.close()

    def __getstate__(self):
        state = self.__dict__
        state["db_txn"] = None
        return state
    
    def _set_db_txn(self):
        env = lmdb.open(
            self.db_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        self.db_txn = env.begin(write=False)

    def __setstate__(self, state):
        self.__dict__ = state
        self._set_db_txn()
        
    def __len__(self):
        return len(self._keys)

    @property
    def sizes(self):
        return self._sizes

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "db_txn") or self.db_txn is None:
            self._set_db_txn()
        datapoint_pickled = self.db_txn.get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data

    def get_aa_len(self, idx):
        if self.aa_len is None:
            return None
        return self.aa_len[idx]
