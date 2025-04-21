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


class LMDBDataset:
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
        self.is_protein = 'seq' in pickle.loads(env.begin().get(f"{0}".encode("ascii")))
        if not os.path.isfile(self.db_path+'.sizes'):
            with env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
            logger.info(self.db_path+'.sizes' + ' does not exist! Rebuild sizes array.')
            # for k in range(len(self._keys)):
            #     if self.is_protein:
            #         self._sizes.append(len(pickle.loads(env.begin().get(f"{k}".encode("ascii")))['seq']))
            #     else:
            #         self._sizes.append(len(pickle.loads(env.begin().get(f"{k}".encode("ascii")))['atoms']))
            # self._sizes = np.array(self._sizes)
            
            self._sizes = np.zeros((len(self._keys),), dtype=np.int32)
            # self._sizes = np.zeros((137159966,), dtype=np.int32)
            with env.begin() as txn:
                with txn.cursor() as cursor:
                    for key, value in cursor.iternext():
                        n = int(key.decode())
                        if self.is_protein:
                            self._sizes[n] = len(pickle.loads(value)['seq'])
                        else:
                            self._sizes[n] = len(pickle.loads(value)['smi'])
            with open(self.db_path+'.sizes', 'wb') as fout:
                pickle.dump(self._sizes, fout)
        else:
            with open(self.db_path+'.sizes', 'rb') as fin:
                self._sizes = pickle.load(fin)
            # self._keys = list(range(len(self._sizes)))
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
        return len(self._sizes)
    # def __len__(self):
    #     return 137159966

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
