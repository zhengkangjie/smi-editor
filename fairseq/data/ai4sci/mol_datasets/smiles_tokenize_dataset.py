# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import torch
from fairseq.data import Dictionary
from functools import lru_cache
from fairseq.data import BaseWrapperDataset
import numpy as np
import re

pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

class SMILESTokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary: Dictionary,
        max_seq_len: int=1024,
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len
        self.regex = re.compile(pattern)

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        raw_data = self.dataset[index]
        token_smi = [token for token in self.regex.findall(raw_data)]
        if len(token_smi) > self.max_seq_len:
            token_smi = token_smi[:self.max_seq_len]
        # assert len(token_smi) < self.max_seq_len and len(token_smi) > 0
        # return torch.from_numpy(self.dictionary.vec_index(raw_data)).long()
        new_data = np.array([self.dictionary.index(x) for x in token_smi], dtype=np.int32)
        return torch.from_numpy(new_data)