# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset, NestedDictionaryDataset, SortDataset, Dictionary, RightPadDataset, NumelDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    load_langpair_dataset,
)
from fairseq.utils import new_arange
import os
import numpy as np

from fairseq.data import (
    ConcatDataset,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    EpochShuffleDataset,
)

from fairseq.data.ai4sci import  (
    LMDBDataset,
    KeyDataset,
    TokenizeDataset,
    SMILESDropNewCombDataset,
    SMILESTokenizeDataset,
)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import logging
logger = logging.getLogger(__name__)

NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])


@dataclass
class TranslationLevenshteinConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={"help": "type of noise"},
    )
    smi_dict: str = field(
        default='smi_dict.txt',
        metadata={"help": "smi_dict"},
    )
    BRICKS_drop_rate: float = field(
        default=0.2,
        metadata={"help": "BRICKS_drop_rate"},
    )
    atoms_drop_rate: float = field(
        default=0.25,
        metadata={"help": "atoms_drop_rate"},
    )
    BRICKS_sample_policy_ratio: float = field(
        default=0.5,
        metadata={"help": "BRICKS_sample_policy_ratio"},
    )
    seed: int = field(
        default=1,
        metadata={"help": "seed"},
    )

@register_task("translation_lev_smi", dataclass=TranslationLevenshteinConfig)
class TranslationLevenshteinSMITask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: TranslationLevenshteinConfig

    def __init__(self, cfg: TranslationLevenshteinConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.mask_idx = self.src_dict.add_symbol("[MASK]")

    @classmethod
    def load_only_mol_dict(cls, cfg):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary(add_special_symbols=False)
        with open(os.path.join(paths[0], cfg.smi_dict), 'r') as fin:
            for idx, line in enumerate(fin):
                sym = line.strip().split()[0].strip()
                dictionary.add_symbol(sym)
        dictionary.bos_index = dictionary.index('[CLS]')
        dictionary.pad_index = dictionary.index('[PAD]')
        dictionary.eos_index = dictionary.index('[SEP]')
        dictionary.unk_index = dictionary.index('[UNK]')
        dictionary.add_symbol("[MASK]")
        logger.info("Molecules dictionary: {} types".format(len(dictionary)))
        return dictionary

    @classmethod
    def setup_task(cls, cfg: TranslationLevenshteinConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        dictionary_m = cls.load_only_mol_dict(cfg)

        return cls(cfg, dictionary_m, dictionary_m)

    def _load_mols_dataset_split(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split = split + '.lmdb'
        split_path = os.path.join(data_path, split)

        dataset = LMDBDataset(split_path)

        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        logger.info("loaded {} molecules samples from: {}".format(len(dataset), split_path))

        return dataset

    def _load_mol_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        raw_dataset = self._load_mols_dataset_split(split, epoch, combine)

        def one_dataset(raw_dataset):
            smi_dataset = KeyDataset(raw_dataset, "smi")

            currpt_smi_dataset = SMILESDropNewCombDataset(smi_dataset,
                BRICKS_drop_rate=self.cfg.BRICKS_drop_rate,
                atoms_drop_rate=self.cfg.atoms_drop_rate,
                BRICKS_sample_policy_ratio=self.cfg.BRICKS_sample_policy_ratio
            )

            target_dataset = KeyDataset(currpt_smi_dataset, "origin")
            target_dataset = SMILESTokenizeDataset(target_dataset, self.tgt_dict, self.cfg.max_source_positions)

            dropped_smi_dataset = KeyDataset(currpt_smi_dataset, "dropped")
            dropped_smi_dataset = SMILESTokenizeDataset(dropped_smi_dataset, self.tgt_dict, self.cfg.max_source_positions)

            input_dict = {
                "src_tokens": RightPadDataset(
                    dropped_smi_dataset,
                    pad_idx=self.tgt_dict.pad(),
                ),
                "src_lengths": NumelDataset(dropped_smi_dataset, reduce=False),
            }
            target_dataset = RightPadDataset(
                target_dataset, pad_idx=self.tgt_dict.pad()
            )

            return input_dict, target_dataset

        net_input, target = one_dataset(raw_dataset)
        # src_dataset = net_input['src_tokens']
        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": net_input,
                "target": target,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(target, reduce=True),
            },
            sizes=[target.sizes],
        )
        return dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset = self._load_mol_dataset(split, epoch, combine)
        
        # with data_utils.numpy_seed(self.cfg.seed + epoch - 1):
        #     shuffle = np.random.permutation(len(dataset))

        # self.datasets[split] = SortDataset(
        #     dataset, sort_order=[shuffle, dataset.sizes]
        # )

        # dataset = EpochShuffleDataset(dataset, len(dataset), self.cfg.seed)

        self.datasets[split] = EpochShuffleDataset(dataset, len(dataset), self.cfg.seed)


        logger.info("totally loaded {} samples for {} set".format(len(dataset), split))

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        sample["prev_target"] = self.inject_noise(sample["net_input"]["src_tokens"])
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample["prev_target"] = self.inject_noise(sample["net_input"]["src_tokens"])
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
