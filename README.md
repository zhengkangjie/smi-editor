# SMI-Editor: Edit-based SMILES Language Model with Fragment-level Supervision

This is the official implementation of our ICLR 2025 paper:  
**SMI-Editor: Edit-based SMILES Language Model with Fragment-level Supervision** ([[Paper Link]](https://openreview.net/pdf?id=M29nUGozPa))

## Abstract

SMILES, a crucial textual representation of molecular structures, has garnered significant attention as a foundation for pre-trained language models (LMs). However, most existing pre-trained SMILES LMs focus solely on single-token level supervision during pre-training, failing to fully leverage the substructural information of molecules. This limitation makes the pre-training task overly simplistic and prevents the models from capturing richer molecular semantic information. Moreover, these LMs are typically trained only on corrupted SMILES, leading to a train-inference mismatch.

To address these challenges, we propose SMI-Editor, a novel edit-based pre-trained SMILES LM. SMI-Editor disrupts substructures within a molecule and trains the model to reconstruct the original SMILES via an editing process. This method introduces fragment-level training signals and enables training with valid SMILES as inputs, enhancing molecular understanding. Our experiments show that SMI-Editor outperforms existing SMILES LMs and even several 3D molecular representation models across a range of downstream tasks.

## Dependencies

- `rdkit`
- `lmdb`

We recommend installing dependencies with:

```bash
pip install rdkit lmdb
```

## Installation

To install the project and compile custom CUDA operations used in the Levenshtein Transformer, run:

```bash
python setup.py build_ext --inplace
python setup.py install
```

## Pre-training Data

We adopt the same training data from [Uni-Mol](https://openreview.net/forum?id=6K2RM6wVqKu).


| Data                     | File Size  | Update Date | Download Link                                                                                                             | 
|--------------------------|------------| ----------- |---------------------------------------------------------------------------------------------------------------------------|
| molecular pretrain       | 114.76GB   | Jun 10 2022 |https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/ligands.tar.gz                                |

Make sure to include the `smi_dict_token.txt` in your dataset directory. Example layout:

```
your_dataset/
├── train.lmdb
├── valid.lmdb
├── smi_dict_token.txt
└── ...
```
You can download our released checkpoints at:  
👉 **[google drive](https://drive.google.com/file/d/1Yqk3qTUTvE_U_qJvhjYI4M54CkTwOTU7/view?usp=sharing)**

### Example Pre-training Command

Number of GPUs: 4

```bash
data_path="/path/to/your/data"  # Replace with your data path
save_dir="/path/to/save/data"   # Replace with your save path
logfile="${save_dir}/train.log"

fairseq-train ${data_path} \
  --save-dir ${save_dir} \
  --ddp-backend=no_c10d --fp16 \
  --task translation_lev_smi \
  --criterion nat_loss \
  --arch levenshtein_encoder \
  --smi-dict smi_dict_token.txt \
  --BRICKS-sample-policy-ratio 1 \
  --max-source-positions 1024 \
  --max-target-positions 1024 \
  --tensorboard-logdir ${save_dir}/tsb \
  --encoder-layers 12 \
  --encoder-embed-dim 768 \
  --encoder-ffn-embed-dim 3072 \
  --encoder-attention-heads 12 \
  --label-smoothing 0.1 \
  --attention-dropout 0.1 \
  --activation-dropout 0 \
  --dropout 0.1 \
  --noise no_noise \
  --skip-invalid-size-inputs-valid-test \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 5e-4 --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 --warmup-updates 10000 \
  --max-update 120000 \
  --weight-decay 0.0 --clip-norm 0.1 \
  --max-tokens 4000 --update-freq 2 \
  --no-progress-bar --log-format 'simple' --log-interval 100 \
  --fixed-validation-seed 7 \
  --seed 1 \
  --save-interval-updates 20000 \
  --no-epoch-checkpoints \
  --fp16-scale-tolerance 0.1 > ${logfile} 2>&1
```

## Evaluation

To evaluate SMI-Editor on MoleculeNet or other downstream molecular property prediction tasks, we recommend following the evaluation protocol used in [Uni-Mol](https://github.com/dptech-corp/Uni-Mol).


## Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{zheng2025smieditor,
  title={SMI-Editor: Edit-based SMILES Language Model with Fragment-level Supervision},
  author={Zheng, Kangjie and Liang, Siyue and Yang, Junwei and Feng, Bin and Liu, Zequn and Ju, Wei and Xiao, Zhiping and Zhang, Ming},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
