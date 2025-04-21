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
import random
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import RWMol

def bricks_drop(smiles, fraction_to_remove=0.2):
    mol = Chem.MolFromSmiles(smiles)

    # 找到需要断开的BRICS键
    brics_bonds = BRICS.FindBRICSBonds(mol)
    brics_bonds = [x[0] for x in brics_bonds]

    # 记录需要断开的BRICS键信息，并为每个断键分配唯一的哑原子标记
    broken_bonds = []
    bond_indices_set = set()  # 用于跟踪已添加的键索引
    for idx, (atom_idx1, atom_idx2) in enumerate(brics_bonds):
        bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
        bond_idx = bond.GetIdx()
        bond_type = bond.GetBondType()
        # 使用同位素标记来区分每个断键
        dummy_label = idx + 1
        broken_bonds.append({
            'bond_idx': bond_idx,
            'atom_idx1': atom_idx1,
            'atom_idx2': atom_idx2,
            'bond_type': bond_type,
            'dummy_label': (dummy_label, dummy_label)
        })
        bond_indices_set.add(bond_idx)

    # 在BRICS键之后，找到环与侧链之间的键并添加到断键列表中
    # 首先，获取所有的键
    current_dummy_label = len(broken_bonds) + 1  # 更新哑原子标记

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        # 检查键是否已经在broken_bonds中
        if bond_idx in bond_indices_set:
            continue  # 跳过已经添加的键

        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        bond_type = bond.GetBondType()

        # 判断是否为环与侧链之间的键
        atom1_in_ring = atom1.IsInRing()
        atom2_in_ring = atom2.IsInRing()

        if atom1_in_ring != atom2_in_ring:
            # 这个键连接了环和侧链
            dummy_label = current_dummy_label
            current_dummy_label += 1
            broken_bonds.append({
                'bond_idx': bond_idx,
                'atom_idx1': atom1.GetIdx(),
                'atom_idx2': atom2.GetIdx(),
                'bond_type': bond_type,
                'dummy_label': (dummy_label, dummy_label)
            })
            bond_indices_set.add(bond_idx)  # 添加到集合中

    # 获取所有需要断开的键的索引列表和对应的哑原子标记
    bond_indices = [bond['bond_idx'] for bond in broken_bonds]
    dummy_labels = [bond['dummy_label'] for bond in broken_bonds]

    # 在指定的键上断开分子，并添加带有标记的哑原子
    mol_frag = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True, dummyLabels=dummy_labels)

    # 获取每个原子所属的片段映射
    frags = Chem.GetMolFrags(mol_frag, asMols=False, sanitizeFrags=False)
    atom_idx_to_frag_id = {}
    for frag_id, atom_indices in enumerate(frags):
        for atom_idx in atom_indices:
            atom_idx_to_frag_id[atom_idx] = frag_id

    # 随机选择一个片段丢弃
    # frag_ids = set(atom_idx_to_frag_id.values())
    # discarded_frag_id = random.choice(list(frag_ids))
    # print(f"丢弃的片段ID: {discarded_frag_id}", f"片段总数量：{len(list(frag_ids))}")

    frag_ids = set(atom_idx_to_frag_id.values())

    num_to_remove = int(len(frag_ids) * fraction_to_remove + np.random.rand())
    K = max(1, min(len(frag_ids) - 1, num_to_remove))

    if K >= len(frag_ids):
        return None, None
        discarded_frag_ids = frag_ids
    else:
        discarded_frag_ids = set(random.sample(list(frag_ids), K))

    atoms_to_discard = [atom_idx for atom_idx, frag_id in atom_idx_to_frag_id.items() if frag_id in discarded_frag_ids]

    # 获取需要丢弃的原子索引列表
    # atoms_to_discard = [atom_idx for atom_idx, frag_id in atom_idx_to_frag_id.items() if frag_id == discarded_frag_id]

    # 创建一个可编辑的分子对象
    mol_edit = Chem.RWMol(mol_frag)

    # 按降序删除需要丢弃的原子，避免索引问题
    atoms_to_discard.sort(reverse=True)
    for atom_idx in atoms_to_discard:
        mol_edit.RemoveAtom(atom_idx)

    # 更新原子索引映射，从旧索引到新索引
    old_to_new_atom_idx = {}
    new_idx = 0
    for old_idx in range(mol_frag.GetNumAtoms()):
        if old_idx not in atoms_to_discard:
            old_to_new_atom_idx[old_idx] = new_idx
            new_idx += 1

    # 构建哑原子标记到哑原子索引的映射
    dummy_label_to_dummy_atoms = {}
    for atom in mol_edit.GetAtoms():
        if atom.GetAtomicNum() == 0:  # 哑原子
            isotope = atom.GetIsotope()
            if isotope not in dummy_label_to_dummy_atoms:
                dummy_label_to_dummy_atoms[isotope] = []
            dummy_label_to_dummy_atoms[isotope].append(atom.GetIdx())

    # 创建一个可编辑的分子对象用于重新连接
    mol_reconstructed = mol_edit

    # 准备一个列表来存储需要重新连接的信息
    reconnection_ops = []

    # 收集所有需要重新连接的操作信息
    for broken_bond in broken_bonds:
        dummy_label = broken_bond['dummy_label'][0]
        bond_type = broken_bond['bond_type']

        # 检查哑原子是否都存在
        if dummy_label in dummy_label_to_dummy_atoms:
            dummy_atom_indices = dummy_label_to_dummy_atoms[dummy_label]
            if len(dummy_atom_indices) != 2:
                # 如果哑原子数量不为2，无法重新连接
                continue

            idx1, idx2 = dummy_atom_indices

            # 获取哑原子的邻居原子（实际的原子）
            atom1 = mol_reconstructed.GetAtomWithIdx(idx1)
            atom2 = mol_reconstructed.GetAtomWithIdx(idx2)

            neighbor1 = [n for n in atom1.GetNeighbors() if n.GetAtomicNum() != 0]
            neighbor2 = [n for n in atom2.GetNeighbors() if n.GetAtomicNum() != 0]

            if len(neighbor1) != 1 or len(neighbor2) != 1:
                continue

            neighbor_idx1 = neighbor1[0].GetIdx()
            neighbor_idx2 = neighbor2[0].GetIdx()

            # 收集需要删除的哑原子索引和需要添加的键信息
            reconnection_ops.append({
                'dummy_indices': (idx1, idx2),
                'neighbor_indices': (neighbor_idx1, neighbor_idx2),
                'bond_type': bond_type
            })

    # 按哑原子索引的降序排序，以避免索引问题
    reconnection_ops.sort(key=lambda x: max(x['dummy_indices']), reverse=True)

    # 执行重新连接操作
    for op in reconnection_ops:
        idx1, idx2 = op['dummy_indices']
        neighbor_idx1, neighbor_idx2 = op['neighbor_indices']
        bond_type = op['bond_type']

        # 删除哑原子，按降序删除
        for idx in sorted([idx1, idx2], reverse=True):
            mol_reconstructed.RemoveAtom(idx)

        # 添加键，重新连接分子
        mol_reconstructed.AddBond(neighbor_idx1, neighbor_idx2, bond_type)

    # 获取最终的分子并进行规范化
    mol_final = mol_reconstructed.GetMol()
    Chem.SanitizeMol(mol_final)
    Chem.SanitizeMol(mol)

    # 比较原始分子和修改后分子的SMILES
    smiles_original = Chem.MolToSmiles(mol)
    smiles_modified = Chem.MolToSmiles(mol_final)
    return smiles_original, smiles_modified

def clean_smiles_regex(smiles):
    # 使用正则表达式匹配并移除形如 [数字*] 的虚拟原子标记
    cleaned_smiles = re.sub(r'\[\d+\*\]', '', smiles)
    return cleaned_smiles

def remove_random_fragment(smiles, fraction_to_remove=0.2):
    try:
        res = bricks_drop(smiles, fraction_to_remove=fraction_to_remove)
        return res
    except:
        return None, None

def remove_random_atoms(smiles, fraction_to_remove=0.2):
    res = smiles
    origin = smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, smiles[::2]
        Chem.SanitizeMol(mol)
        origin = Chem.MolToSmiles(mol)
        num_atoms = mol.GetNumAtoms()
        num_to_remove = int(num_atoms * fraction_to_remove + np.random.rand())
        indices_to_remove = random.sample(range(num_atoms), num_to_remove)
        editable_mol = Chem.EditableMol(mol)
        for idx in sorted(indices_to_remove, reverse=True):
            editable_mol.RemoveAtom(idx)
        new_mol = editable_mol.GetMol()
        Chem.SanitizeMol(new_mol)
        res = Chem.MolToSmiles(new_mol)
    except:
        # return smiles[::2]
        res = ''.join([s for s in smiles if random.random() > fraction_to_remove])
        if len(res) < 3:
            res = smiles[::2]
        return origin, res
    return origin, res

class SMILESDropNewCombDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        BRICKS_drop_rate=0.2,
        atoms_drop_rate=0.2,
        BRICKS_sample_policy_ratio=0.5,
    ):
        self.dataset = dataset
        self.BRICKS_drop_rate = BRICKS_drop_rate
        self.atoms_drop_rate = atoms_drop_rate
        self.BRICKS_sample_policy_ratio = BRICKS_sample_policy_ratio

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        smiles = self.dataset[index]
        # token_smi = [token for token in self.regex.findall(raw_data)]
        # assert len(token_smi) < self.max_seq_len and len(token_smi) > 0
        # new_data = np.array([self.dictionary.index(x) for x in token_smi], dtype=np.int32)
        res = smiles
        if random.random() > self.BRICKS_sample_policy_ratio:
            smiles_original, smiles_modified = remove_random_atoms(smiles, fraction_to_remove=self.atoms_drop_rate)
        else:
            smiles_original, smiles_modified = remove_random_fragment(smiles, fraction_to_remove=self.BRICKS_drop_rate)
            if smiles_modified is not None:
                smiles_modified = clean_smiles_regex(smiles_modified)
            else:
                smiles_original, smiles_modified = remove_random_atoms(smiles, fraction_to_remove=self.atoms_drop_rate)
        return {"origin": smiles_original, "dropped": smiles_modified}