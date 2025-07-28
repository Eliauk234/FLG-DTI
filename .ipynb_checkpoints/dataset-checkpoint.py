import random
from rdkit import Chem
import numpy as np
import torch
from torch.utils.data import Dataset

class NewDataset(Dataset):
    def __init__(self, smiles, proteins, labels, words2idx_d, words2idx_p, max_d, max_p):
        self.smiles = smiles
        self.proteins = proteins
        self.labels = labels
        self.words2idx_d = words2idx_d
        self.words2idx_p = words2idx_p
        self.max_d = max_d
        self.max_p = max_p

    def __len__(self):
        return len(self.labels)

    def protein2emb_encoder(self, x):
        max_p = self.max_p
        try:
            i1 = np.asarray([self.words2idx_p[i] for i in x])
        except:
            i1 = np.array([0])
        l = len(i1)
        if l < max_p:
            i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        else:
            i = i1[:max_p]
        return torch.from_numpy(i)

    def drug2emb_encoder(self, x):
        max_d = self.max_d
        try:
            i1 = np.asarray([self.words2idx_d[i] for i in x])
        except:
            i1 = np.array([0])
        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        else:
            i = i1[:max_d]
        return torch.from_numpy(i)

    def __getitem__(self, index):
        s = self.smiles[index]
        p = self.proteins[index]
        d_v = self.drug2emb_encoder(s)
        p_v = self.protein2emb_encoder(p)
        y = self.labels[index]
        y = int(y)
        return d_v, p_v, y

class AugmentedDataset(NewDataset):
    def __init__(self, smiles, proteins, labels, words2idx_d, words2idx_p, max_drug_seq, max_protein_seq, p_aug=0.3):
        super().__init__(smiles, proteins, labels, words2idx_d, words2idx_p, max_drug_seq, max_protein_seq)
        self.p_aug = p_aug

    def _smiles_augment(self, smiles):
        if isinstance(smiles, list):
            smiles = ''.join(smiles)  # 如果 smiles 是列表，将其转换为字符串
        if random.random() < self.p_aug:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, doRandom=True)
        return smiles

    def _aa_substitution(self, aa):
        substitution_map = {
            'A': ['G', 'V'], 'D': ['E', 'N'], 'R': ['K', 'H'],
            'S': ['T', 'N'], 'L': ['I', 'V'], 'Y': ['F', 'W']
        }
        return random.choice(substitution_map.get(aa, [aa]))

    def __getitem__(self, idx):
        smiles = self._smiles_augment(self.smiles[idx])
        protein = self.proteins[idx]
        if random.random() < self.p_aug:
            protein = ''.join([self._aa_substitution(c) for c in protein])
        label = self.labels[idx]

        label = int(label)

        smile_idx = [self.words2idx_d.get(s, 0) for s in smiles]
        protein_idx = [self.words2idx_p.get(p, 0) for p in protein]

        smile_idx = smile_idx[:self.max_d] + [0] * (self.max_d - len(smile_idx))
        protein_idx = protein_idx[:self.max_p] + [0] * (self.max_p - len(protein_idx))

        return torch.tensor(smile_idx, dtype=torch.long), torch.tensor(protein_idx, dtype=torch.long), torch.tensor(label, dtype=torch.long)