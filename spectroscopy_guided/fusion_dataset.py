import os
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset
from pathlib import Path

import sys
sys.path.append('/home/user/tsj/Spectra2Molecule')

from src.dataset.spectra_with_mol import *
from src.tools.mol import get_selfies



def rebuild_dict(flattened, sep='.'):
    result = {}
    for k, v in flattened.items():
        keys = k.split(sep)
        d = result
        for key in keys[:-1]:  # 遍历到倒数第二个key，构造层级结构
            if key not in d:
                d[key] = {}  # 如果不存在，则初始化为空字典
            d = d[key]  # 深入下一层
        d[keys[-1]] = v  # 设置最终的值
    return result


def load_nmr_tokenizer():
    # https://huggingface.co/spaces/huggingface/number-tokenization-blog
    from transformers import AutoTokenizer
    from tokenizers import pre_tokenizers, Regex

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    # Added step: split by R2L digits
                    pre_tokenizers.Split(pattern = Regex(r"\d{1,3}(?=(\d{3})*\b)"), behavior="isolated", invert = False),
                    # Below: Existing steps from Llama 3's tokenizer
                    pre_tokenizers.Split(pattern=Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"), 
                    behavior="isolated", invert=False), 
                    pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
                ]
            )
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    return tokenizer


def string_specific_spec(nmr_data: np.array, spec_type='nmr-c', split_token='|', decimal_places=2, intensity=False):
    # nmr_data is a 2D array 
    if nmr_data.size == 0:
        return '[MASK]'
    if np.all(nmr_data == 0.):    # 可能存在np.array([0.0])
        return '[MASK]'
    nmr_data = nmr_data.reshape(-1, nmr_data.shape[-1])
    if spec_type == 'nmr-c' or spec_type == 'nmr-h':
        # Fine
        # peak = [f"{row[0]:.{decimal_places}f} {int(row[1])}" for row in nmr_data ]
        
        # 去重
        # peak = list(set([row[0] for row in nmr_data]))
        # peak = [f"{row:.{decimal_places}f}" for row in peak ]
        
        # best result
        # peak = [f"{row[0]:.{decimal_places}f}" for row in nmr_data ]
        
        # Plain
        peak = [f"{row[0]:.{decimal_places}f}" for row in nmr_data for _ in range(int(row[1]))]
    elif spec_type == 'ms':
        peak = [f"{row[0]:.{decimal_places}f} {row[1]:.{decimal_places}f}" for row in nmr_data]
    else:
        raise ValueError(f"Unsupported spec_type: {spec_type}")
    return (split_token.join(peak) + split_token).strip()


def get_molecular_weight(smiles) -> float:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        molecular_weight = Descriptors.MolWt(molecule)
        return molecular_weight
    else:
        raise '[MASK]'


class FusionSpectraDataset(Dataset):
    
    def __init__(
            self,
            root_data, 
            name,
            tokenizer, 
            spec_type=['ir', 'uv', 'xas-c', 'nmr'],
            specific_spec_type=['nmr-c', 'nmr-h', 'ms'],
            transform={}, 
            molecule_weight=False, 
            interp_num=4000, 
            mean=False,
            **kwargs
    ):
        super().__init__()
        self.root_data = Path(root_data)
        self.transform = transform
        self.specific_spec_type = specific_spec_type
        self.molecule_weight = molecule_weight
        self.interp_num = interp_num
        self.tokenizer = tokenizer
        self.mean = mean
        
        # Expand composite types: 'nmr' and 'xas'
        spec_list = []
        self.specific_spec_list = []
        for spec in spec_type:
            if spec == 'nmr':
                spec_list.extend(['nmr-c', 'nmr-h'])
            elif spec == 'xas':
                spec_list.extend(['xas-c', 'xas-n', 'xas-o'])
            else:
                spec_list.append(spec)

        # Use the first spectrum type to define the set of smiles
        primary_spec = spec_list[0]
        # Determine if we need to load both train and test data
        if name == "fg26_{}_spectra.npz":
            train_file = self.root_data / f"train_{name}".format(primary_spec)
            test_file = self.root_data / f"test_{name}".format(primary_spec)
            
            primary_data = {}
            try:
                train_data = rebuild_dict(load_fsz_thread(str(train_file)))
                primary_data.update(train_data)
            except Exception as e:
                print(f"Failed to load {train_file}: {e}")
            try:
                test_data = rebuild_dict(load_fsz_thread(str(test_file)))
                primary_data.update(test_data)
            except Exception as e:
                print(f"Failed to load {test_file}: {e}")
        else:
            # Continue with the existing pattern for other cases
            primary_file = self.root_data / name.format(primary_spec)
            primary_data = rebuild_dict(load_fsz_thread(str(primary_file)))

        self.smiles_list = list(primary_data.keys())
        print('Sample number of Dataset: ', len(self.smiles_list))
        self.datasets = {primary_spec: primary_data}

        # Load all other spectrum types
        for spec in spec_list[1:]:
            if name == "fg26_{}_spectra.npz":
                train_file = self.root_data / f"train_{name}".format(spec)
                test_file = self.root_data / f"test_{name}".format(spec)
                
                combined_data = {}
                try:
                    train_data = rebuild_dict(load_fsz_thread(str(train_file)))
                    combined_data.update(train_data)
                except Exception as e:
                    print(f"Failed to load {train_file}: {e}")
                
                try:
                    test_data = rebuild_dict(load_fsz_thread(str(test_file)))
                    combined_data.update(test_data)
                except Exception as e:
                    print(f"Failed to load {test_file}: {e}")
                    
                self.datasets[spec] = combined_data
            else:
                spec_file = self.root_data / name.format(spec)
                try:
                    self.datasets[spec] = rebuild_dict(load_fsz_thread(str(spec_file)))
                except Exception as e:
                    print(f"Failed to load {spec_file}: {e}")
                    self.datasets[spec] = {} 

        # remove specify spec
        self.spec_list = []
        for spec in spec_list:
            if spec in self.specific_spec_type:
                self.specific_spec_list.append(spec)
            else:
                self.spec_list.append(spec)
        
        if len(self.specific_spec_list) > 0 or molecule_weight:
            # initialize nmr tokenizer as before
            self.specific_spec_tokenizer = load_nmr_tokenizer()
            self.specific_spec_tokenizer.padding_side = 'right'

    def _reinit(self, file_path, separator, resume_path=None):
        self.separator = separator
        self.smiles_list = []
        with open(file_path) as f:
            for line in f.readlines():
                if '#' == line[0]:
                    continue
                else:
                    self.smiles_list.append(line.strip().split('\t')[0])
            # self.smiles_list = [smi.strip() for smi in f.readlines()]
        if resume_path is not None:
            import pandas as pd
            import os
            ready_generate_smiles = pd.read_csv(os.path.join(resume_path, 'sampling.csv')).true_smiles.to_list()
            self.smiles_list = list(set(self.smiles_list) - set(ready_generate_smiles))

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        sm_list = smiles.split(self.separator)

        spec_data_list = []
        for spec in self.spec_list:
            combined_spec = np.zeros((self.interp_num,))
            
            # Retrieve and process spectrum for each SMILES
            for smi in sm_list:
                spec_value = self.datasets.get(spec, {}).get(smi, np.zeros((self.interp_num,)))
                if spec_value is None or spec_value.size == 0:
                    spec_value = np.zeros((self.interp_num,))
                spec_value = self.interp(spec_value)
            
                combined_spec += spec_value
            if self.mean:
                combined_spec /= 2       
            if self.transform.get(spec, None):
                combined_spec = self.transform[spec].apply(combined_spec)
            spec_data_list.append(combined_spec)
        
        spectrum = np.array(spec_data_list)

        if len(self.specific_spec_list) > 0:
            specific_spec = ''
            for spec_type in self.specific_spec_list:
                spec = self.datasets.get(spec_type, {}).get(smiles, np.zeros([]))
                if spec is None:
                    spec = np.zeros()
                specific_spec += f'{spec_type} ' + string_specific_spec(spec, spec_type) + ' '
        else:
            specific_spec = None
        
        if self.molecule_weight:
            mw = get_molecular_weight(smiles)
            if specific_spec:
                specific_spec += ' ' + 'Molecular Weight ' + str(int(mw))
            else:
                specific_spec = 'Molecular Weight ' + str(int(mw))

        if self.transform.get('smiles', None):
            random_smiles = self.transform['smiles'].apply(smiles)
        selfies = get_selfies(random_smiles)
        return smiles, selfies, torch.tensor(spectrum, dtype=torch.float32).unsqueeze(1), specific_spec

    def collate_fn(self, batch):
        smiles, selfies, spectra, specific_spec = zip(*batch)
        spectra = torch.stack(spectra, dim=0)   # [b, n, 1, 4000]
        
        encode = self.tokenizer(
                selfies,
                padding=True,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
        encode['labels'] = encode["input_ids"].contiguous()
        encode["labels"] = torch.where(encode['labels'] != self.tokenizer.pad_token_id, encode['labels'], -100)
        encode['labels'] = encode['labels'][:, 1:]  # remove <bos> token
        if not specific_spec[0]:
            specific_spec = None
        else:
            specific_spec = self.specific_spec_tokenizer(specific_spec, add_special_tokens=False, return_tensors='pt', padding=True)
        return smiles, selfies, encode, spectra, specific_spec

    def interp(self, x):
        if len(x) == self.interp_num:
            return x
        return np.interp(np.arange(self.interp_num), np.arange(len(x)), x)
    
    def __len__(self):
        return len(self.smiles_list)