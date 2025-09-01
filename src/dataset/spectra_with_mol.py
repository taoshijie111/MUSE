import os
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fs.fileIO import load_fsz_thread
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


class SpectraWithSelfiesTokenDataset(Dataset):
    """
    加载多光谱的数据集，允许动态的增加光谱类型，并返回光谱数据和selfies的token id

    Args:
        root_data (str): 数据集根目录
        tokenizer (APETokenizer): selfies的tokenizer
        spectra_type (list): spectra的类型，如['ir', 'raman', 'uv', 'xas-c', 'nmr-c']。当为'xas'时默认加载'xas-c', 'xas-n', 'xas-o'；当为'nmr'时默认加载'nmr-c', 'nmr-h'
        specific_spec_type (list): 需要特殊处理的光谱类型（使用数值tokenzier),默认为['nmr-c', 'nmr-h']
        transform : 光谱变换
        interp_num: 光谱的最大长度，若不够则进行插值。
        molecule_weight： True: 使用分子量，否则为False
    """
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
            **kwargs
    ):
        super().__init__()
        self.root_data = Path(root_data)
        self.transform = transform
        self.specific_spec_type = specific_spec_type
        self.molecule_weight = molecule_weight
        self.interp_num = interp_num
        self.tokenizer = tokenizer
        
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
        primary_file = self.root_data / name.format(primary_spec)
        primary_data = rebuild_dict(load_fsz_thread(str(primary_file)))
        self.smiles_list = list(primary_data.keys())

        # Load each spectrum file into a dict without merging to save memory.
        self.datasets = {primary_spec: primary_data}
        for spec in spec_list[1:]:
            spec_file = self.root_data / name.format(spec)
            try:
                self.datasets[spec] = rebuild_dict(load_fsz_thread(str(spec_file)))
            except Exception as e:
                print(f"Failed to load {spec_file}: {e}")
                self.datasets[spec] = {}  # File not found; later handled by default value

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

    def reset_smiles_list(self, file_path):
        with open(file_path) as f:
            self.smiles_list = [smi.strip() for smi in f.readlines()]

    def interp(self, x):
        if len(x) == self.interp_num:
            return x
        return np.interp(np.arange(self.interp_num), np.arange(len(x)), x)
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        spec_data_list = []
        for spec in self.spec_list:
            spec_value = self.datasets.get(spec, {}).get(smiles, np.zeros((self.interp_num,)))
            if spec_value is None or spec_value.size == 0:
                spec_value = np.zeros((self.interp_num,))
            spec_value = self.interp(spec_value)
            if self.transform.get(spec, None):
                spec_value = self.transform[spec].apply(spec_value)
            spec_data_list.append(spec_value)
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


if __name__ == '__main__':
    import logging
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from src.tools.tokenizer import get_tokenizer
    from src.augmentation import AugmentationPipeline, NormalizeByArea, SmilesCanonical, build_augmentation
    import matplotlib.pyplot as plt
    logger = logging.getLogger(__name__)
    tokenizer = get_tokenizer('./tokenizer/bart-base', '/home/user/tsj/Spectra2Molecule/tokenizer/fg26_tokenzier')
    args = OmegaConf.load('/home/user/tsj/Spectra2Molecule/configs/fg26_spectra2selfies.yaml')
    base_spectrum_pipline = build_augmentation(args.spectrum_pipline.base_spectrum_pipline, logger)
    
    spectrum_pipline = {}
    SPEC_TYPES = ['ir', 'raman', 'uv', 'xas-c', 'xas-n', 'xas-o']
    for spec_type in SPEC_TYPES:
        spectrum_pipline[spec_type] = build_augmentation(args.spectrum_pipline.get(f'{spec_type}_pipline'), logger) + base_spectrum_pipline
    smiles_pipline = AugmentationPipeline([SmilesCanonical(False)])
    
    dataset = SpectraWithSelfiesTokenDataset(root_data='/home/user/tsj/data/fg26/train_test_split/example',
                                              name='example_train_fg26_{}_spectra.npz', 
    tokenizer=tokenizer, transform={**spectrum_pipline, 'smiles': smiles_pipline}, molecule_weight=False, spec_type=['ir', 'nmr'],)

    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=dataset.collate_fn)
    
    for smiles, selfies, encode, spectra, specific_spec in tqdm(dataloader):
        print(specific_spec['input_ids'])
        break
        # print(smiles)
        # print(encode['input_ids'].shape)
        # print(spectra.shape)
        # plt.plot(spectra[0, 3, 0, :])
        # plt.savefig('test.png')
        # break
        # print(spectra.shape)
        # print(specific_spec)
        # print(encode['input_ids'])
        # print(torch.nonzero(encode['input_ids'] == tokenizer.special_tokens['<sep>']))
        # print(encode['labels'])
        # print(encode.keys())
        # print(spectra.shape)