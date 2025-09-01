import numpy as np
import importlib
import re
import random
import torch
from typing import Tuple
from rdkit import Chem
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter



def try_import(module, cls):
    module = importlib.import_module(module)
    return module


def build_augmentation(config, logger=None):
    augmentations = []

    if config:
        for key, value in config.items():
            augmentation_module = try_import(value.module, value.cls)
            module = value.get('module')
            cls = value.get('cls')
            if augmentation_module is not None:
                augmentations.append(getattr(augmentation_module, cls)(**value.get('args', {})))
                if logger:
                    logger.info(f'===>Building {key} from {module}/{cls} successfully')
                else:
                    print(f'===>Building {key} from {module}/{cls} successfully')
            else:
                if logger:
                    logger.info(f'===>Not Building {key} from {module}/{cls} successfully')
                else:
                    print(f'===>Not Building {key} from {module}/{cls} successfully')

                raise ImportError
    return AugmentationPipeline(augmentations)


class AugmentationBase:
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    def apply_augmentation(self, spectrum: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class AugmentationPipeline:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def apply(self, image):
        for augmentation in self.augmentations:
            image = augmentation.apply_augmentation(image)
        return image

    def __repr__(self) -> str:
        all_augmentations = ''
        for augmentation in self.augmentations:
            all_augmentations += augmentation.name + '\t'
        return f"AugmentationPipeline: {all_augmentations}"

    def __add__(self, other):
        if not isinstance(other, AugmentationPipeline):
            raise TypeError("Can only add AugmentationPipeline instances")
        combined_augmentations = self.augmentations + other.augmentations
        return AugmentationPipeline(combined_augmentations)


class SmilesRandom(AugmentationBase):
    def __init__(self, prop=1.0, random_type='unrestricted'):
        super().__init__(name='SmilesRandom')
        self.prop = prop
        self.random_type = random_type

    def apply_augmentation(self, smiles):
        if self.prop >= 1.0 or random.random() < self.prop:
            mol = Chem.MolFromSmiles(smiles)
            if self.random_type == 'unrestricted':
                randomized_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            elif self.random_type == 'restricted': 
                # new_atom_order = list(range(mol.GetNumHeavyAtoms()))
                # random.shuffle(new_atom_order)
                # random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
                # randomized_smiles = Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
                mol.SetProp("_canonicalRankingNumbers", "True")
                idxs = list(range(0, mol.GetNumAtoms()))
                random.shuffle(idxs)
                for i, v in enumerate(idxs):
                    mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
                randomized_smiles = Chem.MolToSmiles(mol)
            else:
                raise ValueError(f'Type {self.random_type} is not valid.')
            return randomized_smiles
        else:
            return smiles


class SpectrumMax(AugmentationBase):
    def __init__(self, intensity_max=None):
        super().__init__(name='normalization')
        self.intensity_max = intensity_max

    def method(self, spectrum: np.ndarray) -> np.ndarray:
        max_value = max(spectrum)
        if max_value == 0:
            return spectrum
        if self.intensity_max:
            spectrum_norm = spectrum / self.intensity_max
        else:
            spectrum_norm = spectrum / max(spectrum) 
        return spectrum_norm

    def apply_augmentation(self, spectrum: np.ndarray) -> np.ndarray:
        normalized_spectrum = np.apply_along_axis(self.method, 0, spectrum)
        return normalized_spectrum


class SpectrumShift(AugmentationBase):
    def __init__(self, shift):
        super().__init__(name='SpectrumShift')
        self.shift = shift
    
    def apply_augmentation(self, spectrum):
        shift = np.random.randint(-self.shift, self.shift)
        spectrum = ndi.shift(spectrum, shift, mode='constant')
        return spectrum
    
    
class SpectrumScale(AugmentationBase):
    def __init__(self, scale=0.001):
        super().__init__(name='SpectrumScale')
        self.scale = scale
        
    
    def apply_augmentation(self, spectrum):
        scale = np.clip(np.random.randn(), -3, 3) * self.scale
        spectrum = spectrum * (1 + scale)
        return spectrum
    
        
class SmilesCanonical(AugmentationBase):
    def __init__(self, isomericSmiles=True):
        super().__init__(name='smiles_canonical')
        self.isomericSmiles = isomericSmiles

    def apply_augmentation(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=self.isomericSmiles)
        return smiles
        

class RemoveSmilesChirality(AugmentationBase):
    def __init__(self, ):
        super().__init__(name='RemoveSmilesChirality')
        pass
    def apply_augmentation(self, smiles: str) -> str:
        # Handle invalid input
        if not smiles or not isinstance(smiles, str):
            return None
        
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # Invalid SMILES

        # Find all tetrahedral chiral centers
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        # Remove chirality at each stereocenter, but keep double bond stereochemistry
        for atom_idx, _ in chiral_centers:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

        # Update property cache and re-perceive stereochemistry
            # This ensures double bond stereochemistry is preserved
        mol.UpdatePropertyCache()
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Generate canonical SMILES with E/Z information but without chiral centers
        return Chem.MolToSmiles(mol, isomericSmiles=True)
        

class Smiles2Selfies(AugmentationBase):
    def __init__(self):
        super().__init__(name='smiles_2_selfies')
        pass

    def apply_augmentation(self, smiles: str) -> str:
        return get_selfies(smiles)


class SelfiesRandomMask(AugmentationBase):
    def __init__(self, rate=0.5):
        super().__init__(name='selfies_random_mask')
        self.rate = rate

    def apply_augmentation(self, selfies: str) -> str:
        # 找到所有的 [] 块
        blocks = re.findall(r'\[.*?\]', selfies)

        # 计算需要替换的块数
        num_blocks = len(blocks)
        num_to_mask = int(num_blocks * self.rate)

        # 随机选择要替换的块
        blocks_to_mask = random.sample(blocks, num_to_mask)

        # 按比例替换块
        for block in blocks_to_mask:
            selfies = selfies.replace(block, '[<mask>]', 1)

        return selfies


class SpectrumRandomMask(AugmentationBase):
    def __init__(self, rate=0.5):
        super().__init__(name='spectrum_random_mask')
        self.rate = rate

    def apply_augmentation(self, spectrum):
        num_elements = spectrum.shape[-1]
        num_to_replace = int(num_elements * self.rate)

        indices = np.random.choice(num_elements, num_to_replace, replace=False)
        spectrum[..., indices] = -1
        return spectrum


class SpectrumRandomShift(AugmentationBase):
    def __init__(self, rate=0.01):
        super().__init__(name='spectrum_random_shift')
        self.rate = rate

    def apply_augmentation(self, spectrum):
        num_elements = spectrum.shape[-1]
        max_shift = int(num_elements * self.rate)
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        # 进行平移
        if shift > 0:  # 向右平移
            shifted_spectrum = np.concatenate((spectrum[shift:], np.zeros(shift)))
        elif shift < 0:  # 向左平移
            shifted_spectrum = np.concatenate((np.zeros(-shift), spectrum[:shift]))
        else:  # 不平移
            shifted_spectrum = spectrum
        
        return shifted_spectrum


class SpectrumRandomNoise(AugmentationBase):
    def __init__(self, start_index=0, end_index=4000, mean=0, std=0.05):
        super().__init__(name='spectrum_random_noise')
        self.start_index = start_index
        self.end_index = end_index
        self.mean = mean
        self.std = std

    def apply_augmentation(self, spectrum):
        # 生成指定区域的高斯噪声
        noise = np.random.normal(self.mean, self.std, size=self.end_index - self.start_index)
        
        # 将噪声加到指定区域
        noisy_spectrum = spectrum.copy()
        noisy_spectrum[self.start_index:self.end_index] += noise
        noisy_spectrum = np.clip(noisy_spectrum, 0, None)
        return noisy_spectrum


class SpectrumRandomSmooth(AugmentationBase):
    def __init__(self, min_sigma=0.1, max_sigma=5):
        super().__init__(name='spectrum_random_smooth')
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def apply_augmentation(self, spectrum: np.ndarray) -> np.ndarray:
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        return gaussian_filter(spectrum, sigma=sigma)
    

class NormalizeByArea(AugmentationBase):
    def __init__(self, delta_lambda=1, scale=1000, mean_area=None):
        super().__init__(name='normalize_by_area')
        self.delta_lambda = delta_lambda
        self.scale = scale
        self.mean_area = mean_area

    def apply_augmentation(self, spectrum):
        if self.mean_area is not None:
            spectrum = spectrum / self.mean_area * self.scale
        else:
            area = self.calculate_area(spectrum, self.delta_lambda)
            # 处理 max(spectrum) == 0 的情况
            if area == 0:
                return np.zeros_like(spectrum, dtype=int)
            spectrum = spectrum / area * self.scale
        return spectrum

    def calculate_area(self, spectrum, delta_lambda):
        # 使用梯形法则计算面积
        area = np.trapz(spectrum, dx=delta_lambda)
        return area


class DynamicSmilesRandom(AugmentationBase):
    """
    SMILES randomization with a probability that changes throughout training.
    
    Args:
        random_type (str): Type of randomization ('unrestricted' or 'restricted')
        start_epoch (int): Epoch to start increasing randomization probability from
        end_epoch (int): Epoch to reach maximum randomization probability
        schedule_type (str): Type of probability schedule ('linear', 'step', 'cosine')
        steps (list, optional): List of step points for step schedule
        prop_values (list, optional): List of probability values for step schedule
        current_epoch (int): Current training epoch
    """
    def __init__(
        self, 
        random_type='unrestricted',
        start_epoch=0, 
        end_epoch=100,
        schedule_type='linear',
        steps=None,
        prop_values=None,
        current_epoch=0
    ):
        super().__init__(name='DynamicSmilesRandom')
        self.random_type = random_type
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.schedule_type = schedule_type
        self.steps = steps if steps is not None else []
        self.prop_values = prop_values if prop_values is not None else []
        self.current_epoch = current_epoch
        self.current_prob = self._calculate_probability()
        
    def _calculate_probability(self):
        """Calculate the current randomization probability based on the epoch and schedule."""
        if self.current_epoch < self.start_epoch:
            return 0.0
        elif self.current_epoch >= self.end_epoch:
            return 1.0
        
        if self.schedule_type == 'linear':
            # Linear increase from 0 to 1
            progress = (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return min(1.0, max(0.0, progress))
        
        elif self.schedule_type == 'step':
            if not self.steps or not self.prop_values:
                # Default to linear if steps not provided
                progress = (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
                return min(1.0, max(0.0, progress))
            
            # Find the appropriate step
            for i, step in enumerate(self.steps):
                if self.current_epoch < step:
                    return self.prop_values[i]
            return self.prop_values[-1]
        
        elif self.schedule_type == 'cosine':
            # Cosine schedule (smoother transition)
            progress = (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1 - np.cos(np.pi * progress))
        
        else:
            # Default to linear
            progress = (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return min(1.0, max(0.0, progress))
    
    def update_epoch(self, epoch):
        """Update the current epoch and recalculate probability."""
        self.current_epoch = epoch
        self.current_prob = self._calculate_probability()
        print(f"Updated SMILES randomization probability to {self.current_prob:.4f} at epoch {epoch}")
        return self.current_prob
    
    def apply_augmentation(self, smiles):
        """Apply SMILES randomization with the current probability."""
        if random.random() < self.current_prob and self.current_prob >= 1:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
                
            if self.random_type == 'unrestricted':
                randomized_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
            elif self.random_type == 'restricted': 
                mol.SetProp("_canonicalRankingNumbers", "True")
                idxs = list(range(0, mol.GetNumAtoms()))
                random.shuffle(idxs)
                for i, v in enumerate(idxs):
                    mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
                randomized_smiles = Chem.MolToSmiles(mol)
            else:
                # Default to unrestricted if invalid type
                randomized_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
                
            return randomized_smiles
        else:
            return smiles


if __name__ == "__main__":
    import sys
    sys.path.append('/home/user/tsj/Spectra2Molecule')
    from fs.fileIO import *
    import matplotlib.pyplot as plt

    data = load_fsz_thread('/home/user/tsj/data/qm9/qm9_ir_spectra.npz')
    smiles = list(data.keys())
    # 定义要使用的数据增强方法
    augmentations = [
        NormalizeByArea(scale=10000),
    ]

    # 创建数据增强 pipeline
    spec = data[smiles[500]]
    pipeline = AugmentationPipeline(augmentations)
    print(pipeline)
    plt.plot(spec, label='Original Spectrum')
    plt.show()
    # print(data.spectra.shape)
    # 应用数据增强
    spec = pipeline.apply(spec)

    plt.plot(spec, 'r', label='augmented')
    plt.legend()
    plt.savefig('test.png')
