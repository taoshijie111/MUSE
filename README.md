# Multimodal Spectroscopy-Guided AI Generation of Molecules with Tailored Properties

This repository contains the implementation of the paper "Multimodal Spectroscopy-Guided AI Generation of Molecules with Tailored Properties". The work presents a novel approach for generating molecules with specific properties by leveraging multiple spectroscopic techniques as guidance.

## Overview

The framework combines multi-modal spectroscopic data (IR, Raman, UV, XAS, NMR) with transformer-based architectures to generate molecular structures with desired properties. The model uses SELFIES (Self-Referencing Embedded Strings) representation for guaranteed chemical validity and employs a BART-based sequence generation architecture.

## Architecture

### Core Components
- **Multi-Spectral Encoder**: Processes multiple spectroscopic techniques using Vision Transformer (ViT), CNN, or MLP architectures
- **Post-Embedding Model**: BERT/RoBERTa encoder for contextual spectral representation
- **BART Decoder**: Generates SELFIES molecular representations autoregressively
- **Spectral Type Embeddings**: Distinguishes between different spectroscopic modalities

### Model Specifications
- **Input**: Multi-dimensional spectral arrays
- **Output**: SELFIES sequences decoded to SMILES molecular representations
- **Supported Spectral Types**: IR, Raman, UV, XAS-C/N/O, NMR variants

## Installation

### Environment Setup
Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate Spec2Mol
```

### Dataset and Model Weights
Download the required datasets and place them in the `dataset/` folder. Pre-trained model weights should be placed in the `logs/` folder. The repository includes two pre-trained models:
- `FG26-SR-1-IR10-NMR50-ViT-BERT_Spectra2Selfies_2025-03-27_17-14/` (IR + NMR)
- `FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11/` (Raman + XAS)

## Usage

### 1. Model Training

Train a model using the configuration file:

```bash
python Spectra2SelfiesTrainerTorch.py --config configs/fg26_spectra2selfies.yaml --device cuda:0
```

**Configuration Options:**
- Modify `spec_type` in the config file to train with different spectral combinations
- Example for multi-spectral training:
```yaml
train_dataset:
  spec_type: ['ir', 'raman', 'xas-c']
model:
  spec_type: ['ir', 'raman', 'xas-c']
```

### 2. Model Inference

Perform inference on test datasets:

```bash
python Spectra2Seflies_Prediciton.py \
  --log_path logs/FG26-SR-1-IR10-NMR50-ViT-BERT_Spectra2Selfies_2025-03-27_17-14 \
  --device cuda:0 \
  --batch_size 50
```

**Generation Parameters:**
- `--num_beams`: Number of beam search paths (default: 100)
- `--num_return_sequences`: Number of sequences to return (default: 100)
- `--top_k`: Top-k sampling parameter (default: 50)
- `--temperature`: Sampling temperature (default: 1.0)
- `--max_length`: Maximum sequence length (default: 100)

### 3. Spectroscopy-Guided Molecule Generation

Generate molecules with specific properties using spectral fusion:

```bash
python spectroscopy_guided/spectral_fusion.py \
  --log_path logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11 \
  --smiles_pair_path properties/extreme_large_gap/round1 \
  --pair_fg_name extreme_gap_fusion \
  --device cuda:0 \
  --batch_size 100 \
  --spectra_dir properties/extreme_large_gap/round1
```

**Input Format:**
- SMILES pair file: Each line contains two SMILES separated by '.' (e.g., `molecule1.molecule2`)
- Lines starting with '#' are treated as comments
- Spectral data should be provided as `.npz` files in the `spectra_dir`

**Output:**
- Results saved as `sampling.csv` in the generated subfolder under `log_path`
- Contains generated SELFIES sequences and corresponding SMILES

## Molecule Generation

### 1. Property-Specific Generation

The repository includes pre-computed results for various molecular properties:

#### Electronic Properties
- **HOMO-LUMO Gap**: 
  ```bash
  # Large gap molecules
  python spectroscopy_guided/spectral_fusion.py --log_path logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11 --smiles_pair_path properties/extreme_large_gap/round1 --pair_fg_name extreme_gap_fusion --device cuda:0 --spectra_dir properties/extreme_large_gap/round1
  
  # Small gap molecules
  python spectroscopy_guided/spectral_fusion.py --log_path logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11 --smiles_pair_path properties/extreme_small_gap/round1 --pair_fg_name extreme_small_gap_fusion --device cuda:0 --spectra_dir properties/extreme_small_gap/round1
  ```

#### Drug-like Properties
- **QED and SA Score**:
  ```bash
  python spectroscopy_guided/spectral_fusion.py --log_path logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11 --smiles_pair_path properties/qed-sa --pair_fg_name qed_sa_fusion --device cuda:0 --spectra_dir properties/qed-sa
  ```

#### Lipophilicity
- **LogP Tuning**:
  ```bash
  # LogP = -2
  python spectroscopy_guided/spectral_fusion.py --log_path logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11 --smiles_pair_path properties/tuning_logp/logp=-2 --pair_fg_name logp_fusion --device cuda:0 --spectra_dir properties/tuning_logp/logp=-2
  
  # LogP = 3 or 5
  python spectroscopy_guided/spectral_fusion.py --log_path logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11 --smiles_pair_path properties/tuning_logp/logp=3 --pair_fg_name logp_fusion --device cuda:0 --spectra_dir properties/tuning_logp/logp=3
  ```

### 2. Structure-Guided Generation

Generate molecules with specific functional group combinations:

```bash
# Example: Alcohol + Ketone combination
python spectroscopy_guided/spectral_fusion.py --log_path logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11 --smiles_pair_path structures/extend_fg26 --pair_fg_name alcohol-ketone_generation --device cuda:0 --spectra_dir structures/extend_fg26/alcohol-ketone_generation_*
```

### 3. Evaluation Metrics

The model uses chemical structure-aware evaluation:
- **Top-K Accuracy**: InChI key-based molecular structure comparison
- **Plain Text Matching**: Exact SMILES string comparison  
- **Molecular Validity**: RDKit-based structure validation
- **Property Prediction**: DFT validation for generated molecules

## Repository Structure

```
├── configs/              # Configuration files
├── dataset/             # Training and test datasets
├── environment.yaml     # Conda environment specification
├── fs/                  # Feature selection utilities
├── logs/                # Training logs and model checkpoints
├── properties/          # Property-specific generation results
├── spectroscopy_guided/ # Molecule generation pipeline
├── src/                 # Core model and data processing code
│   ├── augmentation.py
│   ├── dataset/
│   ├── kerascallbacks/
│   ├── model/
│   ├── scheduler.py
│   └── tools/
├── structures/          # Structure-specific generation results
├── tokenizer/           # SELFIES tokenizer
├── KerasModelTorch.py   # Training framework
├── Spectra2SelfiesTrainerTorch.py  # Training script
└── Spectra2Seflies_Prediciton.py   # Inference script
```

## Key Features

- **Multi-Modal Learning**: Integrates multiple spectroscopic techniques
- **Chemical Validity**: SELFIES representation ensures valid molecular structures
- **Property Control**: Guided generation for specific molecular properties
- **Flexible Architecture**: Supports different spectral encoders (ViT/CNN/MLP)
- **Robust Evaluation**: Structure-based metrics using InChI keys
- **Scalable Training**: Mixed precision and gradient accumulation support

## Requirements

- Python 3.9+
- PyTorch 2.6.0+ with CUDA support
- Transformers 4.51.3+
- RDKit 2024.3.6+
- SELFIES 2.2.0+
- See `environment.yaml` for complete dependency list

## Citation

If you use this code, please cite:
```
Multimodal Spectroscopy-Guided AI Generation of Molecules with Tailored Properties
[Add citation details when available]
```

## License

[Add license information]