import os
import sys
import json
import argparse
from datetime import datetime
import logging
import importlib
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import selfies as sf
import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers import BertTokenizer 
from src.tools.tokenizer import BartTokenizer
from src.dataset.spectra_with_mol import SpectraWithSelfiesTokenDataset
from src.model.spectra2selfies import Spectra2Seflies
from src.augmentation import *
from KerasModelTorch import KerasModel


SPEC_TYPES = ['ir', 'raman', 'uv', 'xas-c', 'xas-n', 'xas-o']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:7')
    # sample args
    sample_args = parser.add_argument_group('Sample Args')
    sample_args.add_argument('--num_beams', type=int, default=100)
    sample_args.add_argument('--num_return_sequences', type=int, default=100)
    sample_args.add_argument('--top_k', type=int, default=50)
    sample_args.add_argument('--top_p', type=float, default=1)
    sample_args.add_argument('--temperature', type=float, default=1.0)
    sample_args.add_argument('--do_sample', type=bool, default=False)
    sample_args.add_argument('--no_repeat_ngram_size', type=int, default=0)
    sample_args.add_argument('--length_penalty', type=float, default=0.0)
    sample_args.add_argument('--max_length', type=int, default=100)    
    
    pred_args = parser.parse_args()
    device = pred_args.device
    
    args = OmegaConf.load(os.path.join(pred_args.log_path, 'config.yaml'))
    args.test_dataset.root_data = '/home/user/tsj/data/fg26/train_test_split_no_chiral'
    # dataset
    tokenizer = BartTokenizer.from_pretrained(os.path.join(pred_args.log_path, 'tokenizer'))
    spectrum_pipline = {}
    for spec_type in SPEC_TYPES:
        spectrum_pipline[spec_type] = build_augmentation(args.spectrum_pipline.get(f'{spec_type}_pipline'), logger=None)
    smiles_pipline = AugmentationPipeline([SmilesCanonical()])
    test_dataset = SpectraWithSelfiesTokenDataset(**args.test_dataset, 
                                             tokenizer=tokenizer, 
                                             transform={**spectrum_pipline, 'smiles': smiles_pipline})
    test_loader = DataLoader(test_dataset, batch_size=pred_args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    # model 
    model = Spectra2Seflies(**args.model, tokenizer=tokenizer)
    logger = logging.getLogger()
    train_model = KerasModel(model, logger=logger)
    
    # sample
    inference_config = {arg.dest: getattr(pred_args, arg.dest) for arg in sample_args._group_actions}
    
    inference_results = train_model.run_inference(
                    test_loader,
                    ckpt_path=os.path.join(pred_args.log_path, 'best.pt'),
                    inference_config=inference_config,
                    device=device,
                    save_dir=os.path.join(pred_args.log_path, f'top100_inference_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
                )