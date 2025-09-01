# Training and inference script for Spectra2Selfies model
import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from src.kerascallbacks.callbacks import EpochCheckpoint

from KerasModelTorch import KerasModel
from src.tools.tokenizer import get_tokenizer
from src.tools.logger import Logger
from src.tools.path import get_project_root_dir
from src.dataset.spectra_with_mol import SpectraWithSelfiesTokenDataset
from src.model.spectra2selfies import Spectra2Seflies
from src.scheduler import CosineWarmupScheduler
from src.augmentation import *


def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/fg26_spectra2selfies.yaml', help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use (e.g., cuda:0, cpu)')
    
    # Sample inference args
    parser.add_argument('--num_beams', type=int, default=20)
    parser.add_argument('--num_return_sequences', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--length_penalty', type=float, default=0.0)
    parser.add_argument('--max_length', type=int, default=100)
    
    return parser.parse_args()


SPEC_TYPES = ['ir', 'raman', 'uv', 'xas-c', 'xas-n', 'xas-o']


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    if config.trainer.mixed_precision == False:
        config.trainer.mixed_precision = 'no'

    # Set random seed
    set_random_seed(config.seed)
    
    # Set up logging
    root = get_project_root_dir()
    logger = Logger(**config.logger, root=os.path.join(root, 'logs'))
    log_path = logger.log_path
    logger.info(f"Saving results to {log_path}")
    
    # Set up device
    device = args.device
    logger.info(f"Using device: {device}")
    
    # Set up inference config
    inference_config = {
        'num_beams': args.num_beams,
        'num_return_sequences': args.num_return_sequences,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'do_sample': args.do_sample,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'length_penalty': args.length_penalty,
        'max_length': args.max_length
    }
    
    # Save inference config for reference
    with open(os.path.join(log_path, 'inference_config.json'), 'w') as f:
        json.dump(inference_config, f, indent=4)

    # Set up tokenizer
    tokenizer = get_tokenizer('./tokenizer/bart-base', config.tokenizer_path)
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    tokenizer.save_pretrained(os.path.join(log_path, 'tokenizer'))
    
    # Set up data transformations
    base_spectrum_pipline = build_augmentation(config.spectrum_pipline.base_spectrum_pipline, logger)
    spectrum_pipline = {}
    for spec_type in SPEC_TYPES:
        spectrum_pipline[spec_type] = build_augmentation(
            config.spectrum_pipline.get(f'{spec_type}_pipline'), logger
        ) + base_spectrum_pipline
    smiles_pipline = build_augmentation(config.smiles_pipline, logger)
    
    # Load datasets
    if config.random_split:
        logger.info('Random split dataset')
        dataset = SpectraWithSelfiesTokenDataset(
            **config.dataset, 
            tokenizer=tokenizer, 
            transform={**spectrum_pipline, 'smiles': smiles_pipline}
        )
        train_len = int(config.train_test_split * len(dataset))
        test_len = len(dataset) - train_len
        train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
        logger.info(f'Training set size: {train_len}, Test set size: {test_len}')
        
        # Save dataset indices for reproducibility
        torch.save({
            'train_indices': train_dataset.indices,
            'test_indices': test_dataset.indices,
        }, os.path.join(log_path, 'dataset_indices.pth'))

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.trainer.batch_size, 
            shuffle=True, 
            collate_fn=train_dataset.dataset.collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.trainer.batch_size, 
            shuffle=False, 
            collate_fn=test_dataset.dataset.collate_fn
        )
    else:
        logger.info('Load dataset from Train/Test files')
        train_dataset = SpectraWithSelfiesTokenDataset(
            **config.train_dataset, 
            tokenizer=tokenizer, 
            transform={**spectrum_pipline, 'smiles': smiles_pipline}
        )
        test_dataset = SpectraWithSelfiesTokenDataset(
            **config.test_dataset, 
            tokenizer=tokenizer, 
            transform={**spectrum_pipline, 'smiles': smiles_pipline}
        )
        logger.info(f'Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}')
    
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.trainer.batch_size, 
            shuffle=True, 
            collate_fn=train_dataset.collate_fn
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.trainer.batch_size // config.trainer.test_rate, 
            shuffle=False, 
            collate_fn=test_dataset.collate_fn
        )

    # Calculate total training steps
    max_steps = len(train_loader) * config.trainer.epochs
    logger.info(f'Total training steps: {max_steps}')
    
    # Create model
    model = Spectra2Seflies(**config.model, tokenizer=tokenizer)
    logger.info(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Load pretrained weights if specified
    if hasattr(config, 'pretrain') and config.pretrain:
        model_dict = torch.load(
            os.path.join(config.pretrain, 'best.pt'), 
            map_location=torch.device("cpu")
        )
        if 'model_state_dict' in model_dict:
            model_dict = model_dict['model_state_dict']
        model.load_state_dict(model_dict)
        logger.info(f'Loaded pretrained model from {config.pretrain}')
    
    # Set up optimizer with weight decay discrimination
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.optimizer.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        params=grouped_params, 
        lr=config.optimizer.lr
    )
    
    lr_scheduler = CosineWarmupScheduler(
        warmup=config.scheduler.warmup, 
        optimizer=optimizer, 
        max_steps=max_steps
    )
    
    # Set up training model
    train_model = KerasModel(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
    )
    
    # Save configuration
    OmegaConf.save(config, os.path.join(log_path, 'config.yaml'))
    
    # Set up checkpoint callback
    checkpoint_callback = EpochCheckpoint(
        os.path.join(log_path, 'checkpoint'), 
        **config.callbacks.EpochCheckpoint
    )
    
    # Set up validation config
    validation_config = {
        'num_beams': 2,
        'num_return_sequences': 2,
        'max_length': 100,
        'top_k': 50
    }
    
    logger.info("Starting training...")
    # Train model
    history = train_model.fit(
        train_loader, 
        test_loader, 
        ckpt_path=os.path.join(log_path, 'best.pt'), 
        device=device,
        callbacks=[checkpoint_callback],
        inference_config=validation_config,
        history_save_path=os.path.join(log_path, 'history.png'),
        **config.trainer

    )
    
    # Evaluate best model
    logger.info("Evaluating best model...")
    val_metrics = train_model.evaluate(
        test_loader, 
        ckpt_path=os.path.join(log_path, 'best.pt'),
        device=device,
        inference_only=True
    )
    logger.info(f"Validation metrics: {val_metrics}")
    
    # Run inference with best model
    logger.info("Running inference with best model...")
    inference_results = train_model.run_inference(
        test_loader,
        ckpt_path=os.path.join(log_path, 'best.pt'),
        inference_config=inference_config,
        device=device,
        save_dir=os.path.join(log_path, f'inference_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    )
    
    logger.info(f"Inference results: Top-1 accuracy: {inference_results['top1_chemical_accuracy']:.4f}")
    logger.info(f"Results saved to: {inference_results['results_path']}")
    logger.info("Training and evaluation completed successfully")


if __name__ == '__main__':
    main()
