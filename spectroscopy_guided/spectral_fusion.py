import logging
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KerasModelTorch import KerasModel
from src.model.spectra2selfies import Spectra2Seflies
from transformers import BartTokenizer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datetime import datetime

from Spectra2SelfiesTrainerTorch import SPEC_TYPES, build_augmentation
from spectroscopy_guided.fusion_dataset import FusionSpectraDataset



parser = argparse.ArgumentParser()
parser.add_argument('--log_path', type=str, default='logs/FG26-SR-1-Raman10-XAS10-ViT-BERT_Spectra2Selfies_2025-03-27_17-11')
parser.add_argument('--smiles_pair_path', type=str, required=True)
parser.add_argument('--pair_fg_name', type=str, required=True) 
parser.add_argument('--spectra_dir', type=str, required=True)
parser.add_argument('--spectra_name', type=str, default='{}.npz')
parser.add_argument('--batch_size', type=int, default=180)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--resume_path', type=str, default=None)
sample_args = parser.add_argument_group('Sample Args')
sample_args.add_argument('--num_beams', type=int, default=3)
sample_args.add_argument('--num_return_sequences', type=int, default=3)
sample_args.add_argument('--top_k', type=int, default=50)
sample_args.add_argument('--top_p', type=float, default=1)
sample_args.add_argument('--temperature', type=float, default=1.0)
sample_args.add_argument('--do_sample', type=bool, default=False)
sample_args.add_argument('--no_repeat_ngram_size', type=int, default=0)
sample_args.add_argument('--length_penalty', type=float, default=0.0)
sample_args.add_argument('--max_length', type=int, default=200)
pred_args = parser.parse_args()

log_path = pred_args.log_path
# pair_smiles_path = os.path.join('/home/user/tsj/data/fg26/train_test_split/split_output/single_fg_combine_47', f'{pred_args.pair_fg_name}.txt')
pair_smiles_path = os.path.join(pred_args.smiles_pair_path, f'{pred_args.pair_fg_name}.txt')
batch_size = pred_args.batch_size
device = pred_args.device
resume_path = pred_args.resume_path

config = OmegaConf.load(os.path.join(log_path, 'config.yaml'))
tokenizer = BartTokenizer.from_pretrained(os.path.join(log_path, 'tokenizer'))
model = Spectra2Seflies(**config.model, tokenizer=tokenizer)

logger = logging.getLogger()
train_model = KerasModel(
        model,
        logger=logger,
    )

spectrum_pipline = {}
for spec_type in SPEC_TYPES:
    spectrum_pipline[spec_type] = build_augmentation(config.spectrum_pipline.get(f'{spec_type}_pipline'), logger=None)
smiles_pipline = build_augmentation(config.smiles_pipline, logger=None)
test_dataset = FusionSpectraDataset(root_data=pred_args.spectra_dir,
                                    name=pred_args.spectra_name, # 'fg26_{}_spectra.npz'  {}.npz
                                    spec_type=config.test_dataset.spec_type,
                                    tokenizer=tokenizer, 
                                    transform={**spectrum_pipline, 'smiles': smiles_pipline},
                                    mean=True)
test_dataset._reinit(pair_smiles_path, '.', resume_path=resume_path)
print(f'Pair Smiles Number is {len(test_dataset)}')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    
inference_config = {arg.dest: getattr(pred_args, arg.dest) for arg in sample_args._group_actions}

inference_results = train_model.run_inference(
        test_loader,
        ckpt_path=os.path.join(log_path, 'best.pt'),
        inference_config=inference_config,
        device=device,
        save_dir=os.path.join(log_path, f'{os.path.splitext(os.path.basename(pair_smiles_path))[0]}_mean_generation_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'),
        calculate_acc=False
    )
    
logger.info("Training and evaluation completed successfully")
