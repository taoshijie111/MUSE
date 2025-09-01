from transformers import AutoTokenizer
from tokenizers import pre_tokenizers, Regex
from omegaconf.listconfig import ListConfig
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import selfies as sf
import numpy as np


import sys
sys.path.append('/home/user/tsj/Spectra2Molecule')
from src.tools.tokenizer import APETokenizer


SPEC_TYPES = ['ir', 'raman', 'uv', 'xas-c', 'xas-n', 'xas-o']


def load_nmr_tokenizer():
    """Load and configure the NMR tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(r"\d{1,3}(?=(\d{3})*\b)"), behavior="isolated", invert=False),
        pre_tokenizers.Split(pattern=Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"), 
                            behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    ])
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    return tokenizer


class Spectra2Seflies(nn.Module):
    def __init__(
        self,
        tokenizer,
        specific_tokenizer=False,
        spec_type=['ir', 'uv', 'xas-c', 'nmr'],
        specific_spec_type=['nmr-c', 'nmr-h', 'ms'], 
        add_spec_type_token=True,
        post_embedding=True,
        interp_num=4000,
        
        pre_embedding_model_config=None,
        embedding_model_config=None,
        llm_model_config=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.specific_tokenizer = specific_tokenizer
        self.interp_num=interp_num
        sepc_list = self.expand_spectra_type(spec_type)
        
        self.specific_spec_list = []
        self.spec_list = []
        for spec in sepc_list:
            if spec in specific_spec_type:
                self.specific_spec_list.append(spec)
            else:
                self.spec_list.append(spec)
        if len(self.specific_spec_list) >= 1:
            self.specific_tokenizer = True
        # Initialize embedders
        ## pre embedding
        self.pre_spec_embedders = self._init_pre_spec_embedders(pre_embedding_model_config.name, pre_embedding_model_config.args)
        ## post embedding
        if post_embedding:
            vocab_size_list = [interp_num // patch for patch in  pre_embedding_model_config.args.patch_sizes]
            self.post_embedding_model = self._init_embedding_model(embedding_model_config.name, vocab_size_list, embedding_model_config.args, is_specific=False)
        if self.specific_tokenizer:
            vocab_size = len(load_nmr_tokenizer())
            self.specific_embedding_model = self._init_embedding_model(embedding_model_config.name, vocab_size, embedding_model_config.args, is_specific=True)
            self.specific_embedding_model.resize_token_embeddings(vocab_size)
        
        # Initialize LLM
        self.llm_model = self._init_llm_model(llm_model_config.name, llm_model_config.args)
        self.llm_model.resize_token_embeddings(len(tokenizer))
         # Initialize spec type embeddings
        self.add_spec_type_token = add_spec_type_token
        if self.add_spec_type_token:
            self.spec_type_embeddings = self._init_spec_type_embeddings(pre_embedding_model_config.args.output_size)
        
        # bos and eos token 
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
    
    def _init_pre_spec_embedders(self, name, config):
        """Initialize spectral embedders."""
        assert len(config.patch_sizes) == len(self.spec_list), "Mismatch between spec_list except specific spectra and patch sizes."
        if name.lower() == 'vit':
            patch_sizes = config['patch_sizes']
            output_size = config['output_size']
            
            return nn.ModuleList([
            nn.Sequential(
                Rearrange('b c (n p) -> b n (p c)', p=patch_sizes[i]),
                nn.LayerNorm(patch_sizes[i]),
                nn.Linear(patch_sizes[i], output_size),
            ) for i in range(len(self.spec_list))
        ])
        elif name.lower() == 'linear':
            from src.model.module import MLP
            patch_sizes = config['patch_sizes']
            hidden_size = config['hidden_size']
            output_size = config['output_size']
            return nn.ModuleList([
                MLP(patch_sizes[i], hidden_size, output_size, 4) for i in range(len(self.spec_list))
            ])
        elif name.lower() == 'cnn':
            from src.model.module import MultiScaleResidualBlock, AttentionPooling
            output_size = config['output_size']
            return nn.ModuleList([
                nn.Sequential(
                    MultiScaleResidualBlock(1, 64 * 3),
                    nn.MaxPool1d(5),
                    
                    MultiScaleResidualBlock(64 * 3, 128 * 3),
                    nn.MaxPool1d(5),
                    
                    MultiScaleResidualBlock(128 * 3, 256 * 3),
                    nn.MaxPool1d(5),
                    
                    MultiScaleResidualBlock(256 * 3, output_size),
                    
                    Rearrange('b c l -> b l c')
                    
                ) for i in range(len(self.spec_list))
            ])
    
    def _init_embedding_model(self, model_name, vocab_size, kwargs, is_specific=False):
        """Initialize embedding model (BERT or RoBERTa).
        
        Args:
            model_name (str): Name of the embedding model ('bert' or 'roberta').
            kwargs (dict): Configuration parameters for the model.
            is_specific (bool): Whether the model is for specific spectrum. If True, returns a single instance instead of a ModuleList.
        
        Returns:
            Union[nn.ModuleList, nn.Module]: Embedding model(s).
        """
        def get_model(model_name, vocab_size, **kwargs):
            if model_name == 'bert':
                from transformers.models.bert import BertModel, BertConfig
                model = BertModel(config=BertConfig(
                    **kwargs,
                    vocab_size=vocab_size,
                    pad_token_id=self.tokenizer.pad_token_id,  # 1
                    bos_token_id=self.tokenizer.bos_token_id,  # 0
                    eos_token_id=self.tokenizer.eos_token_id,  # 2
                ))
            elif model_name == 'roberta':
                from transformers.models.roberta import RobertaModel, RobertaConfig
                model = RobertaModel(config=RobertaConfig(
                    **kwargs,
                    vocab_size=vocab_size,
                    pad_token_id=self.tokenizer.pad_token_id,  # 1
                bos_token_id=self.tokenizer.bos_token_id,  # 0
                eos_token_id=self.tokenizer.eos_token_id,  # 2
                ))
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
            return model
        
        # Return a single model for NMR, otherwise return a ModuleList
        return get_model(model_name, vocab_size, **kwargs) if is_specific else nn.ModuleList([get_model(model_name, vocab_size[idx], **kwargs) for idx in range(len(self.spec_list))])

    def _init_llm_model(self, model_name, kwargs):
        """Initialize the LLM model (BART)."""
        if model_name == 'bart':
            from transformers.models.bart import BartForConditionalGeneration, BartConfig
            return BartForConditionalGeneration(config=BartConfig(
                **kwargs,
                vocab_size=len(self.tokenizer),
                pad_token_id=self.tokenizer.pad_token_id,  # 1
                bos_token_id=self.tokenizer.bos_token_id,  # 0
                eos_token_id=self.tokenizer.eos_token_id,  # 2
                decoder_start_token_id=self.tokenizer.bos_token_id,  # 0
            ))
        raise ValueError(f"Unsupported LLM model: {model_name}")
    
    def _init_spec_type_embeddings(self, hidden_size):
        """Initialize embeddings for spectral types."""
        return nn.ParameterDict({
            spec_type: nn.Parameter(torch.randn(hidden_size))
            for spec_type in SPEC_TYPES
        })
    
    def expand_spectra_type(self, spec_type):
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
        return spec_list
  
    def forward(self, batch):
        """Forward pass for training."""
        smiles, selfies, encode, spectra, specific_spec = batch
        encoder_hidden_states, attention_mask = self._encode_inputs(spectra, specific_spec)
        output = self.llm_model(inputs_embeds=encoder_hidden_states, labels=encode['labels'].contiguous(), attention_mask=attention_mask)
        return {'loss': output['loss']}
    
    @torch.no_grad()
    def infer(self, spectra, specific_spec, pred_config):
        encoder_hidden_states, attention_mask = self._encode_inputs(spectra, specific_spec)
        beam_output = self._generate_output(encoder_hidden_states, pred_config, attention_mask)
        return self._decode_output(beam_output)
    
    def _encode_inputs(self, spectra, specific_spec):
        """Encode input spectra and specific_spec data."""
        if torch.numel(spectra):
            b, s, c, l = spectra.shape
            spec_num = len(self.spec_list)
            assert s == spec_num, "Mismatch between spec_num and input spectra dimension."
            # normal spectra
            spectra = spectra.view(b * s, c, l)
        embeddings = []
        for i, spec_type in enumerate(self.spec_list):
            spec = spectra[i::spec_num]
            # pre embedding
            embedded_spec = self.pre_spec_embedders[i](spec)
            if self.add_spec_type_token:
                type_embedding = self.spec_type_embeddings[spec_type].expand(embedded_spec.size(0), 1, -1).to(embedded_spec.device)
                embedded_spec = torch.cat([type_embedding, embedded_spec, type_embedding], dim=1)
            # post embedding
            if hasattr(self, 'post_embedding_model'):
                embedded_spec = self.post_embedding_model[i](inputs_embeds=embedded_spec).last_hidden_state
            embeddings.append(embedded_spec)
        # specific spectra
        if self.specific_tokenizer:
            specific_spec_embedding = self.specific_embedding_model(**specific_spec).last_hidden_state
            embeddings.append(specific_spec_embedding)
        
        encoder_hidden_states = torch.cat(embeddings, dim=1)
        attention_mask = torch.ones((encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]), dtype=torch.float32, device=encoder_hidden_states.device)
        if self.specific_tokenizer:
            attention_mask[:, -specific_spec['attention_mask'].shape[1]:] = specific_spec['attention_mask']
        return encoder_hidden_states, attention_mask
    
    def _generate_output(self, encoder_hidden_states, pred_config, attention_mask=None):
        return self.llm_model.generate(
            inputs_embeds=encoder_hidden_states,
            decoder_start_token_id=self.bos_token_id,
            return_dict_in_generate=True,
            attention_mask=attention_mask,
            early_stopping=True,
            output_scores=True,
            bad_words_ids=[[4]],    # id 4 是无意义的
            **pred_config
        )

    def _decode_output(self, beam_output, original=False):
        """Decode output sequences to SMILES."""
        result = beam_output.sequences
        sequence_scores = beam_output.sequences_scores
        # cand = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", "")
        #         for g in result]
        
        cand = []
        for g in result:
            try:
                decoded = self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                cand.append(decoded.replace(" ", ""))
            except TypeError:
                print(f"Failed to decode: {g}")
                cand.append("[C]")  # 返回空字符串或占位符


        smiles = [sf.decoder(selfies) for selfies in cand]
        smiles_and_score = []
        for seq, score in zip(smiles, sequence_scores):
            smiles_and_score.append(seq+'_'+str(np.exp(score.item())))
        if original:
            return smiles, smiles_and_score, beam_output
        else:
            return smiles, smiles_and_score

    

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from src.tools.tokenizer import APETokenizer
    from src.dataset.spectra_with_mol import SpectraWithSelfiesTokenDataset
    from src.augmentation import AugmentationPipeline, NormalizeByArea, SpectrumMax
    
    args = OmegaConf.load('/home/user/tsj/Spectra2Molecule/configs/fg26_spectra2selfies.yaml')
    from src.tools.tokenizer import get_tokenizer
    from src.augmentation import AugmentationPipeline, NormalizeByArea, SmilesCanonical, build_augmentation
    import matplotlib.pyplot as plt
    import logging
    
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

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)
    
    model = Spectra2Seflies(**args.model, tokenizer=tokenizer)
    # print(model)
    for batch in dataloader:
        print(model(batch))
        break
    
    