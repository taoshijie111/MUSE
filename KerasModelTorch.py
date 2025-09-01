import sys, datetime
from tqdm import tqdm
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
import torch
import os
import json
import transformers
from rdkit import Chem

def remove_smiles_chirality(smiles):
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

def same_sf(sm1, sm2):
    """Compare two SMILES strings by converting to InChI Keys"""
    try:
        sm1 = remove_smiles_chirality(sm1)
        key1 = Chem.MolToInchiKey(Chem.MolFromSmiles(sm1))
        key2 = Chem.MolToInchiKey(Chem.MolFromSmiles(sm2))
        return key1 == key2 and key1 is not None
    except:
        return False

def topK_metric(gts, topk_preds, plain=False, reduction='sum_weighted'):
    """
    Calculate top-K accuracy metrics for molecular predictions.
    
    Args:
        gts: Ground truth SMILES
        topk_preds: Top-K predicted SMILES for each ground truth
        plain: Whether to use exact string matching (True) or chemical matching (False)
        reduction: Method to reduce metrics ('raw', 'raw_weighted', 'sum', 'sum_weighted')
    
    Returns:
        Metrics based on specified reduction method
    """
    try:
        weights = np.array([0.4, 0, 0.1, 0, 0.1, 0, 0, 0, 0, 0.4])
        hits = np.zeros((len(topk_preds), len(topk_preds[0])))
        for idx, (gt, topk_pred) in tqdm(enumerate(zip(gts, topk_preds)), leave=False, desc='Calculate Accuracy', ncols=100):
            for i, pred in enumerate(topk_pred):
                if not plain:
                    match_func = same_sf
                else:
                    match_func = lambda x, y: x == y
                if match_func(gt, pred):
                    hits[idx, i:] = 1
                    break

        score = np.mean(hits, axis=0)
        if reduction.endswith('weighted'):
            score = score * weights[:hits.shape[1]]
        if reduction.startswith('raw'):
            return score.tolist()
        elif reduction.startswith('sum'):
            return score.sum().item()
    except:
        return 0


class StepRunner:
    def __init__(self, net, device='cuda:0', stage="train", metrics_dict=None,
                 optimizer=None, gradient_accumulation_steps=1, max_norm=1.0, lr_scheduler=None, **kwargs):
        """
        Initialize the StepRunner object

        Args:
            net: Neural network model
            device: Device to run on (e.g., 'cuda:0', 'cpu')
            stage: Training or evaluation stage
            metrics_dict: Dictionary of metrics functions
            optimizer: Optimizer for training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_norm: Maximum norm for gradient clipping
            lr_scheduler: Learning rate scheduler
            **kwargs: Additional keyword arguments
        """
        self.net, self.metrics_dict, self.stage = net, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.kwargs = kwargs
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_norm = max_norm
        
        # Set the network to training mode during the training stage, and evaluation mode otherwise
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch, scaler, step):
        """
        Perform a training or evaluation step.

        Args:
            batch: Input data batch
            scaler: GradScaler for mixed precision
            step: Current step number

        Returns:
            Tuple of dictionaries containing step losses and step metrics.
        """
        # Move batch data to device
        batch = list(batch)
        for i, value in enumerate(batch):
            if isinstance(value, torch.Tensor):
                batch[i] = value.to(self.device)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    value[k] = v.to(self.device)
                batch[i] = value
            elif isinstance(value, dict):
                # Handle dictionary inputs like specific_spec
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.to(self.device)
                batch[i] = value
                
        # Compute loss with automatic mixed precision
        with torch.cuda.amp.autocast():
            loss = self.net(batch)
            train_loss = loss['loss']
            train_loss = train_loss / self.gradient_accumulation_steps
            
            # Backward pass and optimization (only during training)
            if self.stage == "train" and self.optimizer is not None:
                scaler.scale(train_loss).backward()
                
                if step % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.net.parameters(), self.max_norm)
                    
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    self.optimizer.zero_grad()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

        # Collect losses and metrics
        all_loss = {}
        for name, lo in loss.items():
            all_loss[self.stage + '_' + name] = lo.item()

        step_losses = {**all_loss}
        step_metrics = {}

        # Include learning rate in metrics if available
        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
                
        return step_losses, step_metrics


class EpochRunner:
    def __init__(self, step_runner, quiet=False):
        """
        Initialize the EpochRunner object

        Args:
            step_runner: StepRunner object for handling individual training or evaluation steps
            quiet: Flag to control whether to display progress bar and logs
        """
        self.step_runner = step_runner
        self.stage = step_runner.stage
        self.net = step_runner.net
        self.quiet = quiet

    def __call__(self, dataloader, smiles_selected_num=None):
        """
        Perform a complete epoch of training or evaluation

        Args:
            dataloader: DataLoader providing batches of data for the epoch
            smiles_selected_num: Optional dictionary to track SMILES selection frequency

        Returns:
            Dictionary containing aggregated epoch losses and metrics
        """
        # Determine the size of the dataset
        n = dataloader.size if hasattr(dataloader, 'size') else len(dataloader)

        # Initialize tqdm progress bar
        loop = tqdm(enumerate(dataloader, start=1),
                    total=n,
                    file=sys.stdout,
                    ncols=100,
                    leave=False)
                    
        epoch_losses = {}
        scaler = torch.cuda.amp.GradScaler()
        
        for step, batch in loop:
            # Perform a step with the provided StepRunner
            step_losses, step_metrics = self.step_runner(batch, scaler, step)
            step_log = dict(step_losses, **step_metrics)
            
            # Track SMILES selection if needed
            if smiles_selected_num is not None:
                for smi in batch[0]:
                    smiles_selected_num[smi] += 1
            
            # Accumulate step losses for computing epoch losses
            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            # Update progress bar during the epoch
            if step < n:
                loop.set_postfix({**step_log, **step_metrics})

                if hasattr(self, 'progress'):
                    post_log = dict(**{'i': step, 'n': n}, **step_log)
                    self.progress.set_postfix(**post_log)

            # Compute and display epoch-level metrics at the end of the epoch
            elif step == n:
                epoch_metrics = step_metrics
                epoch_metrics.update({self.stage + "_" + name: metric_fn.compute().item()
                                      for name, metric_fn in self.step_runner.metrics_dict.items()})
                epoch_losses = {k: v / step for k, v in epoch_losses.items()}
                epoch_log = dict(epoch_losses, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                # Update progress bar if available
                if hasattr(self, 'progress'):
                    post_log = dict(**{'i': step, 'n': n}, **epoch_log)
                    self.progress.set_postfix(**post_log)

                # Reset stateful metrics for the next epoch
                for name, metric_fn in self.step_runner.metrics_dict.items():
                    metric_fn.reset()
            else:
                break
                
        return epoch_log


class KerasModel(torch.nn.Module):
    StepRunner, EpochRunner = StepRunner, EpochRunner

    def __init__(self, net, metrics_dict=None, optimizer=None, lr_scheduler=None, logger=None, **kwargs):
        """
        Initialize the KerasModel.

        Args:
            net: Neural network model
            metrics_dict: Dictionary of metrics functions
            optimizer: Optimizer for training
            lr_scheduler: Learning rate scheduler
            logger: Logger for output messages
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.net, self.metrics_dict = net, torch.nn.ModuleDict(metrics_dict) if metrics_dict else torch.nn.ModuleDict({})
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.net.parameters(), lr=3e-4)
        self.lr_scheduler = lr_scheduler
        self.kwargs = kwargs
        self.from_scratch = True
        self.logger = logger

    def evaluate_inference(self, val_data, inference_config=None, device='cuda:0'):
        """
        Evaluate the model using inference and calculate top-K accuracy.

        Args:
            val_data: Validation data
            inference_config: Configuration for inference during validation
            device: Device to run inference on

        Returns:
            Dictionary of evaluation metrics including top-K accuracy.
        """
        # Set default inference configuration if not provided
        if inference_config is None:
            inference_config = {
                'num_beams': 2,
                'num_return_sequences': 2,
                'max_length': 100
            }

        # Save the current model state
        was_training = self.net.training
        
        # Set model to evaluation mode
        self.net.eval()
        self.net.to(device)
        
        # Collect predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        
        # Perform inference on validation data
        with torch.no_grad():
            for batch in tqdm(val_data, file=sys.stdout, ncols=100, leave=False, desc='Inference'):
                # Extract batch components
                smiles, selfies, encode, spectra, specific_spec = batch
                
                # Move data to device
                spectra = spectra.to(device)
                if specific_spec is not None:
                    specific_spec = {k: v.to(device) for k, v in specific_spec.items()}
                
                # Perform inference
                pred_smiles, _ = self.net.infer(spectra, specific_spec, inference_config)
                
                # Collect predictions and ground truths
                num_return_sequences = inference_config.get('num_return_sequences', 1)
                for i in range(len(smiles)):
                    all_ground_truths.append(smiles[i])
                    batch_preds = pred_smiles[i*num_return_sequences:(i+1)*num_return_sequences]
                    # Ensure we have exactly num_return_sequences predictions per sample
                    if len(batch_preds) < num_return_sequences:
                        batch_preds = batch_preds + [''] * (num_return_sequences - len(batch_preds))
                    all_predictions.append(batch_preds)
        
        # Calculate metrics
        metrics = {}
        accuracy_raw = topK_metric(all_ground_truths, all_predictions, plain=False, reduction='raw')
        # plain_accuracy_raw = topK_metric(all_ground_truths, all_predictions, plain=True, reduction='raw')
        
        # Create metrics dictionary
        for k, acc in enumerate(accuracy_raw):
            metrics[f'val_top{k+1}_accuracy'] = acc
        # for k, acc in enumerate(plain_accuracy_raw):
        #     metrics[f'val_top{k+1}_plain_accuracy'] = acc
        
        # Restore the model state
        if was_training:
            self.net.train()
        
        # Return metrics
        return metrics

    def save_ckpt(self, ckpt_path=None):
        """
        Save the model checkpoint

        Args:
            ckpt_path: Path to save the checkpoint
        """
        torch.save({'model_state_dict': self.net.state_dict()}, 
                   ckpt_path if ckpt_path is not None else self.ckpt_path)
        
    def load_ckpt(self, ckpt_path=None):
        """
        Load the model checkpoint

        Args:
            ckpt_path: Path to the checkpoint
        """
        net_dict = torch.load(ckpt_path if ckpt_path is not None else self.ckpt_path,
                             map_location='cpu')
        
        if 'model_state_dict' in net_dict:
            net_dict = net_dict['model_state_dict']
            
        self.net.load_state_dict(net_dict)
        self.from_scratch = False

    def forward(self, x):
        """
        Forward pass through the model

        Args:
            x: Input data

        Returns:
            Model predictions
        """
        return self.net.forward(x)

    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint',
            patience=5, monitor="val_top1_accuracy", mode="max",
            callbacks=None, plot=True, wandb=False,
            mixed_precision='no', cpu=False, gradient_accumulation_steps=1, 
            max_norm=1.0, device='cuda:0', inference_config=None, 
            history_save_path='history.png', val_interval=5, **kwargs):
        """
        Train the model.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            ckpt_path: Path to save model checkpoints
            patience: Number of epochs with no improvement after which training will be stopped
            monitor: Metric to monitor for early stopping (default: val_top1_accuracy)
            mode: 'min' for minimizing the monitor metric, 'max' for maximizing (default: max)
            callbacks: List of callback functions
            plot: Whether to plot training progress
            wandb: Whether to use WandB for logging
            mixed_precision: Mixed precision training (currently not used)
            cpu: Use CPU for training (overrides device)
            gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
            max_norm: Maximum norm for gradient clipping
            device: Device to use for training ('cuda:0', 'cpu', etc.)
            inference_config: Configuration for inference during validation
            history_save_path: Path to save training history plot
            val_interval: Frequency (in epochs) to perform validation inference

        Returns:
            DataFrame containing training history.
        """
        self.__dict__.update(locals())

        if cpu:
            device = 'cpu'
            
        device_type = '🐌' if 'cpu' in device else ('⚡️' if 'cuda' in device else '🚀')
        self.logger.info("<<<<<< " + device_type + " " + device + " is used >>>>>>")
        
        self.net.to(device)
        train_dataloader, val_dataloader = train_data, val_data
        train_dataloader.size = train_data.size if hasattr(train_data, 'size') else len(train_data)
        train_dataloader.size = min(train_dataloader.size, len(train_dataloader))

        if val_data:
            val_dataloader.size = val_data.size if hasattr(val_data, 'size') else len(val_data)
            val_dataloader.size = min(val_dataloader.size, len(val_dataloader))

        self.history = {}
        callbacks = callbacks if callbacks is not None else []

        if bool(plot):
            try:
                from src.kerascallbacks.callbacks import VisProgress, VisMetric
                callbacks = [VisMetric(save_path=history_save_path), VisProgress()] + callbacks
            except ImportError:
                self.logger.info("Warning: Could not import visualization callbacks. Continuing without them.")

        if wandb != False:
            try:
                from src.kerascallbacks.callbacks import WandbCallback
                project = wandb if isinstance(wandb, str) else 'torchkeras'
                callbacks.append(WandbCallback(project=project))
            except ImportError:
                self.logger.info("Warning: Could not import WandB callback. Continuing without it.")

        self.callbacks = callbacks

        # Set default inference config if using top-K metrics
        if monitor.startswith('val_top') and inference_config is None:
            inference_config = {
                'num_beams': 2,
                'num_return_sequences': 2,
                'max_length': 100
            }

        # Initialize early stopping variables
        best_score = None
        early_stop_counter = 0
        best_epoch = 0
        last_recorded_values = {}  # To store the last recorded values for each metric
        inference_epochs = set()   # To keep track of which epochs actually ran inference

        for cb in self.callbacks:
            if hasattr(cb, 'on_fit_start'):
                cb.on_fit_start(model=self)

        start_epoch = 1 if self.from_scratch else 0
        quiet = bool(plot)
        
        from collections import defaultdict
        
        for epoch in range(start_epoch, epochs + 1):
            if not quiet:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.logger.info("\n" + "==========" * 8 + "%s" % nowtime)
                self.logger.info("Epoch {0} / {1}".format(epoch, epochs) + "\n")

            # 1, train -------------------------------------------------
            train_step_runner = self.StepRunner(
                net=self.net,
                device=device,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_norm=max_norm,
                stage="train",
                metrics_dict=deepcopy(self.metrics_dict),
                optimizer=self.optimizer if epoch > 0 else None,
                lr_scheduler=self.lr_scheduler if epoch > 0 else None,
                **self.kwargs
            )

            train_epoch_runner = self.EpochRunner(train_step_runner, quiet)
            train_metrics = {'epoch': epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            for cb in self.callbacks:
                if hasattr(cb, 'on_train_epoch_end'):
                    cb.on_train_epoch_end(model=self)
                    
            # 2, validate -------------------------------------------------
            if val_dataloader is not None:
                # Determine if this epoch should perform inference validation based on val_interval
                do_inference = (epoch % val_interval == 0 or epoch == 1 or epoch == epochs)
                
                # Check if we're using top-K metrics that require inference
                using_top_k_metric = monitor.startswith('val_top') and monitor.endswith('_accuracy')
                
                if using_top_k_metric:
                    if do_inference:
                        # Perform inference validation on scheduled epochs
                        self.logger.info(f"Epoch {epoch}: Performing inference validation (scheduled)")
                        val_metrics = {}
                        inference_metrics = self.evaluate_inference(val_dataloader, inference_config, device=device)
                        
                        # Record that this epoch ran actual inference
                        inference_epochs.add(epoch)
                        
                        for name, metric in inference_metrics.items():
                            # Store the latest values for each metric
                            last_recorded_values[name] = metric
                            # Add to history
                            self.history[name] = self.history.get(name, []) + [metric]
                            val_metrics[name] = metric
                            
                            # Log the top-1 accuracy
                            if name == 'val_top1_accuracy':
                                self.logger.info(f"Validation Top-1 Accuracy: {metric:.4f}")
                    else:
                        # For non-inference epochs, use the last recorded values for top-k metrics
                        val_metrics = {}
                        self.logger.info(f"Epoch {epoch}: Skipping inference validation (using last values)")
                        
                        # Get monitored metric value from last inference
                        if monitor in last_recorded_values:
                            # For monitoring purposes only, not used for training
                            val_metrics[monitor] = last_recorded_values[monitor]
                            self.logger.info(f"  Last recorded {monitor}: {val_metrics[monitor]:.4f}")
                            
                            # Add all known metrics to history to maintain consistency
                            for name, value in last_recorded_values.items():
                                self.history[name] = self.history.get(name, []) + [value]
                else:
                    # Traditional loss-based validation (always run)
                    val_step_runner = self.StepRunner(
                        net=self.net,
                        device=device,
                        stage="val",
                        metrics_dict=deepcopy(self.metrics_dict),
                        **self.kwargs
                    )
                    val_epoch_runner = self.EpochRunner(val_step_runner, quiet)
                    with torch.no_grad():
                        val_metrics = val_epoch_runner(val_dataloader)

                    for name, metric in val_metrics.items():
                        self.history[name] = self.history.get(name, []) + [metric]
            
                # Call validation callbacks
                for cb in self.callbacks:
                    if hasattr(cb, 'on_validation_epoch_end'):
                        cb.on_validation_epoch_end(model=self)
                
                # Early stopping check (only consider epochs where inference was actually performed)
                if using_top_k_metric:
                    if do_inference:  # Only update early stopping when we actually did inference
                        current_score = val_metrics[monitor]
                        
                        # First validation or better score
                        if best_score is None or (mode == "max" and current_score > best_score) or \
                           (mode == "min" and current_score < best_score):
                            # Save best model
                            best_score = current_score
                            best_epoch = epoch
                            early_stop_counter = 0
                            
                            self.save_ckpt(ckpt_path)
                            if not quiet:
                                self.logger.info("Saved checkpoint at {}".format(ckpt_path))
                                self.logger.info("<<<<<< reach best {0} : {1} >>>>>>".format(
                                    monitor, best_score))
                        else:
                            # Increment only after validation epochs
                            early_stop_counter += 1
                            
                            if not quiet:
                                self.logger.info(f"Early stopping counter: {early_stop_counter}/{patience}")
                            
                        # Check if patience reached (in terms of validation epochs)
                        if early_stop_counter >= patience:
                            if not quiet:
                                self.logger.info(
                                    "<<<<<< {} without improvement for {} validation epochs, "
                                    "early stopping >>>>>> \n"
                                .format(monitor, patience))
                            break
                else:
                    # Traditional early stopping logic for non-inference metrics
                    arr_scores = self.history[monitor]
                    best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
                    if best_score_idx==len(arr_scores)-1:
                        self.save_ckpt(ckpt_path)
                        if not quiet:
                            self.logger.info("Saved checkpoint at {}".format(ckpt_path))
                            self.logger.info("<<<<<< reach best {0} : {1} >>>>>>".format(
                                monitor, arr_scores[best_score_idx]))

                    if len(arr_scores)-best_score_idx>patience:
                        break
        
        # Create and save history dataframe
        dfhistory = pd.DataFrame(self.history)
        metrics_dir = os.path.dirname(ckpt_path)
        os.makedirs(metrics_dir, exist_ok=True)
        dfhistory.to_csv(os.path.join(metrics_dir, 'history.csv'), index=False)
        
        # Also save a filtered history with only the inference epochs (for analysis)
        if using_top_k_metric and len(inference_epochs) > 0:
            inference_indices = [i for i, e in enumerate(dfhistory['epoch']) if e in inference_epochs]
            inference_history = dfhistory.iloc[inference_indices]
            inference_history.to_csv(os.path.join(metrics_dir, 'inference_history.csv'), index=False)
        
        # Call fit_end callbacks
        for cb in self.callbacks:
            if hasattr(cb, 'on_fit_end'):        
                cb.on_fit_end(model=self) 
                
        if epoch < epochs:
            self.logger.info(
                "<<<<<< {} without improvement, early stopping at epoch {} >>>>>> \n"
            .format(monitor, epoch))
            self.logger.info(f"Best {monitor} value: {best_score:.4f} at epoch {best_epoch}")
        
        return dfhistory
        
    def evaluate(self, val_data, ckpt_path, device='cuda:0', quiet=False, inference_only=True):
        """
        Evaluate the model on validation data

        Args:
            val_data: Validation data
            ckpt_path: Path to the checkpoint
            device: Device to run on
            quiet: Whether to suppress evaluation progress logs
            inference_only: Whether to only perform inference evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load the checkpoint
        self.net.cpu()
        self.load_ckpt(ckpt_path)
        self.net.to(device)
        
        # Initialize metrics dictionary
        val_metrics = {}
        
        # Perform loss-based evaluation if requested
        if not inference_only:
            # Initialize StepRunner for validation
            val_step_runner = self.StepRunner(
                net=self.net, 
                stage="val",
                device=device,
                metrics_dict=deepcopy(self.metrics_dict),
            )

            # Initialize EpochRunner for validation
            val_epoch_runner = self.EpochRunner(val_step_runner, quiet=quiet)

            # Evaluate on validation data without gradient computation
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
        
        # Add inference-based metrics for a more comprehensive evaluation
        inference_config = {
            'num_beams': 2,
            'num_return_sequences': 2,
            'max_length': 100
        }
        
        inference_metrics = self.evaluate_inference(val_data, inference_config, device=device)
        val_metrics.update(inference_metrics)
        
        # Print top-1 accuracy
        if 'val_top1_accuracy' in val_metrics and not quiet:
            self.logger.info(f"Final Validation Top-1 Accuracy: {val_metrics['val_top1_accuracy']:.4f}")
        
        # Save metrics to file
        metrics_dir = os.path.dirname(ckpt_path)
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, 'val_metrics.json'), 'w') as file:
            json.dump(val_metrics, file, indent=4)
            
        return val_metrics
        
    def run_inference(self, test_data, ckpt_path, inference_config, device='cuda:0', save_dir=None, calculate_acc=True):
        """
        Run inference on test data and save results
        
        Args:
            test_data: Test data loader
            ckpt_path: Path to the model checkpoint
            inference_config: Configuration for inference
            device: Device to run inference on
            save_dir: Directory to save results (if None, use checkpoint directory)
        
        Returns:
            DataFrame with inference results
        """
        import pandas as pd
        from src.tools.drawer import setup_nature_style
        from pathlib import Path
        from datetime import datetime
        setup_nature_style(100)
        # Load checkpoint
        self.load_ckpt(ckpt_path)
        self.net.to(device)
        self.net.eval()
        
        # Create save directory if not specified
        if save_dir is None:
            timestep = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = Path(os.path.dirname(ckpt_path)) / f'{timestep}'
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'sampling.csv'
        
        # Save inference config
        with open(os.path.join(save_dir, 'inference_args.json'), 'w') as f:
            json.dump(inference_config, f, indent=4)
        
        # Define function to save big CSV files
        def save_big_csv_file(data_df, save_path, index=False):
            save_path = Path(save_path)
            if save_path.is_file():
                data_df.to_csv(save_path, header=False, mode='a', index=index)
            else:
                data_df.to_csv(save_path, header=True, index=index)
        
        # Run inference
        loop = tqdm(enumerate(test_data, start=1), total=len(test_data), file=sys.stdout, ncols=100, leave=False)
        for step, batch in loop:
            batched_data = []
            true_smiles, true_selfies, encode, spectra, specific_spec = batch
            
            # Move data to device
            spectra = spectra.to(device)
            if specific_spec is not None:
                specific_spec = {k: v.to(device) for k, v in specific_spec.items()}
            
            # Run inference
            pred_smiles, smiles_and_score = self.net.infer(spectra, specific_spec, inference_config)
            
            # Organize results
            num_return_sequences = inference_config.get('num_return_sequences', 10)
            k = 0
            for j in range(0, len(pred_smiles), num_return_sequences):
                single_data = (
                    [true_smiles[k]] +
                    pred_smiles[j : j + num_return_sequences] +
                    smiles_and_score[j : j + num_return_sequences]
                )
                k += 1
                batched_data.append(single_data)
            
            # Save batch results
            batched_data = pd.DataFrame(
                batched_data, 
                columns=['true_smiles'] + 
                        [f'pred_smiles_{i}' for i in range(num_return_sequences)] + 
                        [f'pred_smiles_and_score_{i}' for i in range(num_return_sequences)]
            )
            save_big_csv_file(batched_data, save_path)

        if calculate_acc:
            inference_result = self.calculate_acc(save_path)
            return inference_result
        else:
            return {
                'results_path': str(save_path)
            }
    
    def calculate_acc(self, save_path):
        import matplotlib.pyplot as plt
        from pathlib import Path

        save_dir = Path(os.path.dirname(save_path))
        # Calculate and visualize metrics
        data = pd.read_csv(save_path)
        true_smiles = data['true_smiles'].tolist()
        num = len(data.iloc[0, 1:])
        pred_smiles = data.iloc[:, 1:num//2 + 1]
        pred_smiles = pred_smiles.values.tolist()
        
        # Plain text matching metrics
        metric_sf_plain = topK_metric(true_smiles, pred_smiles, plain=True, reduction='raw')
        with open(save_dir / 'metric_sf_plain.txt', 'w') as f:
            for m in metric_sf_plain:
                f.write(str(m) + '\n')
        
        plt.figure()
        plt.plot(metric_sf_plain, marker='*', markersize=8, color='r')
        plt.title('Plain Text Matching Accuracy')
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.savefig(save_dir / 'metric_sf_plain.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Chemical structure matching metrics
        plt.figure()
        metric_sf = topK_metric(true_smiles, pred_smiles, plain=False, reduction='raw')
        with open(save_dir / 'metric_sf.txt', 'w') as f:
            for m in metric_sf:
                f.write(str(m) + '\n')
        
        plt.plot(metric_sf, marker='*', markersize=8, color='r')
        plt.title('Chemical Structure Matching Accuracy')
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.savefig(save_dir / 'metric_sf.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return results summary
        return {
            'top1_plain_accuracy': metric_sf_plain[0],
            'top1_chemical_accuracy': metric_sf[0],
            'results_path': str(save_path)
        }