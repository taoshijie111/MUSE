"""
Callbacks for KerasModel PyTorch (single-GPU version).
"""
import os 
import sys
import torch
import datetime 
from copy import deepcopy
import numpy as np 
import pandas as pd 
from argparse import Namespace 
from KerasModelTorch import KerasModel


class TensorBoardCallback:
    def __init__(self, save_dir="runs", model_name="model",
                 log_weight=False, log_weight_freq=5):
        """
        TensorBoard callback for logging training progress

        Args:
        -  save_dir (str): Directory to save TensorBoard logs
        -  model_name (str): Name of the model
        -  log_weight (bool): Whether to log model weights
        -  log_weight_freq (int): Frequency of logging model weights during training
        """
        from torch.utils.tensorboard import SummaryWriter
        self.__dict__.update(locals())
        nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(save_dir, model_name, nowtime)
        self.writer = SummaryWriter(self.log_path)

    def on_fit_start(self, model: 'KerasModel'):
        """
        Callback function called at the beginning of model fitting

        Args:
        - model (KerasModel): The KerasModel being trained
        """
        # Log model weights
        if self.log_weight:
            # Direct access to model weights without accelerator
            for name, param in model.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
            self.writer.flush()

    def on_train_epoch_end(self, model: 'KerasModel'):
        """
        Callback function called at the end of each training epoch

        Args:
        - model (KerasModel): The KerasModel being trained.
        """
        epoch = max(model.history['epoch'])

        # Log model weights
        if self.log_weight and epoch % self.log_weight_freq == 0:
            for name, param in model.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()

    def on_validation_epoch_end(self, model: 'KerasModel'):
        """
        Callback function called at the end of each validation epoch

        Args:
        - model (KerasModel): The KerasModel being trained
        """
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        epoch = max(model.history['epoch'])

        # Log metrics
        dic = deepcopy(dfhistory.iloc[n - 1])
        dic.pop("epoch")

        metrics_group = {}
        for key, value in dic.items():
            g = key.replace("train_", '').replace("val_", '')
            metrics_group[g] = dict(metrics_group.get(g, {}), **{key: value})
        for group, metrics in metrics_group.items():
            self.writer.add_scalars(group, metrics, epoch)
        self.writer.flush()

    def on_fit_end(self, model: 'KerasModel'):
        """
        Callback function called at the end of model fitting

        Args:
        - model (KerasModel): The KerasModel being trained
        """
        # Log model weights
        epoch = max(model.history['epoch'])
        if self.log_weight:
            for name, param in model.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()
        self.writer.close()

        # Save history
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(self.log_path, 'dfhistory.csv'), index=None)


class WandbCallback:
    def __init__(self, project=None, config=None, name=None, save_ckpt=True, save_code=True):
        """
        WandbCallback for logging training progress using Weights & Biases

        Args:
        - project (str): Name of the project in W&B
        - config (dict or Namespace): Configuration parameters
        - name (str): Name of the run.
        - save_ckpt (bool): Whether to save model checkpoints
        - save_code (bool): Whether to save code artifacts
        """
        self.__dict__.update(locals())
        if isinstance(config, Namespace):
            self.config = config.__dict__
        if name is None:
            self.name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        import wandb
        self.wb = wandb

    def on_fit_start(self, model: 'KerasModel'):
        """
        Callback function called at the beginning of model fitting

        Args:
        - model (KerasModel): The KerasModel being trained
        """
        if self.wb.run is None:
            self.wb.init(project=self.project, config=self.config, name=self.name, save_code=self.save_code)
        model.run_id = self.wb.run.id

    def on_train_epoch_end(self, model: 'KerasModel'):
        """
        Callback function called at the end of each training epoch

        Args:
        - model (KerasModel): The KerasModel being trained
        """
        pass

    def on_validation_epoch_end(self, model: 'KerasModel'):
        """
        Callback function called at the end of each validation epoch

        Args:
        - model (KerasModel): The KerasModel being trained
        """
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        if n == 1:
            for m in dfhistory.columns:
                self.wb.define_metric(name=m, step_metric='epoch', hidden=False if m != 'epoch' else True)
            self.wb.define_metric(name='best_' + model.monitor, step_metric='epoch')

        dic = dict(dfhistory.iloc[n - 1])
        monitor_arr = dfhistory[model.monitor]
        best_monitor_score = monitor_arr.max() if model.mode == 'max' else monitor_arr.min()
        dic.update({'best_' + model.monitor: best_monitor_score})
        self.wb.run.summary["best_score"] = best_monitor_score
        self.wb.log(dic)

    def on_fit_end(self, model: 'KerasModel'):
        """
        Callback function called at the end of model fitting

        Args:
        - model (KerasModel): The KerasModel being trained
        """
        # Save dfhistory
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(self.wb.run.dir, 'dfhistory.csv'), index=None)

        # Save ckpt
        if self.save_ckpt:
            arti_model = self.wb.Artifact('checkpoint', type='model')
            if os.path.isdir(model.ckpt_path):
                arti_model.add_dir(model.ckpt_path)
            else:
                arti_model.add_file(model.ckpt_path)
            self.wb.log_artifact(arti_model)

        run_dir = self.wb.run.dir
        self.wb.finish()

        # Local save
        try:
            import shutil
            copy_fn = shutil.copytree if os.path.isdir(model.ckpt_path) else shutil.copy
            copy_fn(model.ckpt_path, os.path.join(run_dir, os.path.basename(model.ckpt_path)))
        except Exception as err:
            print(err)


class ProgressBar:
    """Simple progress bar for console output"""
    def __init__(self, iterable=None, total=None, width=30, file=sys.stdout):
        if iterable is not None:
            try:
                self.iterable = iterable
                self.total = len(iterable)
            except (TypeError, AttributeError):
                self.iterable = None
                self.total = total
        else:
            self.iterable = None
            self.total = total
        
        self.width = width
        self.file = file
        self.display = True
        self.n = 0
        self.comment_tail = ""

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < self.total:
            self.update(self.n)
            self.n += 1
            return self.n
        else:
            raise StopIteration
    
    def update(self, n):
        if self.display:
            percent = n / self.total
            size = int(self.width * percent)
            bar = '█' * size + '-' * (self.width - size)
            print(f'\r[{bar}] {int(100 * percent)}% {self.comment_tail}', end='', file=self.file)
            if n >= self.total:
                print(file=self.file)
    
    def set_postfix(self, **kwargs):
        self.comment_tail = ' '.join(f'{k}={v:.4f}' for k, v in kwargs.items())

    def on_interrupt(self, msg='Interrupted'):
        print(f'\n{msg}', file=self.file)


class VisProgress:
    def __init__(self):
        pass

    def on_fit_start(self, model: 'KerasModel'):
        """Callback at the beginning of the training

        Args:
            model (KerasModel): The KerasModel instance
        """
        self.progress = ProgressBar(total=model.epochs)
        model.EpochRunner.progress = self.progress

    def on_train_epoch_end(self, model: 'KerasModel'):
        """Callback at the end of each training epoch.

        Args:
            model (KerasModel): The KerasModel instance
        """
        pass

    def on_validation_epoch_end(self, model: "KerasModel"):
        """Callback at the end of each validation epoch

        Args:
            model (KerasModel): The KerasModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        self.progress.update(dfhistory['epoch'].iloc[-1])

    def on_fit_end(self, model: "KerasModel"):
        """Callback at the end of the entire training process

        Args:
            model (KerasModel): The KerasModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        if dfhistory['epoch'].max() < model.epochs:
            self.progress.on_interrupt(msg='')
        self.progress.display = False


class VisMetric:
    def __init__(self, figsize=(6, 4), save_path='history.png'):
        """Visualization callback for monitoring metrics

        Args:
            figsize (tuple, optional): Figure size. Defaults to (6, 4)
            save_path (str, optional): Path to save the history plot. Defaults to 'history.png'
        """
        self.figsize = figsize
        self.save_path = save_path

    def on_fit_start(self, model: 'KerasModel'):
        """Callback at the beginning of the training

        Args:
            model (KerasModel): The KerasModel instance.
        """
        print('\nMetric plot will be saved to: \n' + os.path.abspath(self.save_path))
        self.metric = model.monitor.replace('val_', '')
        dfhistory = pd.DataFrame(model.history)
        x_bounds = [0, min(10, model.epochs)]
        title = f'best {model.monitor} = ?'
        self.update_graph(model, title=title, x_bounds=x_bounds)

    def on_train_epoch_end(self, model: 'KerasModel'):
        """Callback at the end of each training epoch

        Args:
            model (KerasModel): The KerasModel instance
        """
        pass

    def on_validation_epoch_end(self, model: "KerasModel"):
        """Callback at the end of each validation epoch

        Args:
            model (KerasModel): The KerasModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        x_bounds = [dfhistory['epoch'].min(), min(10 + (n // 10) * 10, model.epochs)]
        title = self.get_title(model)
        self.update_graph(model, title=title, x_bounds=x_bounds)

    def on_fit_end(self, model: "KerasModel"):
        """Callback at the end of the entire training process

        Args:
            model (KerasModel): The KerasModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        title = self.get_title(model)
        self.update_graph(model, title=title)

    def get_best_score(self, model: 'KerasModel'):
        """Get the best score and epoch.

        Args:
            model (KerasModel): The KerasModel instance

        Returns:
            tuple: Best epoch and best score
        """
        dfhistory = pd.DataFrame(model.history)
        arr_scores = dfhistory[model.monitor]
        best_score = np.max(arr_scores) if model.mode == "max" else np.min(arr_scores)
        best_epoch = dfhistory.loc[arr_scores == best_score, 'epoch'].tolist()[0]
        return (best_epoch, best_score)

    def get_title(self, model: 'KerasModel'):
        """Get the title for the plot

        Args:
            model (KerasModel): The KerasModel instance

        Returns:
            str: The title.
        """
        best_epoch, best_score = self.get_best_score(model)
        title = f'best {model.monitor}={best_score:.4f} (@epoch {best_epoch})'
        return title

    def update_graph(self, model: 'KerasModel', title=None, x_bounds=None, y_bounds=None):
        """Update the metric plot.

        Args:
            model (KerasModel): The KerasModel instance
            title (str, optional): Plot title. Defaults to None
            x_bounds (list, optional): x-axis bounds. Defaults to None
            y_bounds (list, optional): y-axis bounds. Defaults to None
        """
        import matplotlib.pyplot as plt
        self.plt = plt
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=self.figsize)
        self.graph_ax.clear()

        dfhistory = pd.DataFrame(model.history)
        epochs = dfhistory['epoch'] if 'epoch' in dfhistory.columns else []

        m1 = "train_" + self.metric
        if m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(epochs, train_metrics, 'bo--', label=m1, clip_on=False)

        m2 = 'val_' + self.metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(epochs, val_metrics, 'co-', label=m2, clip_on=False)

        if self.metric in dfhistory.columns:
            metric_values = dfhistory[self.metric]
            self.graph_ax.plot(epochs, metric_values, 'co-', label=self.metric, clip_on=False)

        self.graph_ax.set_xlabel("epoch")
        self.graph_ax.set_ylabel(self.metric)
        if title:
            self.graph_ax.set_title(title)
            if hasattr(model.EpochRunner, 'progress'):
                model.EpochRunner.progress.comment_tail = title
        if m1 in dfhistory.columns or m2 in dfhistory.columns or self.metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')

        if len(epochs) > 0:
            best_epoch, best_score = self.get_best_score(model)
            self.graph_ax.plot(best_epoch, best_score, 'r*', markersize=15, clip_on=False)

        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        
        self.graph_fig.savefig(self.save_path)
        plt.close()


class EpochCheckpoint:
    def __init__(self, ckpt_dir="weights", save_freq=1, max_ckpt=10):
        """Callback for saving model checkpoints during training

        Args:
            ckpt_dir (str, optional): Directory to save checkpoints. Defaults to "weights"
            save_freq (int, optional): Save frequency (in epochs). Defaults to 1
            max_ckpt (int, optional): Maximum number of checkpoints to keep. Defaults to 10
        """
        self.__dict__.update(locals())
        self.ckpt_idx = 0

    def on_fit_start(self, model: 'KerasModel'):
        """Callback at the beginning of the training

        Args:
            model (KerasModel): The KerasModel instance
        """
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_list = ['' for i in range(self.max_ckpt)]

    def on_train_epoch_end(self, model: 'KerasModel'):
        """Callback at the end of each training epoch

        Args:
            model (KerasModel): The KerasModel instance
        """
        pass

    def on_validation_epoch_end(self, model: "KerasModel"):
        """Callback at the end of each validation epoch

        Args:
            model (KerasModel): The KerasModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        epoch = dfhistory['epoch'].iloc[-1]
        if epoch > 0 and epoch % self.save_freq == 0:
            ckpt_path = os.path.join(self.ckpt_dir, f'checkpoint_epoch{epoch}.pt')
            # Save the model state directly without accelerator
            torch.save({
                'model_state_dict': model.net.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'epoch': epoch,
                'history': model.history
            }, ckpt_path)

            if self.ckpt_list[self.ckpt_idx] != '' and os.path.exists(self.ckpt_list[self.ckpt_idx]):
                os.remove(self.ckpt_list[self.ckpt_idx])
            self.ckpt_list[self.ckpt_idx] = ckpt_path
            self.ckpt_idx = (self.ckpt_idx + 1) % self.max_ckpt

    def on_fit_end(self, model: "KerasModel"):
        """Callback at the end of the entire training process

        Args:
            model (KerasModel): The KerasModel instance
        """
        pass


# Dynamic SMILES Random Callback
class DynamicRandomCallback:
    """
    A callback that updates the DynamicSmilesRandom augmentation probability
    at the beginning of each training epoch.
    
    Args:
        dynamic_random_augmentation: The DynamicSmilesRandom instance to update
        dataloader: The training dataloader
    """
    def __init__(self, dynamic_random_augmentation, dataloader=None):
        super().__init__()
        self.dynamic_random = dynamic_random_augmentation
        self.dataloader = dataloader
        
    def on_train_epoch_end(self, model):
        """Called at the beginning of each training epoch."""
        # Get the current epoch from the model history
        if hasattr(model, 'history') and 'epoch' in model.history:
            current_epoch = model.history['epoch'][-1]
        else:
            # Default to incrementing internal counter if history not available
            if not hasattr(self, '_epoch_counter'):
                self._epoch_counter = 0
            current_epoch = self._epoch_counter
            self._epoch_counter += 1
        
        # Update the randomization probability
        new_prob = self.dynamic_random.update_epoch(current_epoch)
        
        # Print the update
        print(f"Epoch {current_epoch}: SMILES randomization probability set to {new_prob:.4f}")


# Random SMILES Callback
class RandomSmilesCallback:
    """
    A callback that resets the SingleRandomSmilesDataset at the beginning of each epoch
    to select new random SMILES variants for each molecule.
    
    This ensures that different randomized SMILES are used in each epoch.
    """
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader
        
    def on_train_epoch_end(self, model):
        """Called at the beginning of each training epoch."""
        # Check if dataset has the reset_for_new_epoch method
        if hasattr(self.dataloader.dataset, 'reset_for_new_epoch'):
            print("Selecting new random SMILES variants for this epoch...")
            self.dataloader.dataset.reset_for_new_epoch()
        # If using random_split with indices, need to access the underlying dataset
        elif hasattr(self.dataloader.dataset, 'dataset') and hasattr(self.dataloader.dataset.dataset, 'reset_for_new_epoch'):
            print("Selecting new random SMILES variants for this epoch...")
            self.dataloader.dataset.dataset.reset_for_new_epoch()