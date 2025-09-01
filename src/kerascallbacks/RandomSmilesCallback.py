"""
Callback to reset the SingleRandomSmilesDataset at the beginning of each epoch.
"""
import torch

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