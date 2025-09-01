"""
Callback to update the DynamicSmilesRandom augmentation probability at each epoch.
"""


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
        model.accelerator.print(f"Epoch {current_epoch}: SMILES randomization probability set to {new_prob:.4f}")