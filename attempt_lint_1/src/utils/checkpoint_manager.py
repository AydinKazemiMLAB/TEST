import torch
import os
import glob

class CheckpointManager:
    """
    Manages saving and loading model checkpoints during training and for inference.
    """
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer_scheduler, step, loss, best_metric, is_best=False):
        """
        Saves the current state of the model and optimizer.
        Args:
            model (torch.nn.Module): The model to save.
            optimizer_scheduler: The optimizer/scheduler object.
            step (int): Current training step.
            loss (float): Current training loss.
            best_metric (float): The best validation metric achieved so far.
            is_best (bool): True if this is the best model found so far.
        """
        checkpoint_name = f"checkpoint_step_{step:06d}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        state = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_scheduler.state_dict(),
            'loss': loss,
            'best_metric': best_metric,
            'timestamp': torch.tensor(os.path.getmtime(checkpoint_path)) if os.path.exists(checkpoint_path) else torch.tensor(0.0) # Placeholder
        }
        
        torch.save(state, checkpoint_path)
        
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, best_model_path)

        # Manage top K checkpoints (optional, based on config)
        # This logic would typically be in the Trainer or a separate cleanup function
        # For now, we'll assume the Trainer handles keeping top K by deleting old ones.

    def load_checkpoint(self, checkpoint_path):
        """
        Loads a specific checkpoint.
        Args:
            checkpoint_path (str): Full path to the checkpoint file.
        Returns:
            dict: Loaded checkpoint state, or None if not found.
        """
        if not os.path.exists(checkpoint_path):
            return None
        return torch.load(checkpoint_path, map_location='cpu') # Load to CPU first, then move to device

    def load_latest_checkpoint(self):
        """
        Loads the most recent checkpoint based on filename (step number).
        Returns:
            dict: Latest checkpoint state, or None if no checkpoints exist.
        """
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by step number in filename
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
        
        latest_checkpoint_path = checkpoints[-1]
        return self.load_checkpoint(latest_checkpoint_path)

    def load_and_average_checkpoints(self, num_checkpoints, device):
        """
        Loads the last `num_checkpoints` and averages their model weights.
        Args:
            num_checkpoints (int): Number of most recent checkpoints to average.
            device (torch.device): Device to load models onto.
        Returns:
            dict: Averaged model state_dict.
        """
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir} to average.")
        
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]), reverse=True)
        
        selected_checkpoints = checkpoints[:num_checkpoints]
        if not selected_checkpoints:
            raise ValueError(f"Not enough checkpoints found. Expected {num_checkpoints}, found {len(checkpoints)}.")

        self.logger.info(f"Averaging {len(selected_checkpoints)} checkpoints...")
        
        averaged_state_dict = None
        for i, cp_path in enumerate(selected_checkpoints):
            checkpoint = torch.load(cp_path, map_location=device)
            model_state_dict = checkpoint['model_state_dict']

            if averaged_state_dict is None:
                averaged_state_dict = {key: value.clone() for key, value in model_state_dict.items()}
            else:
                for key, value in model_state_dict.items():
                    if key in averaged_state_dict:
                        averaged_state_dict[key] += value
                    else:
                        # Handle cases where some keys might be missing (unlikely for same model)
                        averaged_state_dict[key] = value.clone()
        
        # Divide by the number of checkpoints to get the average
        for key in averaged_state_dict:
            averaged_state_dict[key] /= len(selected_checkpoints)
            
        return averaged_state_dict