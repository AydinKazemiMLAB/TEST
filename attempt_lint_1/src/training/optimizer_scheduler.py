import torch
import torch.optim as optim
import math

class NoamOptimizer:
    """
    Manages the Adam optimizer and the custom learning rate schedule (Noam schedule).
    lrate = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
    """
    def __init__(self, model_params, d_model, warmup_steps, beta1=0.9, beta2=0.98, epsilon=1e-9, gradient_clip_norm=None):
        self.optimizer = optim.Adam(model_params, lr=0, betas=(beta1, beta2), eps=epsilon)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.gradient_clip_norm = gradient_clip_norm

    def _get_lr(self):
        """Calculates the current learning rate based on the Noam schedule."""
        arg1 = self.step_num ** -0.5
        arg2 = self.step_num * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * min(arg1, arg2)

    def step(self):
        """Performs an optimization step and updates the learning rate."""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.gradient_clip_norm)
            
        self.optimizer.step()

    def zero_grad(self):
        """Clears the gradients of all optimized torch.Tensors."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Returns a dictionary containing the current state of the optimizer."""
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_num': self.step_num,
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer's state."""
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.step_num = state_dict['step_num']
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']