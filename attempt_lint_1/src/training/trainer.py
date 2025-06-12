import torch
import torch.nn as nn
import time
from utils.logger import Logger
from utils.checkpoint_manager import CheckpointManager
from evaluation.evaluator import Evaluator
from data.dataloader import TranslationDataLoader # For type hinting

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, vocab_size=None, ignore_index=None):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        if self.smoothing > 0 and (self.vocab_size is None or self.ignore_index is None):
            raise ValueError("vocab_size and ignore_index must be provided for label smoothing.")

    def forward(self, logits, target):
        """
        Args:
            logits (torch.Tensor): Predicted logits (batch_size, seq_len, vocab_size)
            target (torch.Tensor): True target IDs (batch_size, seq_len)
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Reshape logits and target for F.log_softmax and NLLLoss
        logits = logits.view(-1, logits.size(-1)) # (batch_size * seq_len, vocab_size)
        target = target.view(-1) # (batch_size * seq_len)

        log_probs = nn.functional.log_softmax(logits, dim=-1)

        if self.smoothing > 0:
            # Apply label smoothing
            # Create a smoothed target distribution
            one_hot = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
            smoothed_target = one_hot * (1 - self.smoothing) + \
                              (1 - one_hot) * self.smoothing / (self.vocab_size - 1)
            
            # Calculate loss
            loss = -(smoothed_target * log_probs).sum(dim=-1)
        else:
            # Standard NLLLoss
            loss = nn.functional.nll_loss(log_probs, target, reduction='none')

        # Mask out loss for padding tokens
        if self.ignore_index is not None:
            non_pad_mask = (target != self.ignore_index)
            loss = loss.masked_select(non_pad_mask)

        return loss.mean()


class Trainer:
    """
    Orchestrates the training loop, including forward pass, loss calculation,
    backpropagation, and logging.
    """
    def __init__(self, model, train_dataloader: TranslationDataLoader, val_dataloader: TranslationDataLoader,
                 optimizer_scheduler, criterion, config, logger: Logger, checkpoint_manager: CheckpointManager):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer_scheduler = optimizer_scheduler
        self.criterion = criterion
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.device = config.project.device
        
        self.model.to(self.device)
        self.best_val_metric = float('-inf') if config.evaluation.primary_metric == "BLEU" else float('inf')
        self.start_step = 0

        if config.checkpointing.load_latest_checkpoint:
            latest_checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            if latest_checkpoint:
                self.model.load_state_dict(latest_checkpoint['model_state_dict'])
                self.optimizer_scheduler.load_state_dict(latest_checkpoint['optimizer_state_dict'])
                self.start_step = latest_checkpoint['step']
                self.best_val_metric = latest_checkpoint['best_metric']
                self.logger.info(f"Resumed training from step {self.start_step} with best metric {self.best_val_metric:.4f}")
            else:
                self.logger.info("No latest checkpoint found, starting training from scratch.")
        elif config.checkpointing.load_checkpoint_path:
            specific_checkpoint = self.checkpoint_manager.load_checkpoint(config.checkpointing.load_checkpoint_path)
            if specific_checkpoint:
                self.model.load_state_dict(specific_checkpoint['model_state_dict'])
                self.optimizer_scheduler.load_state_dict(specific_checkpoint['optimizer_state_dict'])
                self.start_step = specific_checkpoint['step']
                self.best_val_metric = specific_checkpoint['best_metric']
                self.logger.info(f"Loaded checkpoint from {config.checkpointing.load_checkpoint_path}, resuming from step {self.start_step}")
            else:
                self.logger.warning(f"Checkpoint not found at {config.checkpointing.load_checkpoint_path}, starting training from scratch.")


    def _compute_loss(self, logits, targets):
        """
        Calculates cross-entropy loss with label smoothing.
        """
        # Shift target sequence for loss calculation:
        # logits are predictions for tgt[:, 1:]
        # targets are actual tokens at tgt[:, 1:]
        # Example: tgt = <sos> A B C <eos>
        # input to decoder = <sos> A B C
        # target for loss = A B C <eos>
        
        # Flatten the logits and targets for loss calculation
        # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # targets: (batch_size, seq_len) -> (batch_size * seq_len)
        
        # The criterion (LabelSmoothingLoss) already handles the reshaping and ignore_index.
        return self.criterion(logits, targets)

    def train_epoch(self, current_step):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (src_ids, tgt_ids, src_mask, tgt_mask) in enumerate(self.train_dataloader):
            if current_step >= self.config.training.max_steps:
                self.logger.info(f"Reached max_steps ({self.config.training.max_steps}), stopping training.")
                return False # Signal to stop training

            self.optimizer_scheduler.zero_grad()

            # Shift target for decoder input (teacher forcing)
            # Decoder input: <sos> A B C
            # Target for loss: A B C <eos>
            decoder_input = tgt_ids[:, :-1]
            target_for_loss = tgt_ids[:, 1:]
            
            # Adjust target mask for decoder input
            decoder_mask = tgt_mask[:, :, :-1, :-1] # Remove last token from sequence dimension

            # Forward pass
            logits = self.model(src_ids, decoder_input, src_mask, decoder_mask)
            
            # Calculate loss
            loss = self._compute_loss(logits, target_for_loss)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer_scheduler.step()

            total_loss += loss.item()
            current_step += 1

            if current_step % self.config.training.log_interval_steps == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Step: {current_step}/{self.config.training.max_steps}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {self.optimizer_scheduler._get_lr():.6f}, "
                    f"Time/step: {elapsed_time / self.config.training.log_interval_steps:.4f}s"
                )
                start_time = time.time() # Reset timer

            if current_step % self.config.training.validate_interval_steps == 0:
                self.logger.info(f"--- Running validation at step {current_step} ---")
                evaluator = Evaluator(self.model, self.val_dataloader, self.config, self.logger, 
                                      self.train_dataloader.dataset.src_tokenizer, 
                                      self.train_dataloader.dataset.tgt_tokenizer)
                val_metrics = evaluator.evaluate()
                
                val_loss = val_metrics.get("Perplexity", float('inf')) # Use perplexity as validation loss
                val_bleu = val_metrics.get("BLEU", 0.0)

                self.logger.info(f"Validation Loss: {val_loss:.4f}, BLEU: {val_bleu:.2f}")

                # Save checkpoint
                is_best = False
                if self.config.evaluation.primary_metric == "BLEU":
                    if val_bleu > self.best_val_metric:
                        self.best_val_metric = val_bleu
                        is_best = True
                elif self.config.evaluation.primary_metric == "Perplexity":
                    if val_loss < self.best_val_metric:
                        self.best_val_metric = val_loss
                        is_best = True
                
                if self.config.training.save_best_model and is_best:
                    self.logger.info(f"New best model found! Saving checkpoint at step {current_step}.")
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer_scheduler, current_step, val_loss, self.best_val_metric, is_best=True
                    )
                elif not self.config.training.save_best_model:
                     self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer_scheduler, current_step, val_loss, self.best_val_metric, is_best=False
                    )
                self.model.train() # Set model back to train mode
        
        return True # Signal to continue training for next epoch

    def run_training(self):
        """
        Manages the entire training process for a specified number of steps/epochs.
        """
        self.logger.info("Starting training...")
        current_step = self.start_step
        epoch = 0
        
        while current_step < self.config.training.max_steps:
            epoch += 1
            self.logger.info(f"--- Epoch {epoch} ---")
            should_continue = self.train_epoch(current_step)
            if not should_continue:
                break
            current_step = self.optimizer_scheduler.step_num # Update current_step from optimizer_scheduler
            
        self.logger.info("Training finished.")