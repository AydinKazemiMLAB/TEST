import torch
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.config import Config
from utils.logger import Logger
from utils.checkpoint_manager import CheckpointManager
from data.tokenizer import Tokenizer
from data.dataset import TranslationDataset
from data.dataloader import TranslationDataLoader
from models.transformer import Transformer
from training.optimizer_scheduler import NoamOptimizer
from training.trainer import Trainer, LabelSmoothingLoss
from evaluation.evaluator import Evaluator

def main(config_path):
    # 1. Load Configuration
    config = Config(config_path)

    # 2. Initialize Logging
    logger = Logger(
        log_dir=os.path.join(config.project.output_dir, config.logging.log_dir),
        log_file=config.logging.log_file,
        level=config.logging.level
    )
    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Running on device: {config.project.device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.project.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.project.seed)
    
    # 3. Initialize Checkpoint Manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.join(config.project.output_dir, config.checkpointing.checkpoint_dir)
    )

    # 4. Data Preparation
    data_root = config.data.data_root_dir
    
    # Initialize Tokenizers and Vocabulary
    special_tokens = config.data.tokenizer.special_tokens
    
    # Source Tokenizer
    src_tokenizer = Tokenizer(
        model_path=os.path.join(data_root, config.data.tokenizer.src_tokenizer_model_path),
        vocab_path=os.path.join(data_root, config.data.tokenizer.src_vocab_path),
        special_tokens=special_tokens
    )
    # Target Tokenizer
    if config.data.tokenizer.shared_vocabulary:
        tgt_tokenizer = src_tokenizer # Use the same tokenizer if vocabulary is shared
    else:
        tgt_tokenizer = Tokenizer(
            model_path=os.path.join(data_root, config.data.tokenizer.tgt_tokenizer_model_path),
            vocab_path=os.path.join(data_root, config.data.tokenizer.tgt_vocab_path),
            special_tokens=special_tokens
        )

    if src_tokenizer.get_vocab_size() == 0 or tgt_tokenizer.get_vocab_size() == 0:
        logger.error("Tokenizer models or vocabularies not loaded. Please run preprocess_data.py first.")
        sys.exit(1)

    logger.info(f"Source vocabulary size: {src_tokenizer.get_vocab_size()}")
    logger.info(f"Target vocabulary size: {tgt_tokenizer.get_vocab_size()}")

    # Initialize Datasets
    train_dataset = TranslationDataset(
        data_path_src=os.path.join(data_root, config.data.train_src_file),
        data_path_tgt=os.path.join(data_root, config.data.train_tgt_file),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_seq_len=config.data.max_seq_len
    )
    val_dataset = TranslationDataset(
        data_path_src=os.path.join(data_root, config.data.val_src_file),
        data_path_tgt=os.path.join(data_root, config.data.val_tgt_file),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_seq_len=config.data.max_seq_len
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Initialize DataLoaders
    train_dataloader = TranslationDataLoader(
        dataset=train_dataset,
        max_tokens_per_batch=config.data.max_tokens_per_batch,
        pad_idx=src_tokenizer.get_pad_id(), # Pad ID is same for src/tgt if shared vocab, otherwise use src_tokenizer's
        device=config.project.device,
        shuffle=config.data.shuffle_train_data,
        eval_mode=False
    )
    val_dataloader = TranslationDataLoader(
        dataset=val_dataset,
        max_tokens_per_batch=config.data.eval_batch_size, # For eval, this is batch_size
        pad_idx=tgt_tokenizer.get_pad_id(),
        device=config.project.device,
        shuffle=False,
        eval_mode=True
    )
    logger.info(f"Train DataLoader initialized with max_tokens_per_batch: {config.data.max_tokens_per_batch}")
    logger.info(f"Validation DataLoader initialized with eval_batch_size: {config.data.eval_batch_size}")

    # 5. Model Initialization
    model = Transformer(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        d_model=config.model.d_model,
        N=config.model.num_encoder_layers,
        h=config.model.num_attention_heads,
        d_ff=config.model.d_ff,
        dropout_rate=config.model.dropout_rate,
        max_seq_len=config.data.max_seq_len
    )
    logger.info(f"Transformer model initialized with d_model={config.model.d_model}, N={config.model.num_encoder_layers}, h={config.model.num_attention_heads}")

    # Initialize Optimizer and Learning Rate Scheduler
    optimizer_scheduler = NoamOptimizer(
        model_params=model.parameters(),
        d_model=config.model.d_model,
        warmup_steps=config.lr_scheduler.warmup_steps,
        beta1=config.optimizer.adam_beta1,
        beta2=config.optimizer.adam_beta2,
        epsilon=config.optimizer.adam_epsilon,
        gradient_clip_norm=config.training.gradient_clip_norm
    )
    logger.info(f"Optimizer (Adam) and LR Scheduler (Noam) initialized with warmup_steps={config.lr_scheduler.warmup_steps}")

    # Initialize Loss Criterion
    criterion = LabelSmoothingLoss(
        smoothing=config.regularization.label_smoothing_epsilon,
        vocab_size=tgt_tokenizer.get_vocab_size(),
        ignore_index=tgt_tokenizer.get_pad_id()
    )
    logger.info(f"Loss criterion (Label Smoothing) initialized with epsilon={config.regularization.label_smoothing_epsilon}")

    # 6. Training Process
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer_scheduler=optimizer_scheduler,
        criterion=criterion,
        config=config,
        logger=logger,
        checkpoint_manager=checkpoint_manager
    )
    trainer.run_training()

    # 7. Final Evaluation (Optional, typically done after training is complete)
    logger.info("--- Running final evaluation ---")
    
    # Load best model or averaged model for final evaluation
    if config.evaluation.model_averaging.enabled:
        try:
            averaged_state_dict = checkpoint_manager.load_and_average_checkpoints(
                config.evaluation.model_averaging.num_checkpoints, config.project.device
            )
            model.load_state_dict(averaged_state_dict)
            logger.info(f"Loaded averaged model from last {config.evaluation.model_averaging.num_checkpoints} checkpoints.")
        except Exception as e:
            logger.error(f"Failed to load and average checkpoints: {e}. Using current model state.")
            # Fallback to loading the best_model.pt if averaging fails or no checkpoints
            best_model_path = os.path.join(checkpoint_manager.checkpoint_dir, "best_model.pt")
            best_checkpoint = checkpoint_manager.load_checkpoint(best_model_path)
            if best_checkpoint:
                model.load_state_dict(best_checkpoint['model_state_dict'])
                logger.info("Loaded best_model.pt for final evaluation.")
            else:
                logger.warning("No best_model.pt found. Using the model's current state for final evaluation.")
    else:
        best_model_path = os.path.join(checkpoint_manager.checkpoint_dir, "best_model.pt")
        best_checkpoint = checkpoint_manager.load_checkpoint(best_model_path)
        if best_checkpoint:
            model.load_state_dict(best_checkpoint['model_state_dict'])
            logger.info("Loaded best_model.pt for final evaluation.")
        else:
            logger.warning("No best_model.pt found. Using the model's current state for final evaluation.")

    # Prepare test dataset and dataloader
    test_dataset = TranslationDataset(
        data_path_src=os.path.join(data_root, config.data.test_src_file),
        data_path_tgt=os.path.join(data_root, config.data.test_tgt_file),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_seq_len=config.data.max_seq_len
    )
    test_dataloader = TranslationDataLoader(
        dataset=test_dataset,
        max_tokens_per_batch=config.data.eval_batch_size,
        pad_idx=tgt_tokenizer.get_pad_id(),
        device=config.project.device,
        shuffle=False,
        eval_mode=True
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")

    evaluator = Evaluator(model, test_dataloader, config, logger, src_tokenizer, tgt_tokenizer)
    final_metrics = evaluator.evaluate()
    
    logger.info("--- Final Evaluation Results ---")
    for metric_name, value in final_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    # Example translation
    sample_sentence = "The quick brown fox jumps over the lazy dog."
    logger.info(f"\nTranslating sample sentence: '{sample_sentence}'")
    translated_sentence = evaluator.translate(sample_sentence)
    logger.info(f"Translated: '{translated_sentence}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Transformer NMT training and evaluation.")
    parser.add_argument("--config", type=str, default="configs/base_model.yaml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()
    
    main(args.config)