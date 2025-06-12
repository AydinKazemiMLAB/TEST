import os
import sys
import sentencepiece as spm

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.config import Config
from data.tokenizer import Tokenizer
from data.vocabulary import Vocabulary

def preprocess_data(config_path):
    """
    Preprocesses raw text data: trains tokenizers and creates vocabulary files.
    """
    config = Config(config_path)
    data_config = config.data
    tokenizer_config = data_config.tokenizer

    output_dir = os.path.join(config.project.output_dir, "data", "processed", data_config.dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Combine source and target data for shared vocabulary training
    if tokenizer_config.shared_vocabulary:
        combined_corpus_path = os.path.join(output_dir, "combined_corpus.txt")
        with open(combined_corpus_path, 'w', encoding='utf-8') as outfile:
            for file_key in ['train_src_file', 'train_tgt_file', 'val_src_file', 'val_tgt_file', 'test_src_file', 'test_tgt_file']:
                file_path = os.path.join(data_config.data_root_dir, getattr(data_config, file_key))
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                else:
                    print(f"Warning: {file_path} not found. Skipping for combined corpus.")
        
        print(f"Combined corpus saved to: {combined_corpus_path}")

        # Train a single tokenizer for both source and target
        model_prefix = os.path.join(output_dir, f"{tokenizer_config.type.lower()}_shared")
        vocab_size = tokenizer_config.num_merges if tokenizer_config.type == "BPE" else tokenizer_config.wordpiece_vocab_size
        
        # Initialize a dummy tokenizer to get special token IDs
        dummy_tokenizer = Tokenizer(special_tokens=tokenizer_config.special_tokens)
        
        cmd = (
            f"--input={combined_corpus_path} --model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} --model_type={tokenizer_config.type.lower()} "
            f"--pad_id={dummy_tokenizer.get_pad_id()} --bos_id={dummy_tokenizer.get_sos_id()} "
            f"--eos_id={dummy_tokenizer.get_eos_id()} --unk_id={dummy_tokenizer.get_unk_id()} "
            f"--control_symbols={','.join([t for t in tokenizer_config.special_tokens if t not in ['<pad>', '<unk>']])}"
        )
        print(f"Training shared tokenizer with command: {cmd}")
        spm.SentencePieceTrainer.train(cmd)
        
        # Create and save our custom Vocabulary object
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(f"{model_prefix}.model")
        vocab = Vocabulary(special_tokens=tokenizer_config.special_tokens)
        for i in range(sp_model.get_piece_size()):
            vocab.add_token(sp_model.id_to_piece(i))
        vocab.save(os.path.join(output_dir, "vocab.json"))
        
        print(f"Shared tokenizer model saved to {model_prefix}.model")
        print(f"Shared vocabulary saved to {os.path.join(output_dir, 'vocab.json')}")

    else: # Separate vocabularies for source and target
        # Train source tokenizer
        src_corpus_path = os.path.join(data_config.data_root_dir, data_config.train_src_file)
        src_model_prefix = os.path.join(output_dir, f"{tokenizer_config.type.lower()}_en")
        src_vocab_size = tokenizer_config.wordpiece_vocab_size # Assuming WordPiece for EN-FR
        
        dummy_tokenizer = Tokenizer(special_tokens=tokenizer_config.special_tokens)

        cmd_src = (
            f"--input={src_corpus_path} --model_prefix={src_model_prefix} "
            f"--vocab_size={src_vocab_size} --model_type={tokenizer_config.type.lower()} "
            f"--pad_id={dummy_tokenizer.get_pad_id()} --bos_id={dummy_tokenizer.get_sos_id()} "
            f"--eos_id={dummy_tokenizer.get_eos_id()} --unk_id={dummy_tokenizer.get_unk_id()} "
            f"--control_symbols={','.join([t for t in tokenizer_config.special_tokens if t not in ['<pad>', '<unk>']])}"
        )
        print(f"Training source tokenizer with command: {cmd_src}")
        spm.SentencePieceTrainer.train(cmd_src)
        
        sp_model_src = spm.SentencePieceProcessor()
        sp_model_src.load(f"{src_model_prefix}.model")
        vocab_src = Vocabulary(special_tokens=tokenizer_config.special_tokens)
        for i in range(sp_model_src.get_piece_size()):
            vocab_src.add_token(sp_model_src.id_to_piece(i))
        vocab_src.save(os.path.join(output_dir, "vocab_en.json"))
        print(f"Source tokenizer model saved to {src_model_prefix}.model")
        print(f"Source vocabulary saved to {os.path.join(output_dir, 'vocab_en.json')}")

        # Train target tokenizer
        tgt_corpus_path = os.path.join(data_config.data_root_dir, data_config.train_tgt_file)
        tgt_model_prefix = os.path.join(output_dir, f"{tokenizer_config.type.lower()}_fr")
        tgt_vocab_size = tokenizer_config.wordpiece_vocab_size # Assuming WordPiece for EN-FR

        cmd_tgt = (
            f"--input={tgt_corpus_path} --model_prefix={tgt_model_prefix} "
            f"--vocab_size={tgt_vocab_size} --model_type={tokenizer_config.type.lower()} "
            f"--pad_id={dummy_tokenizer.get_pad_id()} --bos_id={dummy_tokenizer.get_sos_id()} "
            f"--eos_id={dummy_tokenizer.get_eos_id()} --unk_id={dummy_tokenizer.get_unk_id()} "
            f"--control_symbols={','.join([t for t in tokenizer_config.special_tokens if t not in ['<pad>', '<unk>']])}"
        )
        print(f"Training target tokenizer with command: {cmd_tgt}")
        spm.SentencePieceTrainer.train(cmd_tgt)

        sp_model_tgt = spm.SentencePieceProcessor()
        sp_model_tgt.load(f"{tgt_model_prefix}.model")
        vocab_tgt = Vocabulary(special_tokens=tokenizer_config.special_tokens)
        for i in range(sp_model_tgt.get_piece_size()):
            vocab_tgt.add_token(sp_model_tgt.id_to_piece(i))
        vocab_tgt.save(os.path.join(output_dir, "vocab_fr.json"))
        print(f"Target tokenizer model saved to {tgt_model_prefix}.model")
        print(f"Target vocabulary saved to {os.path.join(output_dir, 'vocab_fr.json')}")

    print("Data preprocessing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess data for Transformer NMT.")
    parser.add_argument("--config", type=str, default="configs/base_model.yaml",
                        help="Path to the configuration YAML file for data preprocessing.")
    args = parser.parse_args()
    
    preprocess_data(args.config)