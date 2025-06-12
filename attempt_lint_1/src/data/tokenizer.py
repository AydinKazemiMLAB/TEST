import sentencepiece as spm
import os
from data.vocabulary import Vocabulary

class Tokenizer:
    """
    Manages tokenization processes using SentencePiece (for BPE/WordPiece).
    """
    def __init__(self, model_path=None, vocab_path=None, special_tokens=None):
        self.sp_model = spm.SentencePieceProcessor()
        self.vocab = None
        self.special_tokens = special_tokens if special_tokens else {
            "<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3
        }

        if model_path and os.path.exists(model_path):
            self.sp_model.load(model_path)
            # If a SentencePiece model is loaded, its vocabulary is used
            # We can optionally build a Vocabulary object from it for consistency
            self.vocab = Vocabulary(special_tokens=self.special_tokens)
            for i in range(self.sp_model.get_piece_size()):
                token = self.sp_model.id_to_piece(i)
                # Ensure special tokens are mapped correctly if SentencePiece assigns different IDs
                if token in self.special_tokens and self.vocab.token_to_id(token) != i:
                    # This case is tricky: SentencePiece might assign different IDs to special tokens.
                    # For simplicity, we assume SentencePiece is trained with these special tokens
                    # and their IDs align, or we re-map them.
                    # For now, we'll just add all pieces.
                    pass
                self.vocab.add_token(token)
        elif vocab_path and os.path.exists(vocab_path):
            self.vocab = Vocabulary.load(vocab_path)
        else:
            self.vocab = Vocabulary(special_tokens=self.special_tokens)

    def train(self, corpus_path, model_prefix, vocab_size, model_type="bpe"):
        """
        Trains a new SentencePiece tokenizer model.
        Args:
            corpus_path (str): Path to the raw text corpus.
            model_prefix (str): Prefix for the output model files (e.g., 'my_tokenizer').
            vocab_size (int): Desired vocabulary size.
            model_type (str): 'bpe' or 'unigram' (for WordPiece).
        """
        # SentencePiece command line options
        # --unk_id, --pad_id, --bos_id, --eos_id are important for special tokens
        # Ensure these IDs match our Vocabulary's special token IDs
        cmd = (
            f"--input={corpus_path} --model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} --model_type={model_type} "
            f"--pad_id={self.special_tokens['<pad>']} --bos_id={self.special_tokens['<sos>']} "
            f"--eos_id={self.special_tokens['<eos>']} --unk_id={self.special_tokens['<unk>']} "
            f"--control_symbols={','.join([t for t in self.special_tokens if t not in ['<pad>', '<unk>']])}"
        )
        spm.SentencePieceTrainer.train(cmd)
        self.sp_model.load(f"{model_prefix}.model")
        
        # Rebuild vocabulary from the trained SentencePiece model
        self.vocab = Vocabulary(special_tokens=self.special_tokens)
        for i in range(self.sp_model.get_piece_size()):
            self.vocab.add_token(self.sp_model.id_to_piece(i))
        self.vocab.save(f"{model_prefix}.vocab.json") # Save our custom Vocabulary object

    def encode(self, text, add_special_tokens=True):
        """
        Converts text to a list of token IDs.
        """
        if not self.sp_model.model_is_loaded():
            raise RuntimeError("SentencePiece model not loaded. Train or load a model first.")
        
        if add_special_tokens:
            # SentencePiece handles BOS/EOS if configured during training
            # Otherwise, manually add them using vocab IDs
            ids = self.sp_model.encode_as_ids(text)
            return [self.vocab.sos_id] + ids + [self.vocab.eos_id]
        else:
            return self.sp_model.encode_as_ids(text)

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Converts token IDs back to text.
        """
        if not self.sp_model.model_is_loaded():
            raise RuntimeError("SentencePiece model not loaded. Train or load a model first.")
        
        if skip_special_tokens:
            # Filter out special tokens before decoding
            filtered_ids = [
                _id for _id in token_ids 
                if _id not in [self.vocab.pad_id, self.vocab.sos_id, self.vocab.eos_id, self.vocab.unk_id]
            ]
            return self.sp_model.decode_ids(filtered_ids)
        else:
            return self.sp_model.decode_ids(token_ids)

    def get_vocab_size(self):
        if self.vocab:
            return len(self.vocab)
        if self.sp_model.model_is_loaded():
            return self.sp_model.get_piece_size()
        return 0

    def get_pad_id(self):
        return self.vocab.pad_id if self.vocab else self.special_tokens["<pad>"]

    def get_sos_id(self):
        return self.vocab.sos_id if self.vocab else self.special_tokens["<sos>"]

    def get_eos_id(self):
        return self.vocab.eos_id if self.vocab else self.special_tokens["<eos>"]

    def get_unk_id(self):
        return self.vocab.unk_id if self.vocab else self.special_tokens["<unk>"]