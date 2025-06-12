import torch
from torch.utils.data import Dataset
import os

class TranslationDataset(Dataset):
    """
    Represents the dataset, loading sentence pairs, applying tokenization,
    and numericalizing them.
    """
    def __init__(self, data_path_src, data_path_tgt, src_tokenizer, tgt_tokenizer, max_seq_len):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len
        self.data = self._load_data(data_path_src, data_path_tgt)

    def _load_data(self, src_file, tgt_file):
        """
        Loads and tokenizes sentence pairs.
        Assumes files are line-aligned.
        """
        data_pairs = []
        with open(src_file, 'r', encoding='utf-8') as f_src, \
             open(tgt_file, 'r', encoding='utf-8') as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_ids = self.src_tokenizer.encode(src_line.strip())
                tgt_ids = self.tgt_tokenizer.encode(tgt_line.strip())

                # Truncate if longer than max_seq_len
                if len(src_ids) > self.max_seq_len:
                    src_ids = src_ids[:self.max_seq_len - 1] + [self.src_tokenizer.get_eos_id()]
                if len(tgt_ids) > self.max_seq_len:
                    tgt_ids = tgt_ids[:self.max_seq_len - 1] + [self.tgt_tokenizer.get_eos_id()]
                
                data_pairs.append((src_ids, tgt_ids))
        return data_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_len": len(src_ids),
            "tgt_len": len(tgt_ids)
        }