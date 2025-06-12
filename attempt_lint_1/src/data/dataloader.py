import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np

class BatchSamplerByLength(Sampler):
    """
    Batches indices by approximate sequence length to minimize padding.
    """
    def __init__(self, dataset, max_tokens_per_batch, pad_idx, shuffle=True):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pad_idx = pad_idx
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
        # Sort indices by target length (or source length, or sum of lengths)
        # This is a heuristic for approximate length batching.
        # A more sophisticated approach would involve bucketing.
        self.indices.sort(key=lambda i: dataset[i]["tgt_len"])

    def __iter__(self):
        if self.shuffle:
            # Shuffle within buckets or shuffle the sorted list and then re-sort small chunks
            # For simplicity, we'll just shuffle the pre-sorted indices for now.
            # A proper bucketing sampler would be more complex.
            np.random.shuffle(self.indices) 
            self.indices.sort(key=lambda i: self.dataset[i]["tgt_len"]) # Re-sort to maintain length locality

        batch = []
        current_max_len = 0
        for idx in self.indices:
            src_len = self.dataset[idx]["src_len"]
            tgt_len = self.dataset[idx]["tgt_len"]
            
            # Update current_max_len for the batch
            current_max_len = max(current_max_len, src_len, tgt_len)
            
            # Estimate batch size based on current max length
            # This is a simplified heuristic. A more accurate one would track actual tokens.
            estimated_batch_size = self.max_tokens_per_batch // (current_max_len * 2) # *2 for src+tgt
            
            if len(batch) >= estimated_batch_size and batch:
                yield batch
                batch = []
                current_max_len = 0 # Reset max length for new batch
            
            batch.append(idx)
        
        if batch:
            yield batch

    def __len__(self):
        # This is an approximation. Actual number of batches depends on dynamic batching.
        # For a more precise count, one would need to simulate the batching process.
        return len(list(self.__iter__()))


def collate_fn(batch, pad_idx, device):
    """
    Pads sequences within a batch to the maximum length of that batch.
    Generates padding masks and a look-ahead mask for the target.
    """
    src_ids = [item["src_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]

    # Pad source sequences
    max_src_len = max(len(s) for s in src_ids)
    padded_src_ids = torch.full((len(src_ids), max_src_len), pad_idx, dtype=torch.long, device=device)
    for i, s in enumerate(src_ids):
        padded_src_ids[i, :len(s)] = s.to(device)

    # Pad target sequences
    max_tgt_len = max(len(t) for t in tgt_ids)
    padded_tgt_ids = torch.full((len(tgt_ids), max_tgt_len), pad_idx, dtype=torch.long, device=device)
    for i, t in enumerate(tgt_ids):
        padded_tgt_ids[i, :len(t)] = t.to(device)

    # Create source padding mask (1 where not pad, 0 where pad)
    src_mask = (padded_src_ids != pad_idx).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len)

    # Create target padding mask (1 where not pad, 0 where pad)
    tgt_pad_mask = (padded_tgt_ids != pad_idx).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len)
    
    # Create look-ahead mask for target (upper triangle is 0)
    seq_len = padded_tgt_ids.size(1)
    look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
    
    # Combine target padding mask and look-ahead mask
    tgt_mask = tgt_pad_mask & ~look_ahead_mask # (batch_size, 1, seq_len, seq_len)
    
    return padded_src_ids, padded_tgt_ids, src_mask, tgt_mask


class TranslationDataLoader(DataLoader):
    """
    Batches and pads sequences for efficient training, handling approximate sequence length batching.
    """
    def __init__(self, dataset, max_tokens_per_batch, pad_idx, device, shuffle=True, eval_mode=False):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pad_idx = pad_idx
        self.device = device
        self.shuffle = shuffle
        self.eval_mode = eval_mode

        if eval_mode:
            # For evaluation, use a fixed batch size (eval_batch_size from config)
            # and simple sequential sampler.
            super().__init__(
                dataset,
                batch_size=max_tokens_per_batch, # In eval_mode, max_tokens_per_batch is actually eval_batch_size
                shuffle=False, # No shuffling for evaluation
                collate_fn=lambda b: collate_fn(b, pad_idx, device)
            )
        else:
            # For training, use the custom BatchSamplerByLength for approximate length batching
            batch_sampler = BatchSamplerByLength(
                dataset,
                max_tokens_per_batch=max_tokens_per_batch,
                pad_idx=pad_idx,
                shuffle=shuffle
            )
            super().__init__(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=lambda b: collate_fn(b, pad_idx, device)
            )