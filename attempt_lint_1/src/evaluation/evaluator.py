import torch
import torch.nn.functional as F
from utils.logger import Logger
from evaluation.metrics import Metrics
from data.dataloader import TranslationDataLoader # For type hinting
from data.tokenizer import Tokenizer # For type hinting

class Evaluator:
    """
    Handles model inference, beam search decoding, and evaluation on a test set.
    """
    def __init__(self, model, dataloader: TranslationDataLoader, config, logger: Logger, 
                 src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.logger = logger
        self.metrics = Metrics()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = config.project.device
        self.model.to(self.device)

    def _beam_search_decode(self, enc_output, src_mask, max_output_len, beam_size, length_penalty_alpha):
        """
        Implements beam search decoding.
        Args:
            enc_output (torch.Tensor): Encoder output (1, src_seq_len, d_model)
            src_mask (torch.Tensor): Source padding mask (1, 1, 1, src_seq_len)
            max_output_len (int): Maximum length of the generated sequence.
            beam_size (int): Size of the beam.
            length_penalty_alpha (float): Alpha for length penalty.
        Returns:
            list[int]: List of token IDs for the best hypothesis.
        """
        # enc_output and src_mask are for a single sentence (batch_size=1)
        # Initialize beams with SOS token
        # beams: list of (score, [token_ids])
        beams = [(0.0, [self.tgt_tokenizer.get_sos_id()])] # (log_prob, sequence)

        for _ in range(max_output_len):
            new_beams = []
            for score, tokens in beams:
                if tokens[-1] == self.tgt_tokenizer.get_eos_id():
                    new_beams.append((score, tokens))
                    continue # This beam is finished

                # Prepare decoder input
                decoder_input = torch.tensor(tokens).unsqueeze(0).to(self.device) # (1, current_seq_len)
                
                # Create decoder mask (padding + look-ahead)
                seq_len = decoder_input.size(1)
                tgt_pad_mask = (decoder_input != self.tgt_tokenizer.get_pad_id()).unsqueeze(1).unsqueeze(1)
                look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
                decoder_mask = tgt_pad_mask & ~look_ahead_mask

                # Forward pass through decoder
                # Only need the last token's logits for prediction
                with torch.no_grad():
                    dec_output = self.model.decoder(
                        self.model.tgt_embeddings(decoder_input), 
                        enc_output, 
                        src_mask, 
                        decoder_mask
                    )
                    logits = self.model.output_linear(dec_output[:, -1, :]) # Logits for the next token (1, vocab_size)

                log_probs = F.log_softmax(logits, dim=-1).squeeze(0) # (vocab_size)

                # Get top K candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    next_token_id = top_indices[i].item()
                    next_log_prob = top_log_probs[i].item()
                    new_beams.append((score + next_log_prob, tokens + [next_token_id]))
            
            # Sort all new beams by score and select top beam_size
            # Apply length penalty: score / ((5 + len)**alpha / (5 + 1)**alpha)
            # The paper uses: score / (length_of_sequence ** alpha)
            # A common implementation is (score / (len**alpha))
            # Or (score / (len_norm_factor)) where len_norm_factor = ((5 + len)**alpha / (5 + 1)**alpha)
            # Let's use the simpler one: score / (len**alpha)
            
            # Filter out beams that have already ended if all beams are finished
            # Or, if not all beams are finished, keep finished beams and continue expanding unfinished ones.
            
            # For simplicity, let's sort by score + length_penalty
            # Length penalty from paper: (5 + len)**alpha / (5 + 1)**alpha
            # This means higher alpha penalizes longer sequences more.
            # We want to maximize score, so we divide by penalty.
            
            # Re-evaluate scores with length penalty for sorting
            scored_beams = []
            for s, t in new_beams:
                # Avoid division by zero for length 0 or 1
                current_len = len(t) - 1 # Exclude SOS token
                if current_len == 0: # For the very first token, length penalty is 1
                    len_norm = 1.0
                else:
                    len_norm = ((5 + current_len) ** length_penalty_alpha) / (6 ** length_penalty_alpha)
                
                scored_beams.append((s / len_norm, s, t)) # (normalized_score, raw_score, tokens)

            scored_beams.sort(key=lambda x: x[0], reverse=True)
            beams = [(raw_score, tokens) for _, raw_score, tokens in scored_beams[:beam_size]]

            # Early stopping: if all top beams have ended with EOS, stop
            if all(b[1][-1] == self.tgt_tokenizer.get_eos_id() for b in beams):
                break

        # Return the best sequence (highest raw score among finished beams, or highest normalized score overall)
        # Filter for finished beams first if any, otherwise take the best overall
        finished_beams = [b for b in beams if b[1][-1] == self.tgt_tokenizer.get_eos_id()]
        if finished_beams:
            # Sort finished beams by raw score (or normalized score if preferred)
            finished_beams.sort(key=lambda x: x[0], reverse=True)
            best_sequence = finished_beams[0][1]
        else:
            # If no beam finished, take the best current beam
            best_sequence = beams[0][1]
        
        # Remove SOS and EOS tokens for output
        if best_sequence[0] == self.tgt_tokenizer.get_sos_id():
            best_sequence = best_sequence[1:]
        if best_sequence and best_sequence[-1] == self.tgt_tokenizer.get_eos_id():
            best_sequence = best_sequence[:-1]

        return best_sequence

    def evaluate(self):
        """
        Runs evaluation on the dataset and reports metrics.
        """
        self.model.eval()
        total_loss = 0
        all_hypotheses = []
        all_references = []
        
        # For perplexity calculation
        total_tokens = 0
        total_nll_loss = 0

        with torch.no_grad():
            for batch_idx, (src_ids, tgt_ids, src_mask, tgt_mask) in enumerate(self.dataloader):
                # Move to device
                src_ids = src_ids.to(self.device)
                tgt_ids = tgt_ids.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)

                # For BLEU, we need to decode sentence by sentence
                batch_hypotheses = []
                batch_references = []

                for i in range(src_ids.size(0)): # Iterate over each sentence in the batch
                    single_src_ids = src_ids[i].unsqueeze(0) # (1, seq_len)
                    single_src_mask = src_mask[i].unsqueeze(0) # (1, 1, 1, seq_len)
                    
                    # Encode source for beam search
                    single_src_embedded = self.model.src_embeddings(single_src_ids)
                    single_enc_output = self.model.encoder(single_src_embedded, single_src_mask)

                    # Decode using beam search
                    hyp_ids = self._beam_search_decode(
                        single_enc_output, 
                        single_src_mask, 
                        self.config.evaluation.beam_search.max_output_length_offset + single_src_ids.size(1),
                        self.config.evaluation.beam_search.beam_size,
                        self.config.evaluation.beam_search.length_penalty_alpha
                    )
                    
                    # Convert IDs to tokens for BLEU calculation
                    hyp_tokens = [self.tgt_tokenizer.id_to_token(idx) for idx in hyp_ids]
                    
                    # Get reference tokens (excluding SOS/EOS/PAD for BLEU)
                    ref_ids = tgt_ids[i].tolist()
                    ref_tokens = [self.tgt_tokenizer.id_to_token(idx) for idx in ref_ids 
                                  if idx not in [self.tgt_tokenizer.get_sos_id(), 
                                                 self.tgt_tokenizer.get_eos_id(), 
                                                 self.tgt_tokenizer.get_pad_id()]]
                    
                    batch_hypotheses.append(hyp_tokens)
                    batch_references.append([ref_tokens]) # NLTK expects list of references for each hypothesis

                all_hypotheses.extend(batch_hypotheses)
                all_references.extend(batch_references)

                # For perplexity, use teacher forcing on the whole batch
                decoder_input = tgt_ids[:, :-1]
                target_for_loss = tgt_ids[:, 1:]
                decoder_mask = tgt_mask[:, :, :-1, :-1]

                logits = self.model(src_ids, decoder_input, src_mask, decoder_mask)
                
                # Calculate NLL loss for perplexity
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=self.tgt_tokenizer.get_pad_id())
                nll_loss = loss_fct(logits.view(-1, logits.size(-1)), target_for_loss.view(-1))
                
                total_nll_loss += nll_loss.item()
                total_tokens += (target_for_loss != self.tgt_tokenizer.get_pad_id()).sum().item()

        # Calculate metrics
        bleu_score = self.metrics.calculate_bleu(all_references, all_hypotheses)
        
        # Calculate perplexity
        avg_nll_loss = total_nll_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_nll_loss) if avg_nll_loss > 0 else float('inf')

        results = {
            "BLEU": bleu_score,
            "Perplexity": perplexity
        }
        return results

    def translate(self, src_sentence: str, max_len: int = None):
        """
        Translates a single sentence using beam search.
        Args:
            src_sentence (str): The input sentence to translate.
            max_len (int, optional): Maximum length for the translated output.
                                     If None, uses config's max_output_length_offset + input_len.
        Returns:
            str: The translated sentence.
        """
        self.model.eval()
        
        # Tokenize and numericalize source sentence
        src_ids = self.src_tokenizer.encode(src_sentence.strip())
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(self.device) # (1, seq_len)
        
        # Create source mask
        src_mask = (src_tensor != self.src_tokenizer.get_pad_id()).unsqueeze(1).unsqueeze(1) # (1, 1, 1, seq_len)

        # Determine max output length
        if max_len is None:
            max_len = src_tensor.size(1) + self.config.evaluation.beam_search.max_output_length_offset
        
        # Encode source
        with torch.no_grad():
            src_embedded = self.model.src_embeddings(src_tensor)
            enc_output = self.model.encoder(src_embedded, src_mask)

            # Decode using beam search
            hyp_ids = self._beam_search_decode(
                enc_output, 
                src_mask, 
                max_len,
                self.config.evaluation.beam_search.beam_size,
                self.config.evaluation.beam_search.length_penalty_alpha
            )
        
        # Convert IDs back to text
        translated_text = self.tgt_tokenizer.decode(hyp_ids)
        return translated_text