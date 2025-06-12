import math
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Metrics:
    """
    Provides functions for calculating evaluation metrics like BLEU and Perplexity.
    """
    def __init__(self):
        self.smooth_function = SmoothingFunction().method1 # Default smoothing for BLEU

    def calculate_bleu(self, references, hypotheses):
        """
        Computes BLEU score.
        Args:
            references (list[list[str]]): List of reference tokenized sentences.
                                          Each reference can have multiple versions.
                                          e.g., [['the', 'cat', 'is', 'on', 'the', 'mat']]
            hypotheses (list[str]): List of hypothesis tokenized sentences.
                                    e.g., ['the', 'cat', 'is', 'on', 'the', 'mat']
        Returns:
            float: BLEU score.
        """
        # NLTK's sentence_bleu expects a list of reference sentences (each a list of tokens)
        # and a single hypothesis sentence (list of tokens).
        # For multiple hypotheses, loop and average or use corpus_bleu.
        # Here, we assume references is a list of lists of tokens, and hypotheses is a list of lists of tokens.
        # If references is a list of single reference sentences, wrap each in a list.
        
        # Example: references = [[['ref1_tok1', 'ref1_tok2']], [['ref2_tok1', 'ref2_tok2']]]
        #          hypotheses = [['hyp1_tok1', 'hyp1_tok2'], ['hyp2_tok1', 'hyp2_tok2']]
        
        if not references or not hypotheses or len(references) != len(hypotheses):
            return 0.0 # Cannot compute BLEU if inputs are invalid

        total_bleu = 0.0
        for ref, hyp in zip(references, hypotheses):
            # Ensure ref is a list of lists, even if only one reference is provided per hypothesis
            if not isinstance(ref[0], list):
                ref = [ref]
            total_bleu += sentence_bleu(ref, hyp, smoothing_function=self.smooth_function)
        
        return total_bleu / len(hypotheses) if hypotheses else 0.0

    def calculate_perplexity(self, logits, targets, ignore_index=None):
        """
        Computes perplexity.
        Args:
            logits (torch.Tensor): Predicted logits (batch_size, seq_len, vocab_size)
            targets (torch.Tensor): True target IDs (batch_size, seq_len)
            ignore_index (int, optional): Index to ignore in target (e.g., padding ID).
        Returns:
            float: Perplexity score.
        """
        # Flatten logits and targets
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        # Calculate cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        loss = loss_fct(logits, targets)

        # Perplexity is exp(loss)
        return math.exp(loss.item())

    def calculate_f1(self, predictions, ground_truths):
        """
        Computes F1 score. (Placeholder, specific implementation depends on task)
        Args:
            predictions (list): List of predicted labels/entities.
            ground_truths (list): List of true labels/entities.
        Returns:
            float: F1 score.
        """
        # This is a generic placeholder. For parsing, F1 is typically on constituents.
        # For classification, it's on labels.
        # For now, return 0.0 as it's not directly used in the NMT context.
        return 0.0