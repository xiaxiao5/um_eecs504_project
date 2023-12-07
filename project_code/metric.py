# criteria and loss
import collections
import torch
import torch.nn as nn
import math

def handbleu(pred_seq_batch, label_seq_batch):
    """
    Compute the mean BLEU over a batch of seqs of hand landmarks (1mark / hand).
    
    args:
        pred_seq: A seq of predicted hand pos number. (batch_size, seq_len); list of list 
        label_sqe: A seq of predicted hand pos number. (batch_size, seq_len); list of list
        k: the expected number of grams in predicted seq to be matched. Here it's just future window size
    """
    
    bleu_batch = []
    for i in range(len(pred_seq_batch)):
        pred_seq = ' '.join([str(_) for _ in pred_seq_batch[i]])
        label_seq = ' '.join([str(_) for _ in label_seq_batch[i]])
        bleu_batch.append(bleu(pred_seq, label_seq, len(label_seq)))
    score = mean(bleu_batch)
    
    return score

def bleu(pred_seq, label_seq, k):
    """Compute the BLEU.

    Defined in :numref:`sec_seq2seq_training`"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        """
        args:
            valid_len: torch; (bs, ) (int)
        """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

class MaskedMSELoss(nn.MSELoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        """
        args:
            valid_len: torch; (bs, ) (int)
        """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction="none"
        unweighted_loss = super(MaskedMSELoss, self).forward(
            pred, label)
        weighted_loss = (unweighted_loss * weights).mean() # mean over all not only valid ones
        return weighted_loss


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.
    
    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
