import logging
import os

import numpy as np
import six
from nltk import ngrams


def all_batch_already_decoded(ref_dir, decode_dir, start_idx, batch_len):
    for i in range(batch_len):
        ref_file = os.path.join(ref_dir, "%06d_reference.txt" % (start_idx + i))
        decoded_file = os.path.join(decode_dir, "%06d_decoded.txt" % (start_idx + i))
        if os.path.exists(ref_file) and os.path.exists(decoded_file):
            continue
        else:
            return False
    logging.info('Sample {} - {} has already been decoded, skip...'.format(start_idx, start_idx + batch_len - 1))
    return True


def write_result_to_file(results, file):
    with open(file, "w", encoding='utf-8') as outfile:
        for decoded in results:
            decoded = str.join(" ", decoded)
            outfile.write("%s\n" % decoded)


def decode_target_ids(decode_ids_list, batch, hps):
    vocab_words = hps.vocab_words
    extra_vocab_list = batch.source_oovs
    decoded = []
    for decode_ids, extra_vocab in zip(decode_ids_list, extra_vocab_list):
        syms = []
        extended_vocab = vocab_words + extra_vocab
        for idx in decode_ids:
            if isinstance(idx, six.integer_types):
                sym = extended_vocab[idx]
            else:
                sym = idx
            if sym == hps.pad:
                break
            syms.append(sym)

        decoded.append(syms)
    return decoded


def find_first_pad_token(decode_seq, hps):
    seq_length = []
    for each_seq in decode_seq:
        if hps.padId in each_seq:
            seq_length.append(each_seq.index(hps.padId) + 1)  # length is actual length + 1(first pad token)
        else:
            seq_length.append(len(each_seq))
    return seq_length


def filter_exist_tri_grams(candidate_seqs):
    """
    :type candidate_seqs: tensor. shape [batch, 2 * beam_size, seq_length]
    :return bool tensor. shape [batch, 2 * beam_size]
    """

    def has_dup_tri_grams(seq):
        seq = trunc_pad_tokens(seq)
        return len(seq) - len(set(ngrams(seq, 3))) != 2

    def trunc_pad_tokens(seq):
        pad_idxs = np.where(seq == 0)[0]
        idx = pad_idxs[0] if len(pad_idxs) > 0 else len(seq)
        return seq[:idx]

    shape = candidate_seqs.shape
    batch, beam_size = shape[0], shape[1]
    return np.array([[has_dup_tri_grams(candidate_seqs[b][beam]) for beam in range(beam_size)] for b in range(batch)],
                    dtype=bool)
