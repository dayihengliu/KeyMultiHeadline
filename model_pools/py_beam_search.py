"""Python code to run beam search decoding instead of building static graph"""

import numpy as np

from data_reading.batcher import Batch
from utils.infer_utils import filter_exist_tri_grams


def run_beam_search(data: Batch, model, op_runner, op, max_time_step, hps):
    """Performs beam search decoding on the given example."""
    time_step = 0
    beam_size = hps.beam_size
    batch_size = hps.eval_batch_size
    decode_alpha = hps.decode_alpha
    decode_gamma = hps.decode_gamma
    top_beams = hps.top_beams
    eos_id = hps.padId
    min_value = -1e10

    # the init log prob of first beam is greater than other (beam_size - 1) beams
    # to ensure at first time step(all beam with same input), we have different output with top prob
    log_probs = np.array([[0.] + [min_value] * (beam_size - 1)], dtype=np.float32)  # [1, beam]
    log_probs = np.tile(log_probs, [batch_size, 1])  # [b, beam]
    alive_scores = np.zeros_like(log_probs)  # [b, beam]
    fin_seqs = np.zeros([batch_size, beam_size, max_time_step], np.int32)  # [b, beam, 1]
    fin_scores = np.full([batch_size, beam_size], min_value)  # [b, beam]
    fin_flags = np.zeros([batch_size, beam_size], np.bool)  # [b, beam]

    while time_step < max_time_step:
        # Run one step of the decoder to get the new info
        step_log_probs = op_runner(model, op, data)  # (b * beam, v)
        vocab_size = step_log_probs.shape[-1]

        step_log_probs = np.reshape(step_log_probs, [batch_size, beam_size, vocab_size])
        curr_log_probs = np.expand_dims(log_probs, 2) + step_log_probs  # add current vocab beam with previous one word

        # length penalty
        length_penalty = np.power((5.0 + (time_step + 1)) / 6.0, decode_alpha)
        curr_scores = curr_log_probs / length_penalty  # [b, beam, v]
        #print("hps.use_diverse_beam_search", hps.use_diverse_beam_search)
        if hps.use_diverse_beam_search:
            print('Diverse penalty!!!!!!!!!!!!!!!!!!!!!', hps.decode_gamma)
            # Diverse penalty
            indexs = np.argsort(curr_scores, -1)[:,:,::-1]
            values = np.full_like(curr_scores, np.arange(vocab_size) * decode_gamma)
            curr_scores = curr_scores - np.take(values, indexs)

        # Select top-k candidates
        curr_scores = np.reshape(curr_scores, [-1, beam_size * vocab_size])  # [b, beam * v]
        
        # get indices like: [3, vocab + 7, vocab * 3 + 180, ...], vocab_idx + beam_offset
        top_scores, top_indices = get_top_k(curr_scores, 2 * beam_size)  # [b, 2 * beam]
        beam_indices = top_indices // vocab_size  # [b, 2 * beam]
        symbol_indices = top_indices % vocab_size  # [b, 2 * beam]
        # Build candidate sequences
        seqs = data.decode_seq
        candidate_seqs = gather_2d(seqs, beam_indices)  # [b, 2 * beam, q']
        # put current decoded word id to given sequences
        candidate_seqs[:, :, time_step] = symbol_indices

        # Find alive sequences
        flags = np.equal(symbol_indices, eos_id)  # [b, beam]
        # with our 2 * beam results, we set those eos score to -inf
        alive_scores = top_scores + flags * min_value  # [b, 2 * beam]
        # filter dup tri-gram seqs
        tri_flags = filter_exist_tri_grams(candidate_seqs)
        alive_scores = alive_scores + tri_flags * min_value  # [b, 2 * beam]
        # and keep top beam ones
        alive_scores, alive_indices = get_top_k(alive_scores, beam_size)  # [b, beam]
        # get their correspond vocab ids
        alive_symbols = gather_2d(symbol_indices, alive_indices)  # [b, beam]
        # and their correspond beam indices
        alive_indices = gather_2d(beam_indices, alive_indices)  # [b, beam]
        # get their correspond previous sequences
        alive_seqs = gather_2d(seqs, alive_indices)  # [b, beam, q']
        # concat, ta_da -_-
        alive_seqs[:, :, time_step] = alive_symbols
        alive_log_probs = alive_scores * length_penalty
        # prepare for next loop refine
        data.decode_seq = alive_seqs
        log_probs = alive_log_probs

        # Select finished sequences
        step_fin_scores = top_scores + (1.0 - flags) * min_value  # [b, 2 * beam]
        fin_flags = np.concatenate([fin_flags, flags], axis=1)  # [batch, 3 * beam]
        fin_scores = np.concatenate([fin_scores, step_fin_scores], axis=1)
        fin_scores, fin_indices = get_top_k(fin_scores, beam_size)  # [b, beam]
        fin_flags = gather_2d(fin_flags, fin_indices)

        # we always keep beam fin_seqs along with their scores and use current candidate to update
        fin_seqs = np.concatenate([fin_seqs, candidate_seqs], axis=1)  # [b, 3 * beam, length]
        fin_seqs = gather_2d(fin_seqs, fin_indices)  # [b, beam, length]

        time_step += 1
        data.update_time()

    # if any seq have finalized, select finalized seqs & scores
    final_seqs = fin_seqs if np.any(fin_flags) else data.decode_seq
    final_scores = fin_scores if np.any(fin_flags) else alive_scores

    return final_seqs[:, :top_beams, :]


def gather_2d(params, indices):
    return np.stack([np.take(params[i], indices[i], axis=0) for i in range(params.shape[0])], axis=0)


def get_top_k(scores, k):
    indices = np.argsort(scores, -1)[:, :(-k - 1):-1]
    ret_scores = np.stack([np.take(scores[i], indices[i], axis=-1) for i in range(scores.shape[0])], axis=0)
    return ret_scores, indices
