import logging
import os
import re

import numpy as np
import pyrouge
from numba import jit

import json

import tokenization
from utils.rouge import calc_rouge_score


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def collect_generated(i, idx, original_data, decode_dir_pred, collected_file):
    decoded_file = os.path.join(decode_dir_pred, "%06d_decoded.txt" % idx)
    contex = ""
    with open(decoded_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            contex = contex + line.strip()
    with open(collected_file, 'a', encoding='utf-8') as f:
        data = original_data[i]
        f.write('{}\n'.format(json.dumps({'guid':data.guid, 'input':data.article, 'golden': data.true_summary, 'generated': contex})))


def write_batch_for_rouge(ref_sents_list, all_stage_results, start_idx, ref_dir, decode_dir, trunc_dec_dir, hps,
                          true_num, original_data, decode_dir_pred, collected_file):
    idx = start_idx
    for i in range(len(ref_sents_list)):
        ref_sents = ref_sents_list[i]
        for stage_n, stage_n_results in enumerate(all_stage_results):
            decoded_words = stage_n_results[i]
            ref_len = len(' '.join(ref_sents).split())
            trunc_dir = trunc_dec_dir if i == len(ref_sents_list) - 1 else None
            write_pred_for_rouge(decoded_words, idx, decode_dir[stage_n], hps, ref_len, trunc_dir)
        write_ref_for_rouge(ref_sents, ref_dir, idx, hps)
        collect_generated(i,idx, original_data, decode_dir_pred, collected_file)
        idx += 1
        if idx - start_idx >= true_num:
            break
    return idx

def write_all_beam_candidates(original_data, all_candidates, n_top, result_dir, true_num, hps, start_idx):
    idx = start_idx
    f = open(os.path.join(result_dir, 'beam_results.txt'), 'a', encoding='utf-8')
    for i, data in enumerate(original_data):
        print('n_top', n_top)
        for i_beam in range(n_top):
            this_result = ' '.join(all_candidates[i_beam][0][i])
            this_result = this_result.replace(' ##', '')
            f.write('{}\n'.format(json.dumps({
                'guid': data.guid,
                'input': data.article,
                'golden': data.true_summary,
                'generated': this_result
            })))
        idx = idx + 1
        if idx - start_idx >= true_num:
            break

def tokenize_origin_summary(summary_file, hps):
    tokenizer = tokenization.FullTokenizer(vocab_file=hps.vocab_file, do_lower_case=hps.do_lower_case)
    msg = []
    for line in open(summary_file, 'r', encoding='utf-8'):
        summary_sent = ' '.join(tokenizer.tokenize(line.strip())).replace(' ##', '')
        msg.append(summary_sent.strip())
    with open(summary_file, 'w', encoding='utf-8') as writer:
        for sent in msg:
            writer.write(sent + '\n')


def recover_from_sub_word(decode_file):
    """Recover [a ##pp ##le -> apple]"""
    msg = []
    for line in open(decode_file, 'r', encoding='utf-8'):
        msg.append(line.strip().replace(' ##', ''))
    with open(decode_file, 'w', encoding='utf-8') as writer:
        for sent in msg:
            writer.write(sent + '\n')


def parse_summary_to_sents(text):
    starters = "(mr|st|mrs|ms|dr|co|jr|inc|ltd) [.]"
    float_num = "([0-9]) [.] ([0-9])"
    websites = "[.] (com|net|org|io|gov)"
    abbr_double_word = "([a-z]) [.] ([a-z]) [.]"
    p = '[pointer]'

    text = re.sub(starters, '\\1 {}'.format(p), text)  # title
    text = re.sub(websites, '{} \\1'.format(p), text)

    text = re.sub(' ([a-z]) \.', ' \\1 {}'.format(p), text)  # person middle name
    text = re.sub('no \.', 'no {}'.format(p), text)

    text = re.sub(abbr_double_word, '\\1 {} \\2 {}'.format(p, p), text)  # abbr like: u . s .
    text = re.sub(float_num, '\\1 {} \\2'.format(p), text)  # float number
    text = re.sub('[.] ([0-9]{2}) ', '{} \\1 '.format(p), text)  # guns

    sents = para_to_sents_by_period(text.split())
    res = [sent.strip().replace(p, '.') for sent in sents]
    res = [sent.strip().replace('a . m . .', 'a . m .') for sent in res]
    res = [sent.strip().replace('p . m . .', 'p . m .') for sent in res]
    return res


def parse_nyt_abs_to_sents(abstract):
    delim = ';'
    return [sent.strip() + ' ' + delim for sent in abstract.split(delim) if sent]


def para_to_sents_by_period(para):
    res_sents = []
    while len(para) > 0:
        try:
            fst_period_idx = para.index(".")
        except ValueError:  # there is text remaining that doesn't end in "."
            fst_period_idx = len(para)
        sent = para[:fst_period_idx + 1]  # sentence up to and including the period
        para = para[fst_period_idx + 1:]  # everything else
        res_sents.append(' '.join(sent))
    return res_sents


def write_pred_for_rouge(decoded_words, ex_index, decode_dir, hps, ref_len=None, trunc_dec_dir=None):
    decoded_sents = parse_summary_to_sents(' '.join(decoded_words))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]

    # Write to file
    decoded_file = os.path.join(decode_dir, "%06d_decoded.txt" % ex_index)
    with open(decoded_file, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")
    recover_from_sub_word(decoded_file)

    # write truncated pred
    if hps.task_name == 'nyt' and trunc_dec_dir:
        trunc_dec_file = os.path.join(trunc_dec_dir, "%06d_decoded.txt" % ex_index)
        trunc_dec_words = decoded_words[:ref_len]
        trunc_decoded_sents = parse_nyt_abs_to_sents(' '.join(trunc_dec_words))
        trunc_decoded_sents = [make_html_safe(w) for w in trunc_decoded_sents]
        with open(trunc_dec_file, "w", encoding="utf-8") as f:
            for idx, sent in enumerate(trunc_decoded_sents):
                f.write(sent) if idx == len(trunc_decoded_sents) - 1 else f.write(sent + "\n")
        recover_from_sub_word(trunc_dec_file)


def write_ref_for_rouge(reference_sents, ref_dir, ex_index, hps):
    reference_sents = [make_html_safe(w) for w in reference_sents]
    ref_file = os.path.join(ref_dir, "%06d_reference.txt" % ex_index)
    with open(ref_file, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    tokenize_origin_summary(ref_file, hps)


def write_for_rouge(reference_sents, decoded_words, ex_index, ref_dir, decode_dir, trunc_dec_dir, hps):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
      ref_dir: str, directory for reference seq
      decode_dir: str, directory for decoded seq
      trunc_dec_dir: str, directory for truncated decoded seq
      hps: config
    """
    # First, divide decoded output into sentences
    if hps.task_name == 'nyt':
        ref_len = len(' '.join(reference_sents).split())
        trunc_dec_words = decoded_words[:ref_len]
        decoded_sents = parse_nyt_abs_to_sents(' '.join(decoded_words))
        trunc_decoded_sents = parse_nyt_abs_to_sents(' '.join(trunc_dec_words))
        trunc_decoded_sents = [make_html_safe(w) for w in trunc_decoded_sents]
    else:
        decoded_sents = parse_summary_to_sents(' '.join(decoded_words))
        trunc_decoded_sents = None

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(decode_dir, "%06d_decoded.txt" % ex_index)
    trunc_dec_file = os.path.join(trunc_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")
    if hps.task_name == 'nyt':
        with open(trunc_dec_file, "w", encoding="utf-8") as f:
            for idx, sent in enumerate(trunc_decoded_sents):
                f.write(sent) if idx == len(trunc_decoded_sents) - 1 else f.write(sent + "\n")
        recover_from_sub_word(trunc_dec_file)
    recover_from_sub_word(decoded_file)
    tokenize_origin_summary(ref_file, hps)


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Log ROUGE results to screen and write to file.

    Args:
      results_dict: the dictionary returned by pyrouge
      dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    logging.info(log_str)  # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    logging.info("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w", encoding='utf-8') as f:
        f.write(log_str)


@jit()
def find_first(vec, item):
    res = []
    l = vec.shape[1]
    for i in range(vec.shape[0]):
        if item not in vec[i]:
            res.append(l)
        else:
            for j in range(l):
                if item == vec[i][j]:
                    res.append(j)
                    break
    return res


def calculate_reward(logits, batch, pad_id):
    reward = []
    # shape of logits: [b, length]
    decode_ids = logits.astype(np.int32, copy=False)
    decode_length = find_first(decode_ids, pad_id)
    references = batch.target_ids_oo  # [b, length]
    ref_length = batch.target_len.tolist()
    for each_ids, each_length, each_ref, each_ref_len in zip(decode_ids, decode_length, references, ref_length):
        reward.append(1.0 - calc_rouge_score(each_ids[:each_length], each_ref[:each_ref_len]))
    return np.array(reward)
