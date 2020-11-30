import logging
import os
import re

from utils.metric_utils import rouge_eval, rouge_log, parse_summary_to_sents


def fine_tune_sents(file_sents):
    decoded_sents = parse_summary_to_sents(' '.join(file_sents))
    res = []
    for i, each in enumerate(decoded_sents):
        each = re.sub('"+', '"', each)
        if len(each.split()) < 3 or each in res:  # too short
            logging.info('Short sentence:\n{}'.format(decoded_sents))
            continue
        elif len(set(each.strip().split())) < 2 and len(each.split()) > 6:  # irregular sentence
            # logging.info('All same word:\n{}'.format(decoded_sents))
            continue
        else:
            res.append(remove_repeated_word(each))
    return res


def remove_repeated_word(sent: str):
    words = sent.strip().split()
    repeat_time = 1
    prev = None
    new_words = []
    for i, word in enumerate(words):
        if word == prev:
            repeat_time += 1
        else:
            repeat_time = 1
            prev = word
        if repeat_time > 2:
            continue
        elif repeat_time == 2:
            if len(words) > (i + 1) and words[i + 1] == prev:
                continue
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)


def fine_tune(hps):
    result_dir = os.path.join(hps.output_dir, 'test-results/')
    ref_dir, decode_dir = os.path.join(result_dir, 'ref'), os.path.join(result_dir, 'pred')
    new_decode_dir = os.path.join(result_dir, 'fine_tune_pred')
    if not os.path.exists(new_decode_dir):
        os.makedirs(new_decode_dir)
    for (dir_path, dir_names, file_names) in os.walk(decode_dir):
        for each_file in file_names:
            file = os.path.join(dir_path, each_file)
            file_sents = [line.strip() for line in open(file, 'r', encoding='utf-8')]
            file_sents = fine_tune_sents(file_sents)
            with open(os.path.join(new_decode_dir, each_file), 'w', encoding='utf-8') as out:
                for sent in file_sents:
                    out.write(sent + '\n')

    logging.info('Start calculate ROUGE...')
    # calculate rouge and other metrics
    results_dict = rouge_eval(ref_dir, new_decode_dir)
    rouge_log(results_dict, new_decode_dir)
