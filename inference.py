import logging
import os
from enum import IntEnum

import numpy as np
import config
from data_reading import processors, abstract2sents_func
from data_reading.batcher import Batcher
from model_pools import modeling, model_pools
from model_pools.beamsearch import create_inference_ops_general
from model_pools.py_beam_search import run_beam_search
from utils.fine_tune import fine_tune
from utils.infer_data_processer import prepare_inf_features, prepare_inf_2_features, prepare_inf_2_features_new
from utils.infer_utils import decode_target_ids, all_batch_already_decoded, find_first_pad_token
from utils.metric_utils import write_batch_for_rouge, rouge_eval, rouge_log, write_all_beam_candidates
from utils.utils import init_sentence_level_info


class InferType(IntEnum):
    single = 1
    two_step = 2
    three_step = 3


def determine_infer_type(model_class):
    if hasattr(model_class, 'decode_infer_2') and callable(model_class.decode_infer_2):
        if hasattr(model_class, 'decode_infer_sent') and callable(model_class.decode_infer_sent):
            return InferType.three_step
        else:
            return InferType.two_step
    return InferType.single


# define run functions
def run(dev_model, ops, data):
    return dev_model.gSess_train.run(ops, dev_model.make_infer_feed_dict(data))

def run2(dev_model, ops1, ops2, data):
    return dev_model.gSess_train.run([ops1, ops2], dev_model.make_infer_feed_dict(data))

def run3(dev_model, ops1, ops2, ops3, data):
    return dev_model.gSess_train.run([ops1, ops2, ops3], dev_model.make_infer_feed_dict(data))

def run_infer_2(dev_model, ops, data):
    return dev_model.gSess_train.run(ops, dev_model.make_stage_2_infer_feed_dict(data))


# define inference operations
def create_infer_op(dev_model, hps):
    with dev_model.graph.as_default():
        return create_inference_ops_general(dev_model.decode_infer, hps, feature_prefix="target")


def create_infer_op_2(dev_model, use_beam_search):
    with dev_model.graph.as_default():
        return dev_model.decode_infer_2_bs() if use_beam_search else dev_model.decode_infer_2()


def create_infer_op_sent(dev_model):
    with dev_model.graph.as_default():
        return dev_model.decode_infer_sent()


def single_stage_model_inference(dev_model, infer_op, batch_data, params):
    """Inference for single model"""
    if hasattr(dev_model, 'sentence_rep'):  # fake feed data
        batch_data.tiled_sentence_rep = np.zeros((1, params.hidden_size))
    # get output of encoder
    #print('!!!!!!!!!!!!dev_model.encoder_output', dev_model.encoder_output)
    if hasattr(dev_model, 'encoder_output_origin'):
        encoder_output, topic_word_memory, encoder_output_origin = run3(dev_model, dev_model.encoder_output, dev_model.topic_word_memory, dev_model.encoder_output_origin, batch_data)
    elif hasattr(dev_model, 'topic_word_memory'):
        encoder_output, topic_word_memory = run2(dev_model, dev_model.encoder_output, dev_model.topic_word_memory, batch_data)
    else:
        encoder_output = run(dev_model, dev_model.encoder_output,  batch_data)
        
    if hasattr(dev_model, 'sentence_rep'):
        sentence_rep = run(dev_model, dev_model.sentence_rep, batch_data)
        batch_data.sentence_rep = sentence_rep
        batch_data.tiled_sentence_rep = np.reshape(np.tile(sentence_rep, [1, params.beam_size, 1]),
                                                   [-1, params.hidden_size])
    # prepare decoder data
    if hasattr(dev_model, 'encoder_output_origin'):
        new_batch_data, _, _ = prepare_inf_features(batch_data, params, encoder_output, encoder_output_origin=encoder_output_origin, topic_memory=topic_word_memory)
    elif hasattr(dev_model, 'topic_word_memory'):
        new_batch_data, _, _ = prepare_inf_features(batch_data, params, encoder_output, topic_memory=topic_word_memory)
    else:
        new_batch_data, _, _ = prepare_inf_features(batch_data, params, encoder_output)
    # decoder beam search
    decode_seq = run(dev_model, infer_op, new_batch_data)
    # [batch, top_beam, len] => [batch, len], [batch, top_beam] => [batch]
    decode_seq_first = decode_seq[:, 0, :].tolist()
    n_tops = params.top_beams
    all_candidates = []
    for i in range(n_tops):
        all_candidates.append(decode_seq[:, i, :].tolist())
    return decode_seq_first, all_candidates


def two_stage_model_inference(dev_model, infer_op, infer2_op, batch_data, params):
    """Inference for two step model"""
    decode_seq, decode_beam_candidates = single_stage_model_inference(dev_model, infer_op, batch_data, params)

    # step 2
    decode_length = find_first_pad_token(decode_seq, params)
    feature_prep_func = prepare_inf_2_features_new if params.use_beam_search else prepare_inf_2_features
    new_batch_data = feature_prep_func(batch_data, decode_seq, decode_length, params)
    if not params.use_beam_search:
        fine_tuned_logits = run_infer_2(dev_model, infer2_op, new_batch_data)
    else:
        max_time_step = max(decode_length)
        fine_tuned_logits = run_beam_search(new_batch_data, dev_model, run_infer_2, infer2_op, max_time_step, params)
    fine_tuned_seq = fine_tuned_logits.tolist()
    return [decode_seq, fine_tuned_seq], decode_beam_candidates


def three_stage_model_inference(dev_model, infer_op, infer2_op, infer_sent_op, batch_data, params):
    """Inference for three step model"""
    period_id = params.vocab['.']

    # step 1
    decode_seq, decode_beam_candidates = single_stage_model_inference(dev_model, infer_op, batch_data, params)

    # step 2, sentence level fine tune
    decode_length = find_first_pad_token(decode_seq, params)
    new_batch_data = prepare_inf_2_features(batch_data, decode_seq, decode_length, params)
    new_batch_data.sent_level_attn_bias = init_sentence_level_info(period_id, decode_seq)
    sent_fine_tuned_logits = run_infer_2(dev_model, infer_sent_op, new_batch_data)
    sent_fine_tuned_seq = sent_fine_tuned_logits.tolist()

    # step 3, word level fine tune
    decode_length = find_first_pad_token(sent_fine_tuned_seq, params)
    new_batch_data = prepare_inf_2_features(new_batch_data, sent_fine_tuned_seq, decode_length, params, un_expand=False)
    # run beam search word level decoding
    # new_batch_data = prepare_inf_2_features_new(new_batch_data, sent_fine_tuned_seq, decode_length, params)
    # max_time_step = new_batch_data.decode_seq.shape[-1]
    # fine_tuned_logits, _ = run_beam_search(new_batch_data, dev_model, run_infer_2, infer2_op, max_time_step,
    #                                        params)
    fine_tuned_logits = run_infer_2(dev_model, infer2_op, new_batch_data)
    fine_tuned_seq = fine_tuned_logits.tolist()
    return [decode_seq, sent_fine_tuned_seq, fine_tuned_seq], decode_beam_candidates


def inference(hps):
    # Prepare dir
    print('test_iterate', hps.test_iterate)
    print('use_diverse_beam_search', hps.use_diverse_beam_search)
    print('write_all_beam', hps.write_all_beam)
    print('top_beams', hps.top_beams)
    print('beam_width', hps.beam_size)
    if hps.mode == 'test':
        result_dir = os.path.join(hps.output_dir, hps.test_file + '-results-%d/' % hps.test_iterate)
    else:
        result_dir = os.path.join(hps.output_dir, hps.dev_file + '-results-%d/' % hps.test_iterate)
    ref_dir, decode_dir = os.path.join(result_dir, 'ref'), os.path.join(result_dir, 'pred')
    decode_dir_pred = os.path.join(result_dir, 'pred')
    collected_file = os.path.join(result_dir, 'all.txt')
    dec_dir_stage_1, dec_dir_stage_2 = decode_dir + '_1', decode_dir + '_2'
    trunc_dec_dir = os.path.join(result_dir, 'trunc_pred')
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    if not os.path.exists(decode_dir):
        os.makedirs(decode_dir)
    if not os.path.exists(dec_dir_stage_1):
        os.makedirs(dec_dir_stage_1)
    if not os.path.exists(dec_dir_stage_2):
        os.makedirs(dec_dir_stage_2)
    if not os.path.exists(trunc_dec_dir):
        os.makedirs(trunc_dec_dir)
    abs2sents_func = abstract2sents_func(hps)
    if hps.eval_only:
        # calculate rouge and other metrics
        print('calculate rouge...')
        final_pred_dir = trunc_dec_dir if hps.task_name == 'nyt' else decode_dir
        results_dict = rouge_eval(ref_dir, final_pred_dir)
        rouge_log(results_dict, decode_dir)
        fine_tune(hps)
        exit()

    # Load configs
    bert_config = modeling.BertConfig.from_json_file(hps.bert_config_file)
    bert_config.max_position_embeddings = max(bert_config.max_position_embeddings, hps.max_out_seq_length * 2)
    model = model_pools[hps.model_name]
    processor = processors[hps.task_name.lower()](hps)

    validate_batcher = Batcher(processor, hps)
    # Build model graph
    print("create inference model...")
    dev_model = model(bert_config, validate_batcher, hps)
    dev_model.create_or_load_recent_model()
    print("inference model done")
    # Prepare
    results_num = 0
    idx, skipped_num = 0, 0
    infer_type = determine_infer_type(dev_model)

    # build inference graph
    logging.info('Build inference graph...')
    print('Build inference graph...')
    pred_seq, _ = create_infer_op(dev_model, hps)
    fine_tuned_seq = create_infer_op_2(
        dev_model,
        hps.use_beam_search) if infer_type == InferType.two_step or infer_type == InferType.three_step else None
    sent_fine_tune = create_infer_op_sent(dev_model) if infer_type == InferType.three_step else None
    logging.info('Start inference...')
    print('Start inference...')
    res_dirs = []

    while True:
        # predict one batch
        batch = dev_model.batcher.next_batch()
        if not batch:
            break
        if all_batch_already_decoded(ref_dir, decode_dir, idx, len(batch.source_ids)):
            idx += len(batch.source_ids)
            skipped_num += len(batch.source_ids)
            continue
        # inference ids seq
        if infer_type == InferType.single:
            ids_results, ids_all_candidates= single_stage_model_inference(dev_model, pred_seq, batch, hps)
            res_dirs = [decode_dir]
        elif infer_type == InferType.three_step:
            ids_results, ids_all_candidates = three_stage_model_inference(dev_model, pred_seq, fine_tuned_seq, sent_fine_tune, batch, hps)
            res_dirs = [dec_dir_stage_1, dec_dir_stage_2, decode_dir]
        else:
            ids_results, ids_all_candidates = two_stage_model_inference(dev_model, pred_seq, fine_tuned_seq, batch, hps)
            res_dirs = [dec_dir_stage_1, decode_dir]
        # convert to string
        decode_result = [decode_target_ids(each_seq_ids, batch, hps) for each_seq_ids in [ids_results]]

        n_top = hps.top_beams
        all_candidates = []
        for i in range(n_top):
            all_candidates.append([decode_target_ids(ids_all_candidates[i], batch, hps)])

        results_num += batch.true_num
        # save ref and label
        batch_summaries = [[sent.strip() for sent in abs2sents_func(each.summary)] for each in batch.original_data]
        idx = write_batch_for_rouge(batch_summaries, decode_result, idx, ref_dir, res_dirs, trunc_dec_dir,
                                    hps, batch.true_num, batch.original_data, decode_dir_pred, collected_file)
        if hps.write_all_beam:
          write_all_beam_candidates(batch.original_data, all_candidates, n_top, result_dir, batch.true_num, hps, idx)

        logging.info("Finished sample %d" % (results_num + skipped_num))

    logging.info('Start calculate ROUGE...')
    print('Start calculate ROUGE...')
    # calculate rouge and other metrics
    for i in range(len(res_dirs) - 1):
        results_dict = rouge_eval(ref_dir, res_dirs[i])
        rouge_log(results_dict, res_dirs[i])
    final_pred_dir = trunc_dec_dir if hps.task_name == 'nyt' else decode_dir
    results_dict = rouge_eval(ref_dir, final_pred_dir)
    rouge_log(results_dict, decode_dir)
    logging.info('Start fine tune the predictions...')
    fine_tune(hps)


if __name__ == "__main__":
    inference(config.FLAGS)
