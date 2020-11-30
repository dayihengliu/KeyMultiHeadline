import json
import logging

import numpy as np
import tensorflow as tf


def init_sentence_level_info(period_id, batch_target_ids):
    """Add sentence level bias to deliberate the decoded sequence,
    return shape is [batch, 1, length, length] as other attention_bias return tensor.
    """
    sent_mask = []
    if isinstance(batch_target_ids, np.ndarray):
        batch_target_ids = batch_target_ids.tolist()
    for target_ids in batch_target_ids:
        sent_length = []
        while len(target_ids) > 0:
            try:
                fst_period_idx = target_ids.index(period_id)
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(target_ids) - 1
            target_ids = target_ids[fst_period_idx + 1:]  # everything else
            sent_length.append(fst_period_idx + 1)
        sent_mask.append([calculate_sentence_mask(sent_length)])
    return np.array(sent_mask, dtype=np.float32)  # (batch, 1, length, length)


def calculate_sentence_mask(sent_length, mask_all=False):
    """return shape: [total_length, total_length]"""
    inf = -1e9
    refined_sent_length = [length + sum(sent_length[:i]) for i, length in enumerate(sent_length)]

    def get_sent_idx(idx):
        low = 0
        for i in refined_sent_length:
            if idx + 1 <= i:
                return (idx, i) if not mask_all else (low, i)
            else:
                low = i

    total_length = sum(sent_length)
    ret_mask = []
    for i in range(total_length):
        mask_i = [0. for _ in range(total_length)]
        l, h = get_sent_idx(i)
        mask_i[l: h] = [inf] * (h - l)
        ret_mask.append(mask_i)
    return ret_mask


def get_depth_i_var_name(var_name, depth_i):
    return '/'.join(var_name.split('/')[:depth_i])


def print_parameters(graph, output_detail=False, output_to_logging=False, calculate_depth=4):
    flush = logging.info if output_to_logging else print
    total_parameters = 0
    parameters_string = '\n'
    module_parameters = {}
    with graph.as_default():
        total_num = len(tf.trainable_variables())
        for variable in tf.trainable_variables():

            shape = variable.get_shape()
            var_name = variable.name.split(':')[0]

            device_type = variable.device.strip('/device:') if variable.device else variable.device
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

            for depth_i in range(calculate_depth):
                depth_i_name = get_depth_i_var_name(var_name, depth_i)
                if depth_i_name in module_parameters:
                    module_parameters[depth_i_name] += variable_parameters
                else:
                    module_parameters[depth_i_name] = variable_parameters

            if len(shape) == 1:
                parameters_string += ('%s\t%s\t%d\n' % (device_type, var_name, variable_parameters))
            else:
                parameters_string += ('%s\t%s\t%s\t%d\n' % (device_type, var_name, str(shape), variable_parameters))

    if output_detail:
        flush(parameters_string)
    for key in module_parameters:
        flush('%s: %d params' % (key, module_parameters[key]))
    flush('Total %d variables, %s params' % (total_num, '{:,}'.format(total_parameters)))


def pprint_params(params):
    msg = json.loads(params.to_json())
    del msg['vocab']
    del msg['inv_vocab']
    del msg['vocab_words']
    del msg['vocab_out']
    del msg['inv_vocab_out']
    del msg['vocab_words_out']
    return msg


def gen_sess_config(hps):
    if hps.gpu:
        _config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
        device_str = ",".join([str(i) for i in hps.gpu])
        _config.gpu_options.visible_device_list = device_str
    else:
        _config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True),
                                 device_count={'gpu': 0})
    return _config
