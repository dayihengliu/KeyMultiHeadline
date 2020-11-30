import numpy as np
import tensorflow as tf

from model_pools.model_utils.layer import linear


def calculate_final_logits(decoder_output, all_att_weights, vocab_probs, source_ids_oo, max_out_oovs, src_mask,
                           vocab_size, tgt_seq_len):
    # select last layer weights
    avg_att_weights = all_att_weights[-1]  # [b, q_2, q_1]
    copy_probs = tf.reshape(avg_att_weights, [-1, tf.shape(src_mask)[1]])  # [b * q_2, q_1]
    # calculate copy gate
    with tf.variable_scope('copy_gate'):
        p_gen = tf.nn.sigmoid(linear(decoder_output, 1))  # [b * l_t, 1]

    # gate
    vocab_probs = p_gen * vocab_probs  # [b * l_t, v]
    copy_probs = (1 - p_gen) * copy_probs  # [b * l_t, l_s]

    extended_vocab_size = tf.add(tf.constant(vocab_size), max_out_oovs)  # []
    b = tf.shape(vocab_probs)[0]  # b * l_t
    extra_zeros = tf.zeros(shape=tf.stack([b, max_out_oovs], axis=0))  # [b * l_t, v']
    vocab_prob_extended = tf.concat(axis=1, values=[vocab_probs, extra_zeros])  # [b * l_t, v + v']
    batch_nums = tf.range(0, limit=tf.shape(vocab_probs)[0])  # [b * l_t]  (0, 1, 2, ...)
    batch_nums = tf.expand_dims(batch_nums, 1)  # [b * l_t, 1]
    attn_len = tf.shape(copy_probs)[1]  # q_1
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # [b * l_t, l_s]
    # tile source ids oo, [b, l_s] => [b * l_t, l_s]
    tiled_source_ids_oo = tf.tile(tf.expand_dims(source_ids_oo, 1), [1, tgt_seq_len, 1])
    tiled_source_ids_oo = tf.reshape(tiled_source_ids_oo, [-1, tf.shape(tiled_source_ids_oo)[2]])
    indices = tf.stack((batch_nums, tiled_source_ids_oo), axis=2)  # [b * l_t, l_s, 2]
    shape = tf.stack([tf.shape(vocab_probs)[0], extended_vocab_size], axis=0)  # [2]
    attn_prob_projected = tf.scatter_nd(indices, copy_probs, shape)  # [b * l_t, v + v']
    logits = vocab_prob_extended + attn_prob_projected  # [b * l_t, v + v']
    return logits


def calculate_two_copy_logits(decoder_output, all_att_weights1, vocab_probs, source_ids_oo, max_out_oovs, src_mask,
                           vocab_size, tgt_seq_len, all_att_weights2, memory_ids_oo, mem_mask):
    # select last layer weights
    avg_att_weights1 = all_att_weights1[-1]  # [b, q_2, q_1]
    avg_att_weights2 = all_att_weights2[-1]
    copy_probs1 = tf.reshape(avg_att_weights1, [-1, tf.shape(src_mask)[1]])  # [b * q_2, q_1]
    copy_probs2 = tf.reshape(avg_att_weights2, [-1, tf.shape(mem_mask)[1]])  # [b * q_2, q_1]
    # calculate copy gate
    with tf.variable_scope('copy_gate'):
        p_gen = tf.nn.sigmoid(linear(decoder_output, 1))  # [b * l_t, 1]

    # vocab_prob
    vocab_probs = p_gen * vocab_probs  # [b * l_t, v]
    extended_vocab_size = tf.add(tf.constant(vocab_size), max_out_oovs)  # []
    b = tf.shape(vocab_probs)[0]  # b * l_t
    extra_zeros = tf.zeros(shape=tf.stack([b, max_out_oovs], axis=0))  # [b * l_t, v']
    vocab_prob_extended = tf.concat(axis=1, values=[vocab_probs, extra_zeros])  # [b * l_t, v + v']
    
    # copy_probs1
    copy_probs1 = (1 - p_gen) * copy_probs1  # [b * l_t, l_s]
    batch_nums = tf.range(0, limit=tf.shape(vocab_probs)[0])  # [b * l_t]  (0, 1, 2, ...)
    batch_nums = tf.expand_dims(batch_nums, 1)  # [b * l_t, 1]
    attn_len = tf.shape(copy_probs1)[1]  # q_1
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # [b * l_t, l_s]
    # tile source ids oo, [b, l_s] => [b * l_t, l_s]
    tiled_source_ids_oo = tf.tile(tf.expand_dims(source_ids_oo, 1), [1, tgt_seq_len, 1])
    tiled_source_ids_oo = tf.reshape(tiled_source_ids_oo, [-1, tf.shape(tiled_source_ids_oo)[2]])
    indices = tf.stack((batch_nums, tiled_source_ids_oo), axis=2)  # [b * l_t, l_s, 2]
    shape = tf.stack([tf.shape(vocab_probs)[0], extended_vocab_size], axis=0)  # [2]
    attn_prob_projected1 = tf.scatter_nd(indices, copy_probs1, shape)  # [b * l_t, v + v']
     
    # copy_probs2
    copy_probs2 = (1 - p_gen) * copy_probs2  # [b * l_t, l_s]
    batch_nums = tf.range(0, limit=tf.shape(vocab_probs)[0])  # [b * l_t]  (0, 1, 2, ...)
    batch_nums = tf.expand_dims(batch_nums, 1)  # [b * l_t, 1]
    attn_len = tf.shape(copy_probs2)[1]  # q_1
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # [b * l_t, l_s]
    # tile memory ids oo, [b, l_s] => [b * l_t, l_s]
    tiled_memory_ids_oo = tf.tile(tf.expand_dims(memory_ids_oo, 1), [1, tgt_seq_len, 1])
    tiled_memory_ids_oo = tf.reshape(tiled_memory_ids_oo, [-1, tf.shape(tiled_memory_ids_oo)[2]])
    indices = tf.stack((batch_nums, tiled_memory_ids_oo), axis=2)  # [b * l_t, l_s, 2]
    shape = tf.stack([tf.shape(vocab_probs)[0], extended_vocab_size], axis=0)  # [2]
    attn_prob_projected2 = tf.scatter_nd(indices, copy_probs2, shape)  # [b * l_t, v + v']
    
    # final logits
    logits = vocab_prob_extended + 0.5 * attn_prob_projected1 + 0.5 * attn_prob_projected2  # [b * l_t, v + v']
    return logits


# trunct word idx, change those greater than vocab_size to unk
def trunct(x, vocab_size, unk_id):
    if x >= vocab_size:
        return unk_id
    else:
        return x


np_trunct = np.vectorize(trunct)

np_d_trunct = lambda x, vocab_size, unk_id: np_trunct(x, vocab_size, unk_id).astype(np.int32)


def tf_trunct(x, vocab_size, unk_id, name=None):
    with tf.name_scope(name, "d_spiky", [x]) as name:
        y = tf.py_func(np_d_trunct,
                       [x, vocab_size, unk_id],
                       [tf.int32],
                       name=name,
                       stateful=False)
        return y[0]
