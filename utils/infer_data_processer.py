"""
In order to do beam search, we need to multiply needed features by beam_size times in batch axis,
the first axis changed from [batch] to [batch * beam_size],
this is what functions in this file work for.
"""
import numpy as np


def prepare_inf_features(features, params, encoder_output, encoder_output_origin=None, topic_memory=None):
    """Expand source data: [batch, ...] => [batch * beam_size, ...] """
    decode_length = params.max_out_seq_length
    beam_size = params.beam_size
    batch_size = features.source_ids.shape[0]

    # Expand the inputs
    # [batch, length] => [batch, beam_size, length]
    features.source_ids = np.expand_dims(features.source_ids, 1)
    features.source_ids = np.tile(features.source_ids, [1, beam_size, 1])
    shape = features.source_ids.shape

    # [batch, beam_size, length] => [batch * beam_size, length]
    features.source_ids = np.reshape(features.source_ids, [shape[0] * shape[1], shape[2]])
    # ------------------------------------------------------------------------------------------
    # Expand the word inputs
    # [batch, length] => [batch, beam_size, length]
    if hasattr(features, "topic_words_ids"):
        features.topic_words_ids = np.expand_dims(features.topic_words_ids, 1)
        features.topic_words_ids = np.tile(features.topic_words_ids, [1, beam_size, 1])
        shape = features.topic_words_ids.shape

        # [batch, beam_size, length] => [batch * beam_size, length]
        features.topic_words_ids = np.reshape(features.topic_words_ids, [shape[0] * shape[1], shape[2]])
    # ------------------------------------------------------------------------------------------
    # Expand the inputs oo
    # [batch, length] => [batch, beam_size, length]
    features.source_ids_oo = np.expand_dims(features.source_ids_oo, 1)
    features.source_ids_oo = np.tile(features.source_ids_oo, [1, beam_size, 1])
    shape = features.source_ids_oo.shape

    # [batch, beam_size, length] => [batch * beam_size, length]
    features.source_ids_oo = np.reshape(features.source_ids_oo, [shape[0] * shape[1], shape[2]])
    # ------------------------------------------------------------------------------------------
    # For source sequence length
    features.source_len = np.expand_dims(features.source_len, 1)
    features.source_len = np.tile(features.source_len, [1, beam_size])
    shape = features.source_len.shape

    max_length = np.ones_like(features.source_len) * decode_length  # [batch, beam_size]

    # [batch, beam_size] => [batch * beam_size]
    features.source_len = np.reshape(features.source_len, [shape[0] * shape[1]])
    # ------------------------------------------------------------------------------------------
    if hasattr(features, "topic_words_lens"):
        # For topic mem sequence length
        features.topic_words_len = np.expand_dims(features.topic_words_lens, 1)
        features.topic_words_len = np.tile(features.topic_words_len, [1, beam_size])
        shape = features.topic_words_len.shape

        #max_length = np.ones_like(features.topic_words_lens) * decode_length  # [batch, beam_size]

        # [batch, beam_size] => [batch * beam_size]
        features.topic_words_len = np.reshape(features.topic_words_len, [shape[0] * shape[1]])
        #print('features.topic_words_len', features.topic_words_len.shape)
    # ------------------------------------------------------------------------------------------
    # Expand the encoder output
    # [batch, length, dim] => [batch, beam_size, length, dim]
    features.encoder_output = np.expand_dims(encoder_output, 1)
    features.encoder_output = np.tile(features.encoder_output, [1, beam_size, 1, 1])
    shape = features.encoder_output.shape

    # [batch, beam_size, length, dim] => [batch * beam_size, length, dim]
    features.encoder_output = np.reshape(features.encoder_output, [shape[0] * shape[1], shape[2], shape[3]])
    # ------------------------------------------------------------------------------------------
        
    if topic_memory is not None:
        features.topic_word_memory = np.expand_dims(topic_memory, 1)
        features.topic_word_memory = np.tile(features.topic_word_memory, [1, beam_size, 1, 1])
        shape = features.topic_word_memory.shape
        # [batch, beam_size, length, dim] => [batch * beam_size, length, dim]
        features.topic_word_memory = np.reshape(features.topic_word_memory, [shape[0] * shape[1], shape[2], shape[3]])
        print('features.topic_word_memory', features.topic_word_memory.shape)
        
    if encoder_output_origin is not None:
        features.encoder_output_origin = np.expand_dims(encoder_output_origin, 1)
        features.encoder_output_origin = np.tile(features.encoder_output_origin, [1, beam_size, 1, 1])
        shape = features.encoder_output_origin.shape
        print('features.encoder_output_origin', np.mean(features.encoder_output_origin))
        print('features.encoder_output', np.mean(features.encoder_output))

        # [batch, beam_size, length, dim] => [batch * beam_size, length, dim]
        features.encoder_output_origin = np.reshape(features.encoder_output_origin, [shape[0] * shape[1], shape[2], shape[3]])
    return features, max_length, batch_size


def prepare_inf_2_features(features, decode_seq, decode_length, params, un_expand=True):
    beam_size = params.beam_size
    features.time_step = np.array(0)
    features.decode_seq = np.array(decode_seq, dtype=np.int32)
    features.decode_seq = np.expand_dims(features.decode_seq, 1)
    features.decode_length = np.array(decode_length, dtype=np.int32)
    # freeze the beam size expanded features
    if un_expand:
        features.source_len = np.reshape(features.source_len, [-1, beam_size])[:, 0]
        shape = features.encoder_output.shape
        features.encoder_output = np.reshape(features.encoder_output, [-1, beam_size, shape[1], shape[2]])[:, 0]
        shape = features.source_ids_oo.shape
        features.source_ids_oo = np.reshape(features.source_ids_oo, [-1, beam_size, shape[1]])[:, 0]
    return features


def prepare_inf_2_features_new(features, decode_seq, decode_length, params):
    pad_id = params.padId
    # beam search
    features.time_step = np.array(0)
    # features
    beam_size = params.beam_size
    features.decode_seq = np.array(decode_seq, dtype=np.int32)
    features.decode_length = np.array(decode_length, dtype=np.int32)
    features.decode_seq = features.decode_seq[:, :max(features.decode_length)]
    # pad the decoded sequence
    for i in range(params.eval_batch_size):
        features.decode_seq[i, features.decode_length[i]:] = pad_id

    features.decode_seq = np.expand_dims(features.decode_seq, 1)
    features.decode_seq = np.tile(features.decode_seq, [1, beam_size, 1])  # [b, beam, l_t]
    features.decode_length = np.expand_dims(features.decode_length, 1)
    features.decode_length = np.tile(features.decode_length, [1, beam_size])  # [b, beam]
    shape = features.decode_length.shape
    # [batch, beam_size] => [batch * beam_size]
    features.decode_length = np.reshape(features.decode_length, [shape[0] * shape[1]])

    # expanded features by beam_size
    if features.source_len.shape[0] != params.beam_size * params.eval_batch_size:
        features.encoder_output = np.expand_dims(features.encoder_output, 1)
        features.encoder_output = np.tile(features.encoder_output, [1, beam_size, 1, 1])
        features.source_len = np.expand_dims(features.source_len, 1)
        features.source_len = np.tile(features.source_len, [1, beam_size])
        shape = features.encoder_output.shape
        # [batch, beam_size, length, dim] => [batch * beam_size, length, dim]
        features.encoder_output = np.reshape(features.encoder_output, [shape[0] * shape[1], shape[2], shape[3]])
        shape = features.source_len.shape
        # [batch, beam_size] => [batch * beam_size]
        features.source_len = np.reshape(features.source_len, [shape[0] * shape[1]])

    return features
