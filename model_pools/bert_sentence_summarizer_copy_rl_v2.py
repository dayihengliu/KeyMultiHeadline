import tensorflow as tf

from model_pools import modeling
from model_pools.base_model import BaseModel
from model_pools.model_utils.layer import attention_bias, smooth_cross_entropy
from model_pools.model_utils.module import transformer_decoder
from model_pools.modeling import embedding_lookup, embedding_postprocessor
from utils.copy_utils import calculate_final_logits, tf_trunct


# noinspection PyAttributeOutsideInit
class BertSentenceSummarizerCopyRLV2(BaseModel):
    """
    Based on BertSentenceSummarizerCopyV2 and BertSentenceSummarizerCopyRL.
    CopyV2 Add RL loss.
    """

    def __init__(self, bert_config, batcher, hps):
        super(BertSentenceSummarizerCopyRLV2, self).__init__(hps, bert_config, batcher)

    def build_graph(self):
        with self.graph.as_default():
            self._build_summarization_model()

    def _add_placeholders(self):
        self.batch_size = self.hps.train_batch_size if self.is_training else self.hps.eval_batch_size
        self.input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
        self.input_len = tf.placeholder(tf.int32, [None], name='input_len')
        self.segment_ids = tf.placeholder(tf.int32, [None, None], name='segment_ids')
        self.output_ids = tf.placeholder(tf.int32, [None, None], name='output_ids')
        self.output_len = tf.placeholder(tf.int32, [None], name='output_len')
        self.input_mask = tf.sequence_mask(self.input_len,
                                           maxlen=tf.shape(self.input_ids)[1],
                                           dtype=tf.float32)  # [b, l_s]
        self.output_mask = tf.sequence_mask(self.output_len,
                                            maxlen=tf.shape(self.output_ids)[1],
                                            dtype=tf.float32)  # [b, l_t]
        self.out_segment_ids = tf.zeros_like(self.output_ids, dtype=tf.int32, name='out_segment_ids')
        # copy related placeholder
        self.output_label = tf.placeholder(tf.int32, [None, None], name='output_label')  # [b, l_t], output_ids_oo
        self.max_out_oovs = tf.placeholder(tf.int32, [], name='max_out_oovs')  # []
        self.input_ids_oo = tf.placeholder(tf.int32, [None, None], name='input_ids_oo')  # [b, l_s]
        self.tiled_len = tf.shape(self.output_label)[1]

        # decode sequence for second stage inference
        self.decode_seq = tf.placeholder(tf.int32, [None, None, None], name='decoded_seq')
        self.decode_length = tf.placeholder(tf.int32, [None], name='decoded_length')
        # encoder output for inference
        self.enc_output = tf.placeholder(tf.float32, [None, None, self.hps.hidden_size], name='enc_output')
        self.tiled_sentence_rep = tf.placeholder(tf.float32, [None, self.hps.hidden_size],
                                                 name='tiled_sentence_rep')
        self.infer_sentence_rep = tf.placeholder(tf.float32, [None, 1, self.hps.hidden_size], name='infer_sentence_rep')
        # sentence level attn bias, [batch_size, 1, length, length], during train & inference
        self.sent_level_attn_bias = tf.placeholder(tf.float32, [None, 1, None, None], name='sent_level_attn_bias')
        # word-level beam search inference
        self.time_step = tf.placeholder(tf.int32, [], name='time_step')
        # copy for inference
        self.infer_tiled_len = tf.shape(self.decode_seq)[2]
        # RL placeholder
        self.reward = tf.placeholder(tf.float32, [None], name='reward')

    def _build_summarization_model(self):
        is_training = self.is_training
        config = self.bert_config

        self._add_placeholders()

        '''Creates a classification model.'''
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=self.hps.use_tpu)  # use_one_hot_embeddings=Flags.tpu ?

        encoder_output = model.get_sequence_output()  # [b, l_s, h]

        self.sentence_rep = tf.expand_dims(model.get_pooled_output(), axis=1)  # [b, 1, h]

        self.encoder_output = encoder_output

        hidden_size = encoder_output.shape[2].value

        self.enc_attn_bias = attention_bias(self.input_mask, 'masking')

        with tf.variable_scope('bert', reuse=True):
            with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                # Perform embedding lookup on the target word ids.
                (self.out_embed, self.bert_embeddings) = embedding_lookup(
                    input_ids=self.output_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name='word_embeddings',
                    use_one_hot_embeddings=False)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.out_embed = embedding_postprocessor(
                    input_tensor=self.out_embed,
                    use_token_type=True,
                    token_type_ids=self.out_segment_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name='token_type_embeddings',
                    use_position_embeddings=True,
                    position_embedding_name='position_embeddings',
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

        with tf.variable_scope('decoder'):
            self.decoder_weights = self.bert_embeddings
            self.masked_out_embed = self.out_embed * tf.expand_dims(self.output_mask, -1)
            self.decoder_input = tf.pad(self.masked_out_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left
            # ################################################### decoding train - 1
            self.dec_attn_bias = attention_bias(tf.shape(self.masked_out_embed)[1], 'causal')
            self.all_att_weights, self.decoder_output_1 = transformer_decoder(self.decoder_input, self.encoder_output,
                                                                              self.dec_attn_bias, self.enc_attn_bias,
                                                                              self.hps, scope='t_decoder')
            sentence_rep = tf.tile(self.sentence_rep, [1, tf.shape(self.decoder_output_1)[1], 1])  # [b, l_t, e]
            # [b, l_t, e] => [b*l_t, v]
            copy_rep_1 = tf.concat([sentence_rep, self.decoder_output_1], axis=-1)  # [b, l_t, 2 * e]
            self.decoder_output_1 = tf.reshape(self.decoder_output_1, [-1, hidden_size])
            self.vocab_logits = tf.matmul(self.decoder_output_1, self.decoder_weights, False, True)  # (b*l_t, v)
            self.vocab_probs = tf.nn.softmax(self.vocab_logits)  # [b * l_t, v]
            vocab_size = len(self.hps.vocab)
            with tf.variable_scope('copy', reuse=tf.AUTO_REUSE):
                copy_rep_1 = tf.reshape(copy_rep_1, [-1, hidden_size * 2])
                self.logits = calculate_final_logits(copy_rep_1, self.all_att_weights, self.vocab_probs,
                                                     self.input_ids_oo, self.max_out_oovs, self.input_mask, vocab_size,
                                                     self.tiled_len)  # [b * l_t, v + v']
                self.pred_ids = tf.reshape(tf.argmax(self.logits, axis=-1), [self.batch_size, -1])  # [b, l_t]

            # ################################################### decoding train - 2
            self.second_dec_attn_bias = attention_bias(tf.shape(self.masked_out_embed)[1], 'cloze_bias')
            self.all_att_weights, self.decoder_output_2 = transformer_decoder(self.decoder_input, self.encoder_output,
                                                                              self.second_dec_attn_bias,
                                                                              self.enc_attn_bias,
                                                                              self.hps, scope='t_decoder', reuse=True)
            # [b, l_t, e] => [b*l_t, v]
            copy_rep_2 = tf.concat([sentence_rep, self.decoder_output_2], axis=-1)
            self.decoder_output_2 = tf.reshape(self.decoder_output_2, [-1, hidden_size])
            self.second_logits = tf.matmul(self.decoder_output_2, self.decoder_weights, False, True)  # (b*l_t, v)
            self.vocab_probs_2 = tf.nn.softmax(self.second_logits)  # [b * l_t, v]
            with tf.variable_scope('copy', reuse=tf.AUTO_REUSE):
                copy_rep_2 = tf.reshape(copy_rep_2, [-1, hidden_size * 2])
                self.second_logits = calculate_final_logits(copy_rep_2, self.all_att_weights,
                                                            self.vocab_probs_2,
                                                            self.input_ids_oo, self.max_out_oovs, self.input_mask,
                                                            vocab_size,
                                                            self.tiled_len)  # [b * l_t, v + v']

            # ################################################### decoding train - 3
            self.all_att_weights, self.decoder_output_3 = transformer_decoder(self.decoder_input, self.encoder_output,
                                                                              self.sent_level_attn_bias,
                                                                              self.enc_attn_bias,
                                                                              self.hps, scope='t_decoder', reuse=True)
            # [b, l_t, e] => [b*l_t, v]
            copy_rep_3 = tf.concat([sentence_rep, self.decoder_output_3], axis=-1)
            self.decoder_output_3 = tf.reshape(self.decoder_output_3, [-1, hidden_size])
            self.third_logits = tf.matmul(self.decoder_output_3, self.decoder_weights, False, True)  # (b*l_t, v)
            self.vocab_probs_3 = tf.nn.softmax(self.third_logits)  # [b * l_t, v]
            with tf.variable_scope('copy', reuse=tf.AUTO_REUSE):
                copy_rep_3 = tf.reshape(copy_rep_3, [-1, hidden_size * 2])
                self.third_logits = calculate_final_logits(copy_rep_3, self.all_att_weights,
                                                           self.vocab_probs_3,
                                                           self.input_ids_oo, self.max_out_oovs, self.input_mask,
                                                           vocab_size,
                                                           self.tiled_len)  # [b * l_t, v + v']

        with tf.variable_scope('loss'):
            self.ce = smooth_cross_entropy(
                self.logits,
                self.output_label,
                self.hps.label_smoothing)

            self.ce = tf.reshape(self.ce, tf.shape(self.output_label))  # [b, l_t]

            mle_1 = tf.reduce_sum(self.ce * self.output_mask, -1) / tf.reduce_sum(self.output_mask, -1)  # [b]

            self.first_loss = tf.reduce_sum(self.ce * self.output_mask, -1) / tf.reduce_sum(self.output_mask, -1)
            self.first_loss = tf.reduce_mean(self.first_loss)  # scalar

            self.second_ce = smooth_cross_entropy(
                self.second_logits,
                self.output_label,
                self.hps.label_smoothing)

            self.second_ce = tf.reshape(self.second_ce, tf.shape(self.output_label))  # [b, l_t]

            mle_2 = tf.reduce_sum(self.second_ce * self.output_mask, -1) / tf.reduce_sum(self.output_mask, -1)  # [b]

            self.second_loss = tf.reduce_mean(tf.reduce_sum(self.second_ce * self.output_mask, -1) / tf.reduce_sum(
                self.output_mask, -1))  # scalar

            self.ce = smooth_cross_entropy(
                self.third_logits,
                self.output_ids,
                self.hps.label_smoothing)

            self.ce = tf.reshape(self.ce, tf.shape(self.output_label))  # [b, l_t]

            mle_3 = tf.reduce_sum(self.ce * self.output_mask, -1) / tf.reduce_sum(self.output_mask, -1)  # [b]

            self.third_loss = tf.reduce_mean(tf.reduce_sum(self.ce * self.output_mask, -1) / tf.reduce_sum(
                self.output_mask, -1))  # scalar

            mle = mle_1 + mle_2 + mle_3
            self.rl_loss = tf.reduce_mean(mle * self.reward)  # scalar
            self.ml_loss = self.first_loss + self.second_loss + self.third_loss
            self.loss = self.hps.rl_lambda * self.rl_loss + (1 - self.hps.rl_lambda) * self.ml_loss
            tf.summary.scalar('first_loss', self.first_loss)
            tf.summary.scalar('second_loss', self.second_loss)
            tf.summary.scalar('third_loss', self.third_loss)
            tf.summary.scalar('reward', tf.reduce_mean(self.reward))
            tf.summary.scalar('rl_loss', self.rl_loss)
            tf.summary.scalar('ml_loss', self.ml_loss)
            tf.summary.scalar('loss', self.loss)

    def trunct(self, seq):
        vocab_size = len(self.hps.vocab)
        # trunct word idx, change those greater than vocab_size to zero
        shape = seq.shape
        new_seq = tf_trunct(seq, vocab_size, self.hps.unkId)
        new_seq.set_shape(shape)
        return new_seq

    def decode_infer(self, inputs, state):
        # state['enc']: [b * beam, l_s, e]  ,   state['dec']: [b * beam, q', e]
        # q' = previous decode output length
        # during infer, following graph are constructed using beam search
        with self.graph.as_default():
            config = self.bert_config

            target_sequence = inputs['target']
            target_sequence = self.trunct(target_sequence)
            target_length = inputs['target_length']
            target_seg_ids = tf.zeros_like(target_sequence, dtype=tf.int32, name='target_seg_ids_infer')
            tgt_mask = tf.sequence_mask(target_length,
                                        maxlen=tf.shape(target_sequence)[1],
                                        dtype=tf.float32)  # [b, q']

            with tf.variable_scope('bert', reuse=True):
                with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                    # Perform embedding lookup on the target word ids.
                    (tgt_embed, _) = embedding_lookup(
                        input_ids=target_sequence,
                        vocab_size=config.vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name='word_embeddings',
                        use_one_hot_embeddings=False)

                    # Add positional embeddings and token type embeddings, then layer
                    # normalize and perform dropout.
                    tgt_embed = embedding_postprocessor(
                        input_tensor=tgt_embed,
                        use_token_type=True,
                        token_type_ids=target_seg_ids,
                        token_type_vocab_size=config.type_vocab_size,
                        token_type_embedding_name='token_type_embeddings',
                        use_position_embeddings=True,
                        position_embedding_name='position_embeddings',
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings,
                        dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope('decoder', reuse=True):
                # [b, l_t, e]
                masked_tgt_embed = tgt_embed * tf.expand_dims(tgt_mask, -1)
                dec_attn_bias = attention_bias(tf.shape(masked_tgt_embed)[1], "causal")
                decoder_input = tf.pad(masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left

                infer_decoder_input = decoder_input[:, -1:, :]
                infer_dec_attn_bias = dec_attn_bias[:, :, -1:, :]

                all_att_weights, decoder_output, decoder_state = transformer_decoder(infer_decoder_input,
                                                                                     self.enc_output,
                                                                                     infer_dec_attn_bias,
                                                                                     self.enc_attn_bias,
                                                                                     self.hps,
                                                                                     state=state['decoder'],
                                                                                     scope='t_decoder')
                decoder_output = decoder_output[:, -1, :]  # [b * beam, e]
                logits = tf.matmul(decoder_output, self.decoder_weights, False, True)  # [b * beam, v]
                vocab_probs = tf.nn.softmax(logits)  # [b * l_t, v]
                vocab_size = len(self.hps.vocab)
                with tf.variable_scope('copy', reuse=tf.AUTO_REUSE):
                    copy_rep = tf.concat([self.tiled_sentence_rep, decoder_output], axis=-1)  # [b * beam, 2 * e]
                    logits = calculate_final_logits(copy_rep, all_att_weights, vocab_probs,
                                                    self.input_ids_oo, self.max_out_oovs, self.input_mask,
                                                    vocab_size, 1)  # [b * l_t, v + v']
                log_prob = tf.log(logits)
        return log_prob, {'encoder': state['encoder'], 'decoder': decoder_state}

    def decode_infer_2_bs(self):
        # beam search version
        # during second stage decoding, we have a decoded sequence, so do not need to feed state(no incremental dec)
        # at time i, we calculate i-th attn_bias, get i-th decoder output
        with self.graph.as_default():
            config = self.bert_config

            target_sequence = tf.reshape(self.decode_seq, [self.hps.eval_batch_size * self.hps.beam_size, -1])
            target_length = self.decode_length
            target_seg_ids = tf.zeros_like(target_sequence, dtype=tf.int32, name='target_seg_ids_infer')
            tgt_mask = tf.sequence_mask(target_length,
                                        maxlen=tf.shape(target_sequence)[1],
                                        dtype=tf.float32)  # [b, q']

            with tf.variable_scope('bert', reuse=True):
                with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                    # Perform embedding lookup on the target word ids.
                    (tgt_embed, _) = embedding_lookup(
                        input_ids=target_sequence,
                        vocab_size=config.vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name='word_embeddings',
                        use_one_hot_embeddings=False)

                    # Add positional embeddings and token type embeddings, then layer
                    # normalize and perform dropout.
                    tgt_embed = embedding_postprocessor(
                        input_tensor=tgt_embed,
                        use_token_type=True,
                        token_type_ids=target_seg_ids,
                        token_type_vocab_size=config.type_vocab_size,
                        token_type_embedding_name='token_type_embeddings',
                        use_position_embeddings=True,
                        position_embedding_name='position_embeddings',
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings,
                        dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope('decoder', reuse=True):
                # [b, l_t, e]
                masked_tgt_embed = tgt_embed * tf.expand_dims(tgt_mask, -1)
                dec_attn_bias = attention_bias(tf.shape(masked_tgt_embed)[1], "cloze_bias")
                # this operation is necessary as the att bias is shifted
                infer_decoder_input = tf.pad(masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left

                # This operation is wrong!!!
                # infer_dec_attn_bias = dec_attn_bias[:, :, self.time_step:self.time_step + 1, :]

                all_att_weights, decoder_output = transformer_decoder(infer_decoder_input,
                                                                      self.enc_output,
                                                                      dec_attn_bias,
                                                                      self.enc_attn_bias,
                                                                      self.hps,
                                                                      scope='t_decoder')
                decoder_output = decoder_output[:, self.time_step, :]  # [b * beam, e]
                logits = tf.matmul(decoder_output, self.decoder_weights, False, True)  # [b * beam, v]
                log_prob = tf.nn.log_softmax(logits)
        return log_prob

    def decode_infer_2(self):
        # stage 2, word level inference using decoded sequence
        # l_t = decode sequence length
        # during infer, following graph are constructed using beam search
        hidden_size = self.bert_config.hidden_size
        with self.graph.as_default():
            config = self.bert_config

            target_sequence = tf.squeeze(self.decode_seq, axis=1)
            target_sequence = self.trunct(target_sequence)
            target_length = self.decode_length
            target_seg_ids = tf.zeros_like(target_sequence, dtype=tf.int32, name='target_seg_ids_infer_2')
            tgt_mask = tf.sequence_mask(target_length,
                                        maxlen=tf.shape(target_sequence)[1],
                                        dtype=tf.float32)  # [b, q']

            with tf.variable_scope('bert', reuse=True):
                with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                    # Perform embedding lookup on the target word ids.
                    (tgt_embed, _) = embedding_lookup(
                        input_ids=target_sequence,
                        vocab_size=config.vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name='word_embeddings',
                        use_one_hot_embeddings=False)

                    # Add positional embeddings and token type embeddings, then layer
                    # normalize and perform dropout.
                    tgt_embed = embedding_postprocessor(
                        input_tensor=tgt_embed,
                        use_token_type=True,
                        token_type_ids=target_seg_ids,
                        token_type_vocab_size=config.type_vocab_size,
                        token_type_embedding_name='token_type_embeddings',
                        use_position_embeddings=True,
                        position_embedding_name='position_embeddings',
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings,
                        dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope('decoder', reuse=True):
                masked_tgt_embed = tgt_embed * tf.expand_dims(tgt_mask, -1)
                second_dec_attn_bias = attention_bias(tf.shape(masked_tgt_embed)[1], 'cloze_bias')
                infer_decoder_input = tf.pad(masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left
                all_att_weights, decoder_output = transformer_decoder(infer_decoder_input,
                                                                      self.enc_output,
                                                                      second_dec_attn_bias,
                                                                      self.enc_attn_bias,
                                                                      self.hps,
                                                                      scope='t_decoder')
                # [b, l_t, e] => [b*l_t, v]
                seq_len = tf.shape(decoder_output)[1]
                decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
                second_logits = tf.matmul(decoder_output, self.decoder_weights, False, True)  # (b*l_t, v)
                vocab_probs = tf.nn.softmax(second_logits)  # [b * l_t, v]
                vocab_size = len(self.hps.vocab)
                with tf.variable_scope('copy', reuse=tf.AUTO_REUSE):
                    infer_sent_rep = tf.reshape(tf.tile(self.infer_sentence_rep, [1, seq_len, 1]),
                                                [-1, self.hps.hidden_size])
                    copy_rep = tf.concat([infer_sent_rep, decoder_output], axis=-1)  # [b * l_t, 2 * e]
                    logits = calculate_final_logits(copy_rep, all_att_weights, vocab_probs,
                                                    self.input_ids_oo, self.max_out_oovs, self.input_mask,
                                                    vocab_size, self.infer_tiled_len)  # [b * l_t, v + v']
                second_log_prob = tf.log(logits)
                # (b, l_t, v)
                extend_vocab_size = tf.add(tf.constant(vocab_size), self.max_out_oovs)
                second_log_prob = tf.reshape(second_log_prob, [-1, tf.shape(target_sequence)[1], extend_vocab_size])
                second_log_id = tf.argmax(second_log_prob, axis=-1)  # (b, l_t)
        return second_log_id

    def decode_infer_sent(self):
        # stage 2, sentence level inference using decoded sequence
        # l_t = decode sequence length
        # during infer, following graph are constructed using beam search
        hidden_size = self.bert_config.hidden_size
        with self.graph.as_default():
            config = self.bert_config
            target_sequence = tf.squeeze(self.decode_seq, axis=1)
            target_sequence = self.trunct(target_sequence)
            target_length = self.decode_length
            target_seg_ids = tf.zeros_like(target_sequence, dtype=tf.int32, name='target_seg_ids_infer_2')
            tgt_mask = tf.sequence_mask(target_length,
                                        maxlen=tf.shape(target_sequence)[1],
                                        dtype=tf.float32)  # [b, q']

            with tf.variable_scope('bert', reuse=True):
                with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                    # Perform embedding lookup on the target word ids.
                    (tgt_embed, _) = embedding_lookup(
                        input_ids=target_sequence,
                        vocab_size=config.vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name='word_embeddings',
                        use_one_hot_embeddings=False)

                    # Add positional embeddings and token type embeddings, then layer
                    # normalize and perform dropout.
                    tgt_embed = embedding_postprocessor(
                        input_tensor=tgt_embed,
                        use_token_type=True,
                        token_type_ids=target_seg_ids,
                        token_type_vocab_size=config.type_vocab_size,
                        token_type_embedding_name='token_type_embeddings',
                        use_position_embeddings=True,
                        position_embedding_name='position_embeddings',
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings,
                        dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope('decoder', reuse=True):
                masked_tgt_embed = tgt_embed * tf.expand_dims(tgt_mask, -1)
                infer_decoder_input = tf.pad(masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left
                all_att_weights, decoder_output = transformer_decoder(infer_decoder_input,
                                                                      self.enc_output,
                                                                      self.sent_level_attn_bias,
                                                                      self.enc_attn_bias,
                                                                      self.hps,
                                                                      scope='t_decoder')
                # [b, l_t, e] => [b*l_t, v]
                seq_len = tf.shape(decoder_output)[1]
                decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
                second_logits = tf.matmul(decoder_output, self.decoder_weights, False, True)  # (b*l_t, v)
                vocab_probs = tf.nn.softmax(second_logits)  # [b * l_t, v]
                vocab_size = len(self.hps.vocab)
                with tf.variable_scope('copy', reuse=tf.AUTO_REUSE):
                    infer_sent_rep = tf.reshape(tf.tile(self.infer_sentence_rep, [1, seq_len, 1]),
                                                [-1, self.hps.hidden_size])
                    copy_rep = tf.concat([infer_sent_rep, decoder_output], axis=-1)  # [b * l_t, 2 * e]
                    logits = calculate_final_logits(copy_rep, all_att_weights, vocab_probs,
                                                    self.input_ids_oo, self.max_out_oovs, self.input_mask,
                                                    vocab_size, self.infer_tiled_len)  # [b * l_t, v + v']
                second_log_prob = tf.log(logits)
                # (b, l_t, v)
                extend_vocab_size = tf.add(tf.constant(vocab_size), self.max_out_oovs)
                second_log_prob = tf.reshape(second_log_prob, [-1, tf.shape(target_sequence)[1], extend_vocab_size])
                second_log_id = tf.argmax(second_log_prob, axis=-1)  # (b, l_t)
        return second_log_id

    def _make_feed_dict_rl(self, batch):
        feed_dict = {}
        for k in self.input_keys_rl:
            feed_dict[self.tensor_list[k]] = getattr(batch, k)
        return feed_dict

    def run_reward(self, batch):
        to_return = {}
        for k in self.output_keys_rl:
            to_return[k] = self.tensor_list[k]
        feed_dict = self._make_feed_dict_rl(batch)
        return self.gSess_train.run(to_return, feed_dict)

    def _make_input_key(self):
        """The key name should be equal with property name in Batch class"""
        self.tensor_list = {'source_ids': self.input_ids,
                            'source_len': self.input_len,
                            'source_ids_oo': self.input_ids_oo,
                            'source_seg_ids': self.segment_ids,
                            'target_ids': self.output_ids,
                            'target_ids_oo': self.output_label,
                            'max_oov_num': self.max_out_oovs,
                            'target_len': self.output_len,
                            'loss': self.loss,
                            'logits': self.logits,
                            'encoder_output': self.enc_output,
                            'sentence_rep': self.infer_sentence_rep,
                            'tiled_sentence_rep': self.tiled_sentence_rep,
                            'decode_seq': self.decode_seq,
                            'decode_length': self.decode_length,
                            'sent_level_attn_bias': self.sent_level_attn_bias,
                            'time_step': self.time_step,
                            'reward': self.reward,
                            'pred_ids': self.pred_ids
                            }
        if self.is_training:
            self.tensor_list.update({
                'train_opt': self.train_op,
                'grad_accum': self.accum_op,
                'summaries': self._summaries
            })
        self.input_keys_infer = ['source_ids', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num',
                                 'encoder_output', 'tiled_sentence_rep']
        self.input_keys_infer_stage_2 = ['source_ids', 'source_ids_oo', 'source_len', 'decode_seq', 'decode_length',
                                         'encoder_output', 'sent_level_attn_bias', 'time_step', 'max_oov_num',
                                         'sentence_rep']
        self.input_keys = ['source_ids', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num', 'target_ids',
                           'target_len', 'sent_level_attn_bias', 'target_ids_oo', 'reward']
        self.input_keys_rl = ['source_ids', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num',
                              'target_ids', 'target_len', 'sent_level_attn_bias', 'target_ids_oo']
        self.output_keys_rl = ['pred_ids']
        self.output_keys_train = ['loss', 'train_opt', 'summaries']
        self.output_keys_grad_accum = ['grad_accum']
        self.output_keys_dev = ['loss', 'logits']
