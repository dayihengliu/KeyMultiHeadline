import tensorflow as tf

from model_pools import modeling
from model_pools.base_model import BaseModel
from model_pools.model_utils.layer import attention_bias, smoothed_softmax_cross_entropy
from model_pools.model_utils.module import transformer_decoder
from model_pools.modeling import embedding_lookup, embedding_postprocessor


# noinspection PyAttributeOutsideInit
class BertTwoStageSummarizer(BaseModel):
    """
    BertSummarizer + TwoStageDecoder.
    """

    def __init__(self, bert_config, batcher, hps):
        super(BertTwoStageSummarizer, self).__init__(hps, bert_config, batcher)

    def build_graph(self):
        with self.graph.as_default():
            self._build_summarization_model()

    def _add_placeholders(self):
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
        # decode sequence for second stage inference
        self.decode_seq = tf.placeholder(tf.int32, [None, None], name='decoded_seq')
        self.decode_length = tf.placeholder(tf.int32, [None], name='decoded_length')
        # encoder output for inference
        self.enc_output = tf.placeholder(tf.float32, [None, None, self.hps.hidden_size], name='enc_output')

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
            self.dec_attn_bias = attention_bias(tf.shape(self.masked_out_embed)[1], 'causal')
            self.decoder_input = tf.pad(self.masked_out_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left
            self.all_att_weights, self.decoder_output = transformer_decoder(self.decoder_input, self.encoder_output,
                                                                            self.dec_attn_bias, self.enc_attn_bias,
                                                                            self.hps)
            # [b, l_t, e] => [b*l_t, v]
            self.decoder_output = tf.reshape(self.decoder_output, [-1, hidden_size])
            self.logits = tf.matmul(self.decoder_output, self.decoder_weights, False, True)  # (b*l_t, v)
        with tf.variable_scope('second_stage_decoder'):
            self.second_dec_attn_bias = attention_bias(tf.shape(self.masked_out_embed)[1], 'cloze_bias')
            self.all_att_weights, self.decoder_output = transformer_decoder(self.decoder_input, self.encoder_output,
                                                                            self.second_dec_attn_bias,
                                                                            self.enc_attn_bias,
                                                                            self.hps)
            # [b, l_t, e] => [b*l_t, v]
            self.decoder_output = tf.reshape(self.decoder_output, [-1, hidden_size])
            self.second_logits = tf.matmul(self.decoder_output, self.decoder_weights, False, True)  # (b*l_t, v)

        with tf.variable_scope('loss'):
            self.ce = smoothed_softmax_cross_entropy(
                self.logits,
                self.output_ids,
                self.hps.label_smoothing,
                True
            )

            self.ce = tf.reshape(self.ce, tf.shape(self.output_ids))  # [b, l_t]

            self.first_loss = tf.reduce_sum(self.ce * self.output_mask) / tf.reduce_sum(self.output_mask)  # scalar

            self.second_ce = smoothed_softmax_cross_entropy(
                self.second_logits,
                self.output_ids,
                self.hps.label_smoothing,
                True
            )

            self.second_ce = tf.reshape(self.second_ce, tf.shape(self.output_ids))  # [b, l_t]

            self.second_loss = tf.reduce_sum(self.second_ce * self.output_mask) / tf.reduce_sum(
                self.output_mask)  # scalar

            self.loss = self.first_loss + self.second_loss
            tf.summary.scalar('first_loss', self.first_loss)
            tf.summary.scalar('second_loss', self.second_loss)
            tf.summary.scalar('loss', self.loss)

    def decode_infer(self, inputs, state):
        # state['enc']: [b * beam, l_s, e]  ,   state['dec']: [b * beam, q', e]
        # q' = previous decode output length
        # during infer, following graph are constructed using beam search
        with self.graph.as_default():
            config = self.bert_config

            target_sequence = inputs['target']
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

                # feed last decoded token embedding of input, with state['decoder'] to hold all state
                infer_decoder_input = decoder_input[:, -1:, :]
                infer_dec_attn_bias = dec_attn_bias[:, :, -1:, :]

                all_att_weights, decoder_output, decoder_state = transformer_decoder(infer_decoder_input,
                                                                                     self.enc_output,
                                                                                     infer_dec_attn_bias,
                                                                                     self.enc_attn_bias,
                                                                                     self.hps,
                                                                                     state=state['decoder'])
                decoder_output = decoder_output[:, -1, :]  # [b * beam, e]
                logits = tf.matmul(decoder_output, self.decoder_weights, False, True)  # [b * beam, v]
                log_prob = tf.nn.log_softmax(logits)
        return log_prob, {'encoder': state['encoder'], 'decoder': decoder_state}

    def decode_infer_2(self):
        # stage 2, inference using decoded sequence
        # l_t = decode sequence length
        # during infer, following graph are constructed using beam search
        hidden_size = self.bert_config.hidden_size
        with self.graph.as_default():
            config = self.bert_config

            target_sequence = self.decode_seq
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

            with tf.variable_scope('second_stage_decoder', reuse=tf.AUTO_REUSE):
                masked_tgt_embed = tgt_embed * tf.expand_dims(tgt_mask, -1)
                second_dec_attn_bias = attention_bias(tf.shape(masked_tgt_embed)[1], 'cloze_bias')
                infer_decoder_input = tf.pad(masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left
                all_att_weights, decoder_output = transformer_decoder(infer_decoder_input,
                                                                      self.enc_output,
                                                                      second_dec_attn_bias,
                                                                      self.enc_attn_bias,
                                                                      self.hps)
                # [b, l_t, e] => [b*l_t, v]
                decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
                second_logits = tf.matmul(decoder_output, self.decoder_weights, False, True)  # (b*l_t, v)
                # (b, l_t, v)
                second_logits = tf.reshape(second_logits, [-1, tf.shape(target_sequence)[1], config.vocab_size])
                second_log_prob = tf.nn.log_softmax(second_logits)
                second_log_id = tf.argmax(second_log_prob, axis=-1)  # (b, l_t)
        return second_log_id

    def _make_input_key(self):
        """The key name should be equal with property name in Batch class"""
        self.tensor_list = {'source_ids': self.input_ids,
                            'source_len': self.input_len,
                            'source_seg_ids': self.segment_ids,
                            'target_ids': self.output_ids,
                            'target_len': self.output_len,
                            'loss': self.loss,
                            'logits': self.logits,
                            'encoder_output': self.enc_output,
                            'decode_seq': self.decode_seq,
                            'decode_length': self.decode_length
                            }
        if self.is_training:
            self.tensor_list.update({
                'train_opt': self.train_op,
                'grad_accum': self.accum_op,
                'summaries': self._summaries
            })
        self.input_keys_infer = ['source_ids', 'source_len', 'source_seg_ids', 'encoder_output']
        self.input_keys_infer_stage_2 = ['source_ids', 'source_len', 'decode_seq', 'decode_length', 'encoder_output']
        self.input_keys = ['source_ids', 'source_len', 'source_seg_ids', 'target_ids', 'target_len']
        self.output_keys_train = ['loss', 'train_opt', 'summaries']
        self.output_keys_grad_accum = ['grad_accum']
        self.output_keys_dev = ['loss', 'logits']
