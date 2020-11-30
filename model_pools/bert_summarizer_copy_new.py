import tensorflow as tf

from model_pools import modeling
from model_pools.base_model import BaseModel
from model_pools.model_utils.layer import attention_bias
from model_pools.model_utils.module import smooth_cross_entropy, transformer_decoder
from model_pools.modeling import embedding_lookup, embedding_postprocessor
from utils.copy_utils import calculate_final_logits, tf_trunct


# noinspection PyAttributeOutsideInit
class BertSummarizerCopyNew(BaseModel):
    """
    Based on BertSummarizerCopy, change some model settings, 41.9 on CNN/DM
    """

    def __init__(self, bert_config, batcher, hps):
        super(BertSummarizerCopyNew, self).__init__(hps, bert_config, batcher)

    def build_graph(self):
        with self.graph.as_default():
            self._build_summarization_model()

    def _add_placeholders(self):
        self.batch_size = self.hps.train_batch_size if self.is_training else self.hps.eval_batch_size
        self.input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')  # [b, l_s]
        self.input_len = tf.placeholder(tf.int32, [None], name='input_len')  # [b]
        self.segment_ids = tf.placeholder(tf.int32, [None, None], name='segment_ids')
        self.output_ids = tf.placeholder(tf.int32, [None, None], name='output_ids')  # [b, l_t], not use
        self.output_len = tf.placeholder(tf.int32, [None], name='output_len')  # [b]

        # copy related placeholder
        self.output_label = tf.placeholder(tf.int32, [None, None], name='output_label')  # [b, l_t], output_ids_oo
        self.max_out_oovs = tf.placeholder(tf.int32, [], name='max_out_oovs')  # []
        self.input_ids_oo = tf.placeholder(tf.int32, [None, None], name='input_ids_oo')  # [b, l_s]

        self.input_mask = tf.sequence_mask(self.input_len,
                                           maxlen=tf.shape(self.input_ids)[1],
                                           dtype=tf.float32)  # [b, l_s]
        self.output_mask = tf.sequence_mask(self.output_len,
                                            maxlen=tf.shape(self.output_label)[1],
                                            dtype=tf.float32)  # [b, l_t]
        self.out_segment_ids = tf.zeros_like(self.output_label, dtype=tf.int32, name='out_segment_ids')
        self.tiled_len = tf.shape(self.output_label)[1]
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

        out_dict_size = len(self.hps.vocab_out)
        with tf.variable_scope('bert-output'):
            with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                # Perform embedding lookup on the target word ids.
                (self.out_embed, self.bert_embeddings) = embedding_lookup(
                    input_ids=self.output_ids,  # here the embedding input of decoder have to be output_ids
                    vocab_size=out_dict_size,  # decode dictionary modified
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

        with tf.variable_scope('decode'):
            self.decoder_weights = self.bert_embeddings
            self.masked_out_embed = self.out_embed * tf.expand_dims(self.output_mask, -1)
            self.dec_attn_bias = attention_bias(tf.shape(self.masked_out_embed)[1], 'causal')
            self.decoder_input = tf.pad(self.masked_out_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left
            self.all_att_weights, self.decoder_output = transformer_decoder(self.decoder_input,
                                                                            self.encoder_output,
                                                                            self.dec_attn_bias,
                                                                            self.enc_attn_bias,
                                                                            self.hps)
            # [b, l_t, e] => [b*l_t, v]
            self.decoder_output = tf.reshape(self.decoder_output, [-1, hidden_size])
            self.vocab_logits = tf.matmul(self.decoder_output, self.decoder_weights, False, True)  # (b * l_t, v)
            self.vocab_probs = tf.nn.softmax(self.vocab_logits)  # [b * l_t, v]
            # vocab_size = len(self.hps.vocab)
            with tf.variable_scope('copy'):
                self.logits = calculate_final_logits(self.decoder_output, self.all_att_weights, self.vocab_probs,
                                                     self.input_ids_oo, self.max_out_oovs, self.input_mask,
                                                     out_dict_size,
                                                     self.tiled_len)  # [b * l_t, v + v']
                self.pred_ids = tf.reshape(tf.argmax(self.logits, axis=-1), [self.batch_size, -1])

        with tf.variable_scope('loss'):
            self.ce = smooth_cross_entropy(
                self.logits,
                self.output_label,
                self.hps.label_smoothing)

            self.ce = tf.reshape(self.ce, tf.shape(self.output_label))  # [b, l_t]

            self.loss = tf.reduce_sum(self.ce * self.output_mask) / tf.reduce_sum(self.output_mask)  # scalar
            tf.summary.scalar('loss', self.loss)

    def decode_infer(self, inputs, state):
        # state['enc']: [b * beam, l_s, e]  ,   state['dec']: [b * beam, q', e]
        # q' = previous decode output length
        # during infer, following graph are constructed using beam search
        with self.graph.as_default():
            config = self.bert_config

            target_sequence = inputs['target']  # [b * beam, q']
            vocab_size = len(self.hps.vocab_out)
            # trunct word idx, change those greater than vocab_size to unkId
            shape = target_sequence.shape
            unkid = self.hps.vocab_out[self.hps.unk]
            # target_sequence = tf_trunct(target_sequence, vocab_size, self.hps.unkId)
            target_sequence = tf_trunct(target_sequence, vocab_size, unkid)
            target_sequence.set_shape(shape)

            target_length = inputs['target_length']
            target_seg_ids = tf.zeros_like(target_sequence, dtype=tf.int32, name='target_seg_ids_infer')
            tgt_mask = tf.sequence_mask(target_length,
                                        maxlen=tf.shape(target_sequence)[1],
                                        dtype=tf.float32)  # [b, q']

            # with tf.variable_scope('bert', reuse=True):
            out_dict_size = len(self.hps.vocab_out)
            with tf.variable_scope('bert-output', reuse=True):
                with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                    # Perform embedding lookup on the target word ids.
                    (tgt_embed, _) = embedding_lookup(
                        input_ids=target_sequence,
                        vocab_size=out_dict_size,  # out vocab size
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

            with tf.variable_scope('decode', reuse=True):
                # [b, q', e]
                masked_tgt_embed = tgt_embed * tf.expand_dims(tgt_mask, -1)
                dec_attn_bias = attention_bias(tf.shape(masked_tgt_embed)[1], "causal")
                decoder_input = tf.pad(masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left

                infer_decoder_input = decoder_input[:, -1:, :]
                infer_dec_attn_bias = dec_attn_bias[:, :, -1:, :]

                ret = transformer_decoder(infer_decoder_input,
                                          self.enc_output,
                                          infer_dec_attn_bias,
                                          self.enc_attn_bias,
                                          self.hps,
                                          state=state['decoder'])

                all_att_weights, decoder_output, decoder_state = ret
                decoder_output = decoder_output[:, -1, :]  # [b * beam, e]
                vocab_logits = tf.matmul(decoder_output, self.decoder_weights, False, True)  # [b * beam, v]
                vocab_probs = tf.nn.softmax(vocab_logits)
                vocab_size = out_dict_size  # out vocabsize
                # we have tiled source_id_oo before feed, so last argument is set to 1
                with tf.variable_scope('copy'):
                    logits = calculate_final_logits(decoder_output, all_att_weights,
                                                    vocab_probs,
                                                    self.input_ids_oo, self.max_out_oovs, self.input_mask, vocab_size,
                                                    tgt_seq_len=1)
                log_prob = tf.log(logits)  # [b * beam, v + v']
        return log_prob, {'encoder': state['encoder'], 'decoder': decoder_state}

    def _make_input_key(self):
        """The key name should be equal with property name in Batch class"""
        self.tensor_list = {'source_ids': self.input_ids,
                            'source_ids_oo': self.input_ids_oo,
                            'source_len': self.input_len,
                            'source_seg_ids': self.segment_ids,
                            'target_ids': self.output_ids,
                            'target_ids_oo': self.output_label,
                            'max_oov_num': self.max_out_oovs,
                            'target_len': self.output_len,
                            'loss': self.loss,
                            'logits': self.logits,
                            'encoder_output': self.enc_output,
                            'pred_ids': self.pred_ids,
                            # debug
                            'train_encoded': self.encoder_output,
                            'out_embed': self.out_embed,
                            'all_att_weights': self.all_att_weights
                            }
        if self.is_training:
            self.tensor_list.update({
                'train_opt': self.train_op,
                'grad_accum': self.accum_op,
                'summaries': self._summaries
            })
        self.input_keys_infer = ['source_ids', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num',
                                 'encoder_output']
        self.input_keys = ['source_ids', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num',
                           'target_ids', 'target_ids_oo', 'target_len']
        self.output_keys_train = ['loss', 'train_opt', 'summaries', 'pred_ids', 'train_encoded', 'logits', 'out_embed',
                                  'all_att_weights']
        self.output_keys_grad_accum = ['grad_accum']
        self.output_keys_dev = ['loss', 'logits']
