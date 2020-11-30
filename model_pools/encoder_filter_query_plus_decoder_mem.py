import tensorflow as tf

from model_pools import modeling
from model_pools.base_model import BaseModel
from model_pools.model_utils.layer import *
from model_pools.model_utils.module import smooth_cross_entropy, transformer_decoder, transformer_decoder_three
from model_pools.modeling import embedding_lookup, embedding_postprocessor
from model_pools.model_utils.attention import multihead_attention
from utils.copy_utils import calculate_final_logits, tf_trunct



def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu
    return _assign

def sum_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# noinspection PyAttributeOutsideInit
class QueryEncoderFilterPlusDecoderMem(BaseModel):
    """
    Based on BertSummarizerCopy, change some model settings, 41.9 on CNN/DM
    MultiGPU version (training only)
    """

    def __init__(self, bert_config, batcher, hps):
        super(QueryEncoderFilterPlusDecoderMem, self).__init__(hps, bert_config, batcher)

    def build_graph(self):
        with self.graph.as_default():
            self._build_summarization_model()

    def _add_placeholders(self):
        self.batch_size = self.hps.train_batch_size if self.is_training else self.hps.eval_batch_size
        self.batch_size = int(self.batch_size / self.hps.n_gpu)
        
        self.input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')  # [b, l_s]
        #self.topic_ids = tf.placeholder(tf.int32, [None], name='topic_ids')
        self.input_len = tf.placeholder(tf.int32, [None], name='input_len')  # [b]
        self.segment_ids = tf.placeholder(tf.int32, [None, None], name='segment_ids')
        self.output_ids = tf.placeholder(tf.int32, [None, None], name='output_ids')  # [b, l_t], not use
        self.output_len = tf.placeholder(tf.int32, [None], name='output_len')  # [b]
        self.topic_words_ids = tf.placeholder(tf.int32, [None, None], name='topic_words_ids')
        self.topic_words_len = tf.placeholder(tf.int32, [None], name='topic_words_len')
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
        self.topic_words_mask = tf.sequence_mask(self.topic_words_len,
                                            maxlen=tf.shape(self.topic_words_ids)[1],
                                            dtype=tf.float32)  # [b, l_t]
        self.out_segment_ids = tf.zeros_like(self.output_label, dtype=tf.int32, name='out_segment_ids')
        self.mem_segment_ids =  tf.placeholder(tf.int32, [None, None], name='mem_segment_ids')
        self.tiled_len = tf.shape(self.output_label)[1]
        # encoder output for inference
        self.enc_output = tf.placeholder(tf.float32, [None, None, self.hps.hidden_size], name='enc_output')
        self.topic_memory = tf.placeholder(tf.float32, [None, None, self.hps.hidden_size], name='topic_memory')

    def _n_gpu_split_placeholders(self, n):
        # index_ids
        #self.topic_ids_ngpu = tf.split(self.topic_ids, n)
        self.input_ids_ngpu = tf.split(self.input_ids, n)
        self.topic_words_ids_ngpu = tf.split(self.topic_words_ids, n)
        self.output_ids_ngpu = tf.split(self.output_ids, n)
        # lens
        self.input_len_ngpu = tf.split(self.input_len, n)
        self.topic_words_len_ngpu = tf.split(self.topic_words_len, n)
        self.output_len_ngpu = tf.split(self.output_len, n)
        # copies
        self.input_ids_oo_ngpu = tf.split(self.input_ids_oo, n)
        self.output_label_ngpu = tf.split(self.output_label, n)
        # mask
        self.input_mask_ngpu = tf.split(self.input_mask, n)
        self.topic_words_mask_ngpu =tf.split(self.topic_words_mask, n)
        self.output_mask_ngpu = tf.split(self.output_mask, n)
        # segment
        self.segment_ids_ngpu = tf.split(self.segment_ids, n)
        self.out_segment_ids_ngpu = tf.split(self.out_segment_ids, n)
        self.mem_segment_ids_ngpu = tf.split(self.mem_segment_ids, n)

        
    def _build_summarization_model(self):
        is_training = self.is_training
        config = self.bert_config

        gpu_pred_ids = []
        gpu_logits = []
        gpu_train_encoded = []
        gpu_loss = []
        gpu_out_embed = []
        gpu_grads = []
        self._add_placeholders()
        self._n_gpu_split_placeholders(self.hps.n_gpu)
        
        for i in range(self.hps.n_gpu):
            do_reuse = True if i > 0 else None
            with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
                
                '''Creates a classification model.'''
                model = modeling.BertModel(
                    config=self.bert_config,
                    is_training=is_training,
                    input_ids=self.input_ids_ngpu[i],
                    input_mask=self.input_mask_ngpu[i],
                    token_type_ids=self.segment_ids_ngpu[i],
                    use_one_hot_embeddings=self.hps.use_tpu)  # use_one_hot_embeddings=Flags.tpu ?
                encoder_output = model.get_sequence_output()  # [b, l_s, h]
                self.enc_attn_bias = attention_bias(self.input_mask_ngpu[i], 'masking')
                
                hidden_size = encoder_output.shape[2].value
                encoder_out_length = tf.shape(encoder_output)[1]
                
                """Get topic word memory"""
                out_dict_size = len(self.hps.vocab_out)
                ## for topic word memory
                with tf.variable_scope('bert', reuse=True):
                    with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                        # Perform embedding lookup on the target word ids.
                        (topic_word_memory, _) = embedding_lookup(
                            input_ids=self.topic_words_ids_ngpu[i],  # here the embedding input of decoder have to be output_ids
                            vocab_size=out_dict_size,  # decode dictionary modified
                            embedding_size=config.hidden_size,
                            initializer_range=config.initializer_range,
                            word_embedding_name='word_embeddings',
                            use_one_hot_embeddings=False)
                        # Add positional embeddings and token type embeddings, then layer
                        # normalize and perform dropout.
                        self.topic_word_memory = embedding_postprocessor(
                            input_tensor=topic_word_memory,
                            use_token_type=True,
                            token_type_ids=self.mem_segment_ids_ngpu[i],
                            token_type_vocab_size=config.type_vocab_size,
                            token_type_embedding_name='token_type_embeddings',
                            use_position_embeddings=False,
                            position_embedding_name='position_embeddings',
                            initializer_range=config.initializer_range,
                            max_position_embeddings=config.max_position_embeddings,
                            dropout_prob=config.hidden_dropout_prob)
                self.topic_attn_bias = attention_bias(self.topic_words_mask_ngpu[i], 'masking')
                
                #print('topic_word_memory!!!!', self.topic_word_memory)
                #print('encoder_output_topic_emb!!!!', encoder_output_topic_emb)
                #print('self.topic_attn_bias!!!!', self.topic_attn_bias)
                #print('self.enc_attn_bias!!!!', self.enc_attn_bias)
                """encoder_topic_attention"""
                with tf.variable_scope("encoder_topic_attention"):
                    params = self.hps
                    y = multihead_attention(
                        layer_process(encoder_output, params.layer_preprocess),
                        self.topic_word_memory,
                        self.topic_attn_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout
                    )
                self.encoder_output = y["outputs"] + encoder_output
                
                    
                """decoder"""
                with tf.variable_scope('bert', reuse=True):
                    with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                        # Perform embedding lookup on the target word ids.
                        (self.out_embed, self.bert_embeddings) = embedding_lookup(
                            input_ids=self.output_ids_ngpu[i],  # here the embedding input of decoder have to be output_ids
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
                            token_type_ids=self.out_segment_ids_ngpu[i],
                            token_type_vocab_size=config.type_vocab_size,
                            token_type_embedding_name='token_type_embeddings',
                            use_position_embeddings=True,
                            position_embedding_name='position_embeddings',
                            initializer_range=config.initializer_range,
                            max_position_embeddings=config.max_position_embeddings,
                            dropout_prob=config.hidden_dropout_prob)

                with tf.variable_scope('decode'):
                    self.decoder_weights = self.bert_embeddings
                    self.masked_out_embed = self.out_embed * tf.expand_dims(self.output_mask_ngpu[i], -1)
                    self.dec_attn_bias = attention_bias(tf.shape(self.masked_out_embed)[1], 'causal')
                    self.decoder_input = tf.pad(self.masked_out_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left
                    self.all_att_weights, _, self.decoder_output = transformer_decoder_three(self.decoder_input,
                                                                                    self.encoder_output,
                                                                                    self.topic_word_memory,
                                                                                    self.dec_attn_bias,
                                                                                    self.enc_attn_bias,
                                                                                    self.topic_attn_bias,
                                                                                    self.hps)
                    # [b, l_t, e] => [b*l_t, v]
                    self.decoder_output = tf.reshape(self.decoder_output, [-1, hidden_size])
                    self.vocab_logits = tf.matmul(self.decoder_output, self.decoder_weights, False, True)  # (b * l_t, v)
                    self.vocab_probs = tf.nn.softmax(self.vocab_logits)  # [b * l_t, v]
                    # vocab_size = len(self.hps.vocab)
                    with tf.variable_scope('copy'):
                        self.single_logits = calculate_final_logits(self.decoder_output, self.all_att_weights, self.vocab_probs,
                                                             self.input_ids_oo_ngpu[i], self.max_out_oovs, self.input_mask_ngpu[i],
                                                             out_dict_size,
                                                             self.tiled_len)  # [b * l_t, v + v']
                        self.single_pred_ids = tf.reshape(tf.argmax(self.single_logits, axis=-1), [self.batch_size, -1])
                
                with tf.variable_scope('loss'):
                    self.single_ce = smooth_cross_entropy(
                        self.single_logits,
                        self.output_label_ngpu[i],
                        self.hps.label_smoothing)

                    self.single_ce = tf.reshape(self.single_ce, tf.shape(self.output_label_ngpu[i]))  # [b, l_t]

                    self.single_loss = tf.reduce_sum(self.single_ce * self.output_mask_ngpu[i]) / tf.reduce_sum(self.output_mask_ngpu[i])  # scalar
            
                gpu_pred_ids.append(self.single_pred_ids)
                gpu_logits.append(self.single_logits)
                gpu_train_encoded.append(self.encoder_output)
                gpu_loss.append(self.single_loss)
                gpu_out_embed.append(self.out_embed)
                params = tf.trainable_variables()
                grads = tf.gradients(self.single_loss, params)
                grads = list(zip(grads, params))
                gpu_grads.append(grads)
                #gpu_ops.append([loss, logits])

        self.pred_ids = tf.concat(gpu_pred_ids, axis=0)
        self.logits = tf.concat(gpu_logits, axis=0)
        self.loss = tf.reduce_mean(gpu_loss)
        self.encoder_output = tf.concat(gpu_train_encoded, axis=0)
        self.out_embed = tf.concat(gpu_out_embed, axis=0)
        # end for
        grads = sum_grads(gpu_grads)
        grads = [g for g, p in grads]
        self.total_gradient = grads

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
            with tf.variable_scope('bert', reuse=True):
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

                ret = transformer_decoder_three(infer_decoder_input,
                                          self.enc_output,
                                          self.topic_memory,
                                          infer_dec_attn_bias,
                                          self.enc_attn_bias,
                                          self.topic_attn_bias,
                                          self.hps,
                                          state=state['decoder'])

                all_att_weights, _, decoder_output, decoder_state = ret
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
                            'topic_words_ids': self.topic_words_ids,
                            'topic_words_len': self.topic_words_len,
                            'mem_segment_ids': self.mem_segment_ids,
                            'loss': self.loss,
                            'logits': self.logits,
                            'encoder_output': self.enc_output,
                            'topic_word_memory': self.topic_memory,
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
        self.input_keys_infer = ['source_ids', 'topic_words_ids', 'topic_words_len', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num', 'encoder_output', 'topic_word_memory', 'mem_segment_ids']
        self.input_keys = ['source_ids', 'topic_words_ids', 'topic_words_len', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num', 'target_ids', 'target_ids_oo', 'target_len', 'mem_segment_ids']
        self.output_keys_train = ['loss', 'train_opt', 'summaries', 'pred_ids', 'train_encoded', 'logits', 'out_embed',
                                  'all_att_weights']
        self.output_keys_grad_accum = ['grad_accum']
        self.output_keys_dev = ['loss', 'logits']
