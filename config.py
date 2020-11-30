import tensorflow as tf

import tokenization

tf.set_random_seed(111)  # a seed value for randomness

flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    'data_dir', None,
    'The input data dir. Should contain the .tsv files (or other data files) '
    'for the task.')

flags.DEFINE_string(
    'bert_config_file', None,
    'The config json file corresponding to the pre-trained BERT model. '
    'This specifies the model architecture.')

flags.DEFINE_string('vocab_file', None,
                    'The vocabulary file that the BERT model was trained on.')

flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.')

# Model and Task(Dataset)

flags.DEFINE_string(
    'model_name', 'summarize_bert_baseline',
    'Specify the model name.'
)

flags.DEFINE_string('task_name', 'cnn_dm', 'The name of the task to train.')

# Decoder inference parameters
flags.DEFINE_bool(
    'eval_only', False, 'Only evaluate current predicted results.')

flags.DEFINE_bool(
    'write_all_beam', False, 'write beams.')

flags.DEFINE_integer(
    'beam_size', 8, 'Decode beam size.')

flags.DEFINE_integer(
    'top_beams', 1, 'The number of printed beams.')

flags.DEFINE_float(
    'decode_alpha', 0.6, 'Word penalty.')

flags.DEFINE_float(
    'decode_gamma', 0.8, 'Diveres penalty.')

flags.DEFINE_bool(
    'use_beam_search', False, 'Use or do not use beam search during second decoder inference.'
)

flags.DEFINE_bool(
    'use_diverse_beam_search', False, 'Use or do not use diverse beam search.'
)


flags.DEFINE_bool(
    'load_pre_train', False, 'load pretrain model rather than bert'
)

# Topic parameters
flags.DEFINE_integer(
    'num_topic', 0, 'Topic number.')

flags.DEFINE_integer(
    'topic_embedding_size', 0, 'Topic emedding size.')

# Decoder train parameters

flags.DEFINE_integer(
    'num_decoder_layers', 6, 'Decoder layer number.')

flags.DEFINE_integer(
    'num_heads', 8, 'Attention head number.')

flags.DEFINE_integer(
    'attention_key_channels', 0, 'Attention key channel, if 0 use hidden_size.')


flags.DEFINE_integer(
    'test_iterate', 0, 'Iteration of the inference model')

flags.DEFINE_integer(
    'attention_value_channels', 0, 'Attention value channel, if 0 use hidden_size.')

flags.DEFINE_integer(
    'hidden_size', 768, 'Hidden size, should equal to which in bert.')

flags.DEFINE_integer(
    'filter_size', 3072, 'Neural size of FFN.')

flags.DEFINE_float(
    'attention_dropout', 0.1, 'Attention dropout rate.')

flags.DEFINE_float(
    'residual_dropout', 0.1, 'Residual dropout rate.')

flags.DEFINE_float(
    'relu_dropout', 0.1, 'Relu dropout rate.')

flags.DEFINE_float(
    'label_smoothing', 0.1, 'Label smoothing value.')

flags.DEFINE_string(
    'layer_preprocess', 'layer_norm', 'Layer preprocess.')

# Other train specific parameters

flags.DEFINE_integer(
    'evaluate_every_n_step', 1000, 'Evaluate and save model every n steps.')

flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')

flags.DEFINE_float('learning_rate', 5e-5, 'The initial learning rate for Adam.')

flags.DEFINE_float('num_train_epochs', 3.0,
                   'Total number of training epochs to perform.')

flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.')

flags.DEFINE_string(
    'train_file', 'train.txt',
    'the train file')

flags.DEFINE_string(
    'test_file', 'test.txt',
    'the test file')

flags.DEFINE_string(
    'dev_file', 'dev.txt',
    'the dev file')


flags.DEFINE_string(
    'init_checkpoint', None,
    'Initial checkpoint (usually from a pre-trained BERT model).')

flags.DEFINE_integer(
    'accumulate_step', 1,
    'Use gradient accumulation, each `accumulate_step` update the model parameters with accumulated loss')

flags.DEFINE_bool(
    'refine_train_summary', False,
    'Refine summary label in train or not.')

flags.DEFINE_float(
    'rl_lambda', 0.99,
    'RL reward weight.')

# train specific parameters for BertSummarizerDecDraft model
flags.DEFINE_float(
    'start_portion_to_feed_draft', 0.25,
    'Which portion of train step should we start to feed draft instead of ground-truth'
)

flags.DEFINE_integer(
    'draft_feed_freq', 10,
    'frequency to feed draft'
)

flags.DEFINE_float(
    'mask_percentage', 0.15,
    'Percentage to change to mask for refine decoder'
)

flags.DEFINE_float(
    'total_percentage', 0.2,
    'Total percentage loss in refine decoder'
)

# train specific parameters for pre-train process
flags.DEFINE_float(
    'repeat_percentage', 0.15,
    'Percentage to repeat the word during pre-train'
)

flags.DEFINE_float(
    'switch_percentage', 0.15,
    'Percentage to switch the word or phrases during pre-train'
)

# Other parameters

flags.DEFINE_string('mode', 'train', 'must be one of train/dev/test')

flags.DEFINE_list(
    'gpu', [0],
    'Use which GPU to train, `[]` means use CPU.')

flags.DEFINE_bool(
    'debug', False,
    'Run in CPU to debug.')

flags.DEFINE_string(
    'log_file', None, 'Log file, if `None` log to the screen.')

flags.DEFINE_bool(
    'do_lower_case', True,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')

flags.DEFINE_integer(
    'max_seq_length', 400,
    'The maximum total input sequence length after WordPiece tokenizing. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')

flags.DEFINE_integer(
    'max_out_seq_length', 100,
    'The maximum total output sequence length during train and eval.')

flags.DEFINE_integer('eval_batch_size', 8, 'Total batch size for eval.')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or GPU/CPU, not in use but do not delete it.')


def add_special_chars(hps):
    special_words = {
        'pad': '[PAD]',  # pad, as well as eos
        'unk': '[UNK]',
        'cls': '[CLS]',  # cls, as well as bos
        'sep': '[SEP]',  # to separate the sentence part
        'mask': '[MASK]'  # to mask the word during LM train
    }
    tokenizer = tokenization.FullTokenizer(vocab_file=hps.vocab_file, do_lower_case=hps.do_lower_case)
    vocab = tokenizer.vocab
    hps.add_hparam('vocab', vocab)
    hps.add_hparam('inv_vocab', tokenizer.inv_vocab)
    hps.add_hparam('vocab_words', list(tokenizer.inv_vocab.values()))

    hps.add_hparam('vocab_out', vocab)
    hps.add_hparam('inv_vocab_out', tokenizer.inv_vocab)
    hps.add_hparam('vocab_words_out', list(tokenizer.inv_vocab.values()))
    for word_name in special_words:
        word = special_words[word_name]
        if vocab.get(word, -1) < 0:
            raise KeyError('Bert vocab file does not have special word: {}, which is necessary.'.format(word))
        else:
            hps.add_hparam(word_name + 'Id', vocab[word])
            hps.add_hparam(word_name, word)
    assert (hps.padId == 0), 'Pad ID must be 0.'
    return hps


# noinspection PyProtectedMember,PyPep8Naming,PyUnresolvedReferences
def parse_args():
    # Make a hParams hps, containing the values of the hyperparameters that the model needs
    hps = tf.contrib.training.HParams()
    for key, val in FLAGS.__flags.items():  # for each flag
        hps.add_hparam(key, val._value)
    hps = add_special_chars(hps)
    if hps.gpu == ['-1']:  # CPU mode
        hps.gpu = None
    return hps
