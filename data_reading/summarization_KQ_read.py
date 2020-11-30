import logging
import pickle
from model_pools.model_utils.copy_mechanism import copy_mechanism_preprocess


class SummarizeKQProcessor(object):
    def __init__(self, config=None):
        self.config = config
        
        
    def get_train_examples(self, data_dir, train_file):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, dev_file):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, test_file):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    #@staticmethod
    def convert_example_to_feature(self, example, tokenizer, config):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # key
        key_str = example.key
        key_str_pieces = []
        mem_segment_ids = []
        for t in key_str.strip().split():
            t_tok = tokenizer.tokenize(t)
            if len(t_tok) < 1:
                continue
            key_str_pieces.append(t_tok[0])
            mem_segment_ids.append(0)
        #print(key_str_pieces)
        
        # article
        article_pieces = tokenizer.tokenize(example.article)
        article_tokens = []
        segment_ids = []
        article_tokens.append(config.cls)
        segment_ids.append(0)
        for token in article_pieces:
            article_tokens.append(token)
            segment_ids.append(0)
        article_tokens.append(config.sep)
        segment_ids.append(0)

        # summary
        summary_tokens = tokenizer.tokenize(example.true_summary)
        summary_tokens.append(config.pad)

        input_ids = tokenizer.convert_tokens_to_ids(article_tokens)
        topic_words_ids = tokenizer.convert_tokens_to_ids(key_str_pieces)
        summary_ids = tokenizer.convert_tokens_to_ids(summary_tokens)
        summary_len = len(summary_ids)
        input_len = len(input_ids)
        topic_words_lens = len(topic_words_ids)
        input_ids_oo, src_oovs, summary_ids_oo = copy_mechanism_preprocess(article_tokens, summary_tokens,
                                                                           config, tokenizer.vocab)
        assert len(input_ids) == len(input_ids_oo)
        assert len(summary_ids) == len(summary_ids_oo)

        """logging.info('*** Example ***')
        logging.info('guid: %s' % example.guid)
        logging.info('article: %s' % (' '.join(article_tokens)))
        logging.info('summary: %s' % (' '.join(summary_tokens)))
        logging.info('input_ids: %s' % (' '.join([str(x) for x in input_ids])))
        logging.info('input_ids_oo: %s' % (' '.join([str(x) for x in input_ids_oo])))
        logging.info('summary_ids: %s' % (' '.join([str(x) for x in summary_ids])))
        logging.info('summary_ids_oo: %s' % (' '.join([str(x) for x in summary_ids_oo])))
        logging.info('src oovs: %s' % (' '.join([str(x) for x in src_oovs])))
        logging.info('input_len: %d' % input_len)
        logging.info('summary_len: %d' % summary_len)
        logging.info('segment_ids: %s' % (' '.join([str(x) for x in segment_ids])))"""

        feature = {
            'origin_sample': example,
            'topic_words_ids': topic_words_ids,
            'topic_words_lens': topic_words_lens,
            'article_ids': input_ids,
            'mem_segment_ids': mem_segment_ids,
            'article_ids_oo': input_ids_oo,
            'article_lens': input_len,
            'article_seg_ids': segment_ids,
            'summary_ids': summary_ids,
            'summary_ids_oo': summary_ids_oo,
            'summary_lens': summary_len,
            'src_oovs': src_oovs
        }
        return feature

    @staticmethod
    def filter_examples(examples):
        """# See https://github.com/abisee/pointer-generator/issues/1"""
        return [example for example in examples if len(example.article) != 0]

    @staticmethod
    def log_statistics(examples):
        logging.info('Data Samples: {}'.format(len(examples)))
        max_target_len = max(len(sample.true_summary.split()) for sample in examples)
        max_source_len = max(len(sample.article.split()) for sample in examples)
        mean_target_len = sum(len(sample.true_summary.split()) for sample in examples) / len(examples)
        mean_source_len = sum(len(sample.article.split()) for sample in examples) / len(examples)
        logging.info('Max article length is {}'.format(max_source_len))
        logging.info('Max summary length is {}'.format(max_target_len))
        logging.info('Mean article length is {}'.format(mean_source_len))
        logging.info('Mean summary length is {}'.format(mean_target_len))
