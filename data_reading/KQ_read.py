import json
import os
import pickle
from data_reading.summarization_KQ_read import SummarizeKQProcessor


class InputExample(object):
    def __init__(self, guid, key, query, title, article, click):
        self.guid = guid
        self.key = key
        self.query = query
        self.true_summary = title
        self.summary = title
        self.article = article
        self.click = click

class NewsKQProcessor(SummarizeKQProcessor):
    def get_test_examples(self, data_dir, test_file):
        return self._create_examples(os.path.join(data_dir, test_file))

    def get_dev_examples(self, data_dir, dev_file):
        return self._create_examples(os.path.join(data_dir, dev_file))

    def get_train_examples(self, data_dir, train_file):
        return self._create_examples(os.path.join(data_dir, train_file))

    @staticmethod
    def abstract2sents(abstract: str):
        return [abstract]

    @staticmethod
    def _create_examples(file_path):
        examples = []
        guid = 0
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            data = json.loads(line.strip())
            examples.append(InputExample(guid, data['key'], data['query'], data['title'], data['article'], data['click']))
            guid += 1
        return examples