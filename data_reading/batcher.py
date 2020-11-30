import logging
import queue
import random
import time
from random import shuffle
from threading import Thread
from typing import Iterable

import numpy as np

import tokenization
from data_reading import processors
from utils.data_utils import refine_train_summary
from utils.utils import init_sentence_level_info


class Sample(object):
    """Class representing a train/dev/test sample."""

    def __init__(self, sample, params, mode):
        self.params = params
        self.mode = mode
        
        self.original_data = sample['origin_sample']

        # Process the source sequence
        self.source_ids = sample['article_ids']
        self.source_len = sample['article_lens']
        self.source_ids_oo = sample['article_ids_oo']
        self.source_seg_ids = sample['article_seg_ids']
        self.source_oovs = sample['src_oovs']
        self.source_oov_num = len(sample['src_oovs'])
        #print('sample.keys()!!!!', sample.keys())
        if 'topic_ids' in sample:
            #print('!!topic_ids in sample')
            self.topic_ids = sample['topic_ids']
        if 'topic_words_ids' in sample:
            self.topic_words_ids = sample['topic_words_ids']
        if 'mem_segment_ids' in sample:
            self.mem_segment_ids = sample['mem_segment_ids']
        if 'topic_words_lens' in sample:
            self.topic_words_lens = sample['topic_words_lens']
            
        # Process the target sequence
        self.target_ids = sample['summary_ids']
        self.target_ids_oo = sample['summary_ids_oo']
        self.target_len = sample['summary_lens']


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
class Batch(object):
    """Class representing a minibatch of train samples."""

    def __init__(self, sample_list: Iterable[Sample], true_num, params, mode):
        """Turns the sample_list into a Batch object."""
        self.true_num = true_num
        self.params = params
        self.period_id = self.params.vocab['.']
        self.mode = mode
        self.max_src_len = params.max_seq_length
        self.max_tgt_len = params.max_out_seq_length
        self.pad_id = params.padId
        self.init_encoder_seq(sample_list)  # initialize the input to the encoder
        self.init_decoder_seq(sample_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(sample_list)  # store the original strings

    def init_encoder_seq(self, sample_list):
        # group
        self.source_ids = [ex.source_ids for ex in sample_list]
        self.source_len = [ex.source_len for ex in sample_list]
        if hasattr(sample_list[0], 'topic_words_lens'):
            self.topic_words_lens = [ex.topic_words_lens for ex in sample_list]
            self.topic_words_len = np.array(self.topic_words_lens, dtype=np.int32)
        self.source_seg_ids = [ex.source_seg_ids for ex in sample_list]
        self.source_ids_oo = [ex.source_ids_oo for ex in sample_list]
        self.source_oov_num = [ex.source_oov_num for ex in sample_list]
        #print('!!!!hasattr', hasattr(sample_list[0], 'topic_ids'))
        if hasattr(sample_list[0], 'topic_ids'):
            self.topic_ids = [ex.topic_ids for ex in sample_list]
            self.topic_ids = np.array(self.topic_ids, dtype=np.int32)
        
        if hasattr(sample_list[0], 'topic_words_ids'):
            self.topic_words_ids = [ex.topic_words_ids for ex in sample_list]
            max_src_len = min(max(self.source_len), self.max_src_len)
            self.topic_words_ids = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len] for ids in self.topic_words_ids]
            self.topic_words_ids = np.array(self.topic_words_ids, dtype=np.int32)
            
        if hasattr(sample_list[0], 'mem_segment_ids'):
            self.mem_segment_ids = [ex.mem_segment_ids for ex in sample_list]
            self.mem_segment_ids = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len] for ids in self.mem_segment_ids]
            self.mem_segment_ids = np.array(self.mem_segment_ids, dtype=np.int32)
        # pad
        max_src_len = min(max(self.source_len), self.max_src_len)
        self.source_ids = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                           for ids in self.source_ids]
        self.source_ids_oo = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                              for ids in self.source_ids_oo]
        self.source_seg_ids = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                               for ids in self.source_seg_ids]

        # to numpy array
        self.source_ids = np.array(self.source_ids, dtype=np.int32)
        self.source_ids_oo = np.array(self.source_ids_oo, dtype=np.int32)
        self.source_seg_ids = np.array(self.source_seg_ids, dtype=np.int32)
        
        self.source_len = np.array(self.source_len, dtype=np.int32)

        # Determine the max number of in-article OOVs in this batch
        self.max_oov_num = max([len(ex.source_oovs) for ex in sample_list])
        # Store the in-article OOVs themselves
        self.source_oovs = [ex.source_oovs for ex in sample_list]
        # Fake encoder output
        """
        if hasattr(sample_list[0], 'topic_ids'):
            self.encoder_output = np.zeros([1, 1, self.params.hidden_size + self.params.topic_embedding_size], dtype=np.float32)
        else:"""
        self.encoder_output = np.zeros([1, 1, self.params.hidden_size], dtype=np.float32)
        self.encoder_output_origin = np.zeros([1, 1, self.params.hidden_size], dtype=np.float32)
        # fake memory
        if hasattr(sample_list[0], 'topic_words_ids'):
            self.topic_word_memory = np.zeros([1, 1, self.params.hidden_size], dtype=np.float32)
            
    def init_decoder_seq(self, sample_list):
        # group
        self.target_ids = [ex.target_ids for ex in sample_list]
        self.target_len = [ex.target_len for ex in sample_list]
        self.target_ids_oo = [ex.target_ids_oo for ex in sample_list]

        # pad
        max_tgt_len = min(max(self.target_len), self.max_tgt_len)
        self.target_ids = [(ids + [self.pad_id] * (max_tgt_len - len(ids)))[: max_tgt_len]
                           for ids in self.target_ids]
        self.target_ids_oo = [(ids + [self.pad_id] * (max_tgt_len - len(ids)))[: max_tgt_len]
                              for ids in self.target_ids_oo]

        self.sent_level_attn_bias = init_sentence_level_info(self.period_id, self.target_ids)

        # to numpy array
        self.target_ids = np.array(self.target_ids, dtype=np.int32)
        self.target_ids_oo = np.array(self.target_ids_oo, dtype=np.int32)
        self.target_len = np.array(self.target_len, dtype=np.int32)
        self.init_lm_placeholder()

    def init_lm_placeholder(self):
        mask_id = self.params.maskId
        shape = self.target_ids.shape
        batch, length = shape[0], shape[1]
        target_ids = np.expand_dims(self.target_ids, 1)  # (b, 1, l_t)
        target_ids = np.tile(target_ids, [1, length, 1])  # (b, l_t, l_t)
        self.lm_output_ids = np.reshape(target_ids, [-1, length])  # (b * l_t, l_t)
        self.lm_position = np.array([list(range(length)) for _ in range(batch)], dtype=np.int32)  # (b, l_t)
        self.lm_position = np.reshape(self.lm_position, -1)  # (b * l_t)
        lm_position = np.expand_dims(self.lm_position, 1)  # (b * l_t, 1)
        # set i-th word id to MASK_ID
        for i in range(batch * length):
            self.lm_output_ids[i][lm_position[i]] = mask_id

    def store_orig_strings(self, sample_list):
        """Store the original strings in the Batch object"""
        self.original_data = [ex.original_data for ex in sample_list]  # list of lists

    def update_time(self):
        self.time_step += 1


# noinspection PyAttributeOutsideInit
class Batcher(object):
    """A class to generate minibatches of data. Buckets samples together based on length of the encoder sequence."""

    def __init__(self, processor, hps):
        """Initialize the batcher. Start threads that process the data into batches."""
        logging.info('Init data batcher...')
        self.mode = hps.mode
        self.is_train = self.mode == 'train'
        self.processor = processor
        self._config = hps
        self.batch_num = 0
        logging.info('Prepare data features...')
        print('Prepare data features...')
        self.prepare_examples()
        print('features done...')

        if not self.is_train:
            self._config.batch_size = self._config.eval_batch_size
        else:
            self._config.batch_size = self._config.train_batch_size

        self.BATCH_QUEUE_MAX = 100 if self.is_train else 500000  # max number of batches the batch_queue can hold
        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._sample_queue = queue.Queue(self.BATCH_QUEUE_MAX * self._config.batch_size)

        if not self.is_train:
            self._num_sample_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch samples
            self._bucketing_cache_size = 1  # this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
            self._finished_reading_sample = False  # this will tell us when we're finished reading the sample
        else:
            self._num_sample_q_threads = 16  # num threads to fill sample queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            self._bucketing_cache_size = 100  # how many batches-worth of samples to load into cache before bucketing

        # Start the threads that load the queues
        self._sample_q_threads = []
        for _ in range(self._num_sample_q_threads):
            self._sample_q_threads.append(Thread(target=self.fill_sample_queue))
            self._sample_q_threads[-1].daemon = True
            self._sample_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            target = self.fill_batch_queue if self.is_train else self.fill_infer_batch_queue
            self._batch_q_threads.append(Thread(target=target))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if self.is_train:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def prepare_examples(self):
        processor = self.processor
        mode = self.mode
        config = self._config

        if mode == 'train':
            examples = processor.get_train_examples(config.data_dir, config.train_file)
            random.shuffle(examples)
        elif mode == 'dev':
            examples = processor.get_dev_examples(config.data_dir, config.dev_file)
        elif mode == 'test':
            examples = processor.get_test_examples(config.data_dir, config.test_file)
        else:
            raise ValueError('Only train dev test modes are supported: %s' % mode)

        self.examples = self.processor.filter_examples(examples)
        self.samples_number = len(examples)
        self.processor.log_statistics(examples)

    def next_batch(self):
        """Return a Batch from the batch queue"""
        try:
            if self._batch_queue.qsize() == 0:
                if self.is_train:
                    logging.warning('Bucket input queue is empty when calling next_batch. ',
                                    'Bucket queue size: %i, Input queue size: %i',
                                    self._batch_queue.qsize(), self._sample_queue.qsize())
                # During infer, If the batch queue is empty, return None
                else:
                    if self._finished_reading:
                        logging.info('Finish read all element in batch queue...')
                        return None
                    else:  # Process is quicker than data prepare, wait for a short time
                        logging.warning('During infer, data exhaust when _finished_reading is False...')
                        while self._batch_queue.qsize() == 0:
                            time.sleep(1)
            self.batch_num += 1
            batch = self._batch_queue.get()  # get the next Batch
            return batch
        except Exception as why:
            logging.error('next batch error: {}'.format(why))

    def fill_sample_queue(self):
        """Reads data from file and processes into Examples which are then placed into the sample queue."""
        sample_num = 0
        sample_gen = self.sample_generator()
        while True:
            try:
                sample = sample_gen.__next__()
            except StopIteration:  # if there are no more samples:
                if self.is_train:
                    logging.info("The sample generator for this sample queue filling thread has exhausted data.")
                    raise Exception("The sample generator is out of data during train; error.")
                else:
                    logging.info('Sample reading done, total {} samples...'.format(sample_num))
                    self._finished_reading_sample = True
                    break

            sample_num += 1
            sample = Sample(sample, self._config, self.mode)
            self._sample_queue.put(sample)  # place the Sample in the sample queue.

    def fill_batch_queue(self):
        """
        Takes Examples out of sample queue, sorts them by encoder sequence length,
        processes into Batches and places them in the batch queue.
        """
        while True:
            # Get bucketing_cache_size-many batches of Examples into a list, then sort
            inputs = []
            for _ in range(self._config.batch_size * self._bucketing_cache_size):
                inputs.append(self._sample_queue.get())
            inputs = sorted(inputs, key=lambda inp: inp.source_len)  # sort by length of encoder sequence

            # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
            batches = []
            for i in range(0, len(inputs), self._config.batch_size):
                batches.append(inputs[i:i + self._config.batch_size])
            shuffle(batches)
            for b in batches:  # each b is a list of Example objects
                self._batch_queue.put(Batch(b, len(b), self._config, self.mode))

    def fill_infer_batch_queue(self):
        inputs = []
        total_batch_num = 0
        while True:
            try:
                # Get bucketing_cache_size-many batches of Examples into a list
                for _ in range(self._config.eval_batch_size * self._bucketing_cache_size):
                    inputs.append(self._sample_queue.get(True, 5))
                # Group Samples into batches, place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._config.eval_batch_size):
                    batches.append(inputs[i:i + self._config.eval_batch_size])
            except queue.Empty:  # sample data exhaust OR sample data reader is TOO slow
                if self._finished_reading_sample:
                    if inputs:
                        total_batch_num += 1
                    logging.info('Batch reading done, total {} batches...'.format(total_batch_num))
                    self._finished_reading = True  # sample reading is finished
                # get data in the tail
                batches = [inputs] if inputs else []
            for b in batches:  # each b is a list of Example objects
                total_batch_num += 1
                true_num = len(b)
                while len(b) < self._config.eval_batch_size:
                    b.append(b[-1])
                self._batch_queue.put(Batch(b, true_num, self._config, self.mode), True, 100)
            if self._finished_reading:
                break
            inputs = []

    def watch_threads(self):
        """Watch sample queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._sample_q_threads):
                if not t.is_alive():  # if the thread is dead
                    logging.error('Found sample queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_sample_queue)
                    self._sample_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def sample_generator(self):
        """Read data features"""
        tokenizer = tokenization.FullTokenizer(vocab_file=self._config.vocab_file,
                                               do_lower_case=self._config.do_lower_case)
        dataset = self.examples
        while True:
            idx = np.random.permutation(len(dataset)) if self.is_train else range(len(dataset))
            for i in idx:
                if not self.is_train:
                    yield self.processor.convert_example_to_feature(dataset[i], tokenizer, self._config)
                else:
                    feature = refine_train_summary(dataset[i], self._config)
                    yield self.processor.convert_example_to_feature(feature, tokenizer, self._config)
            if not self.is_train:
                break


class EvalData:
    """Single thread data batcher class"""

    def __init__(self, hps):
        tokenizer = tokenization.FullTokenizer(vocab_file=hps.vocab_file,
                                               do_lower_case=hps.do_lower_case)
        # load custom processer from task name
        task_name = hps.task_name.lower()
        if task_name not in processors:
            raise ValueError('Task not found: %s' % task_name)
        processor = processors[task_name](hps)
        examples = processor.get_dev_examples(hps.data_dir, hps.dev_file)
        examples = processor.filter_examples(examples)

        self.features = [Sample(processor.convert_example_to_feature(example, tokenizer, hps), hps, hps.mode)
                         for example in examples]
        self.batches = []
        for i in range(0, len(self.features), hps.eval_batch_size):
            if i + hps.eval_batch_size > len(self.features):
                self.batches.append(Batch(self.features[i:], len(self.features[i:]), hps, hps.mode))
            else:
                self.batches.append(Batch(self.features[i:i + hps.eval_batch_size], hps.eval_batch_size, hps, hps.mode))
        self.cur_batch_num = 0

    def __len__(self):
        return len(self.batches)

    def next_batch(self):
        if self.cur_batch_num < len(self):
            res = self.batches[self.cur_batch_num]
            self.cur_batch_num += 1
        else:
            res = None
        return res

    def restart(self):
        self.cur_batch_num = 0
