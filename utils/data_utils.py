import re
from collections import Counter

from data_reading import abstract2sents_func
from utils.refine_utils import calculate_rouge

sent_recall = []
purged_sent_num = 0


def draw():
    import matplotlib.pyplot as plt
    plt.hist(sent_recall, bins=20)
    plt.show()


def refine_train_summary(feature, params):
    global sent_recall, purged_sent_num
    if not params.refine_train_summary:
        return feature
    # 1. split the summary by sentence
    sentences = [sent.strip() for sent in abstract2sents_func(params)(feature.summary)]
    # 2. for each summary sentence, calculate recall of this sent and the article
    refined_sentences = []
    for sentence in sentences:
        sentence_word = sentence.split()
        # keep_it = calculate_recall_v2(feature.article.split()[:params.max_seq_length], sentence_word)
        keep_it, score = calculate_rouge(feature.article.split()[:params.max_seq_length], sentence_word)
        # keep_it = calculate_recall(feature.article.split()[:params.max_seq_length], sentence_word)
        # for recall lower than a threshold, delete the summary sentence
        if keep_it or len(sentence_word) < 5:
            refined_sentences.append(sentence)
        else:
            purged_sent_num += 1
    feature.true_summary = ' '.join(refined_sentences)
    return feature


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def calculate_recall(prediction, ground_truth):
    return precision_recall_f1(prediction, ground_truth)[1]


def split_article(article):
    return re.split('[.?!]', article)


def calculate_recall_v1(article_word, sentence_word, threshold=0.45):
    return calculate_recall(article_word, sentence_word) > threshold


def calculate_recall_v2(article_word, sentence_word, threshold=0.25):
    for article_sent in split_article(' '.join(article_word)):
        if calculate_recall(article_sent.split(), sentence_word) > threshold:
            return True
    return False
