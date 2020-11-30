from collections import Counter, deque

from cytoolz import curry

special_token = [',', ':', '.', '?', 'this', 'it', 'he', 'she', '-lrb-', 'cnn', '-rrb-']


def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count


@curry
def compute_rouge_n(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


@curry
def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b) + 1)]
          for _ in range(0, len(a) + 1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


def _lcs(a, b):
    """ compute the longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = deque()
    while (i > 0 and j > 0):
        if a[i - 1] == b[j - 1]:
            lcs.appendleft(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs


def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    return ngrams


@curry
def mark_words(output, reference_sents, n=3):
    mark_ids = []
    n_grams = []
    for ref in reference_sents:
        n_grams += list(make_n_grams(ref, n))
    ref_grams = Counter(n_grams)
    for i in range(len(output) - n + 1):
        if ref_grams[tuple(output[i:i + n])] > 0:
            for j in range(n):
                if not mark_ids or mark_ids[-1] < (i + j):
                    mark_ids.append(i + j)

    return mark_ids


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def calculate_rouge(art_words, abstract_sent, threshold=0.15):
    res = []
    article = ' '.join(art_words)
    for art_sent in article.split('.'):
        res.append(compute_rouge_n(art_sent.split(), reference=abstract_sent, n=1, mode='f'))
    return max(res) > threshold, max(res)
