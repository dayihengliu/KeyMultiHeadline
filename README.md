# Diverse, Controllable, and Keyphrase-Aware: A Corpus and Method for News Multi-Headline Generation

This repo contains the code and data of the following paper:
> **Diverse, Controllable, and Keyphrase-Aware: A Corpus and Method for News Multi-Headline Generation**, Dayiheng Liu, Yeyun Gong, Yu Yan, Jie Fu, Bo Shao, Daxin Jiang, Jiancheng Lv, Nan Duan, EMNLP2020


# Prerequisites
- Python 3.6
- numba 0.49.1
- tensorflow 1.10.0
- numpy 1.16.2
- nltk 3.3+
- cuda 9.0


# Datasets
Download the dataset file at [here](https://drive.google.com/file/d/17xEdwdXwLar1w7JkRqnsXh1n6kFokodN/view?usp=sharing).
```
tar -xzvf keyaware_news_emnlp20.tar
```
The data file directory is as follows
```
dataset/
|-- dev_keyaware_news_KQTAC.txt
|-- test_keyaware_news_KQTAC.txt
|-- test_keyaware_news_KQTAC_5slot.txt
|-- test_keyaware_news_KQTAC_multi.txt
`-- train_keyaware_news_KQTAC.txt
```
The files `train_keyaware_news_KQTAC.txt`, `dev_keyaware_news_KQTAC.txt`, and `test_keyaware_news_KQTAC.txt` contain 5-tuple <keyphrase, query, title, article, click_times>.

The file `test_keyaware_news_KQTAC_multi.txt` provides 5 tuples with different predicted keyphrases for each test example. For each article, we obtained 5 keyphrases by the SEQ2SEQ model as described in our paper.

Similarly, the file `test_keyaware_news_KQTAC_5slot.txt` provides 5 tuples with different predicted keyphrases for each test example. For each article, we obtained 5 keyphrases by the SLOT model as described in our paper.

# Baselines

Our baselines are based on BERT-base-uncased model, which can be downloaded at [here](https://drive.google.com/file/d/13K_OUOJvwTAFvaPs9faub49zfd28aWKq/view?usp=sharing).

run ``run_base.sh`` for BASE model training and testing.

run ``run.sh`` for our model training and testing.

The detailed hyper-parameters can be found in `run.sh`  and `config.py`.

The model checkpoints and log file will be saved at `OUTPUT_DATA_DIR` and `LOG_FILE` in `run.sh`, respectively.

Note that we also provide some variants of the keyphrase-aware headline generation model and keyphrase-agnostic baselines, which can be found in `model_pools/`. You can replace the `MODEL=${2:-encoder_filter_query_plus_decoder_mem}` in `run.sh` to other models (the model names can be found in `model_pools/__init__.py` if you want to use other baselines.


# Citation
```
@inproceedings{liu2020keynews,
    title = "Diverse, Controllable, and Keyphrase-Aware: A Corpus and Method for News Multi-Headline Generation",
    author={Liu, Dayiheng and Gong, Yeyun and Yan, Yu and Fu, Jie and Shao, Bo and Jiang, Daxin and Lv, Jiancheng and Duan, Nan},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020"
}
```