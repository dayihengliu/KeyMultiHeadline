#/bin/bash

export CUDA_VISIBLE_DEVICES="0"

SRC_DIR=".";
OUTPUT_ROOT_DIR="outputs_dir";



EXP_NAME=${1:-exp_1}
MODEL=${2:-baseline_copy_same_vocab_multi_gpu}
DATA_DIR=${3:-dataset}
EPOCHS=${4:-10}
ENCODER_SEQ_LENGHT=${5:-512}
DECODER_SEQ_LENGHT=${6:-100}
TASKNAME=${7:-news_query}
BERT_BASE_DIR=${8:-uncased_L-12_H-768_A-12}

DP_RATE="0.14"
DECODER_PARAMS="--num_decoder_layers=12 --num_heads=12 --filter_size=3072"
GPU_LIST=("0")
TRAIN=1
SELECT_MODEL=1
TEST=1
EVAL_ONLY=False
LOG_FILE="--log_file=${OUTPUT_ROOT_DIR}/log/${MODEL}-${TASKNAME}-${EXP_NAME}.log"
OUTPUT_DATA_DIR="${OUTPUT_ROOT_DIR}/${MODEL}-${TASKNAME}-${EXP_NAME}"

echo "---folder---"
echo "SRC_DIR=${SRC_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DATA_DIR=${OUTPUT_DATA_DIR}"

echo "---paramter---"
echo "MODEL=${MODEL}"
echo "DATASET=${DATASET}"
echo "EXP_NAME=${EXP_NAME}"

if [[ ${TRAIN} == 1 ]]
then

python $SRC_DIR/run.py \
  ${DECODER_PARAMS} \
  ${LOG_FILE} \
  --output_dir=${OUTPUT_DATA_DIR} \
  --model_name=${MODEL} \
  --task_name=${TASKNAME} \
  --mode=train \
  --data_dir=${DATA_DIR} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --load_pre_train=False \
  --train_file=train_keyaware_news_KQTAC.txt \
  --attention_dropout=${DP_RATE} \
  --residual_dropout=${DP_RATE} \
  --relu_dropout=${DP_RATE} \
  --gpu=${GPU_LIST} \
  --num_train_epochs=${EPOCHS} \
  --learning_rate=3e-4 \
  --max_seq_length=${ENCODER_SEQ_LENGHT} \
  --evaluate_every_n_step=500 \
  --train_batch_size=3 \
  --accumulate_step=12 \
  --rl_lambda=0.99 \
  --start_portion_to_feed_draft=0.99 \
  --draft_feed_freq=1 \
  --mask_percentage=0.15 \
  --repeat_percentage=0.15 \
  --switch_percentage=0.15 \
  --max_out_seq_length=${DECODER_SEQ_LENGHT}

fi

if [[ ${SELECT_MODEL} == 1 ]]
then

GPU_LIST=("0")

python $SRC_DIR/run.py \
  ${DECODER_PARAMS} \
  ${LOG_FILE} \
  --output_dir=${OUTPUT_DATA_DIR} \
  --model_name=${MODEL} \
  --task_name=${TASKNAME} \
  --mode=eval \
  --dev_file=dev_keyaware_news_KQTAC.txt \
  --data_dir=${DATA_DIR} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --gpu=${GPU_LIST} \
  --max_seq_length=${ENCODER_SEQ_LENGHT} \
  --eval_batch_size=20 \
  --max_out_seq_length=${DECODER_SEQ_LENGHT}

fi

if [[ ${TEST} == 1 ]]
then

GPU_LIST=("0")

python $SRC_DIR/run.py \
  ${DECODER_PARAMS} \
  ${LOG_FILE} \
  --output_dir=${OUTPUT_DATA_DIR} \
  --model_name=${MODEL} \
  --task_name=${TASKNAME} \
  --mode=test \
  --test_file=test_keyaware_news_KQTAC.txt \
  --eval_only=${EVAL_ONLY} \
  --data_dir=${DATA_DIR} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --gpu=${GPU_LIST} \
  --max_seq_length=${ENCODER_SEQ_LENGHT} \
  --eval_batch_size=20 \
  --beam_size=4 \
  --decode_alpha=1.0 \
  --max_out_seq_length=${DECODER_SEQ_LENGHT} \
  --use_beam_search=True

fi
