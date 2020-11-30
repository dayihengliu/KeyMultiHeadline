@echo off
set DATA_DIR="../data/cnn-dm/chunked/train_*"
set VOCAB="../data/cnn-dm/vocab.txt"
set EXP_NAME="11-19-1"
@echo on

python run_summarization.py ^
  --coverage ^
  --batch_size=6 ^
  --mode=train ^
  --data_path=%DATA_DIR% ^
  --vocab_path=%VOCAB% ^
  --log_root="log/" ^
  --exp_name=%EXP_NAME%