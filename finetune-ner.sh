data_dir=$1
bert_model=$2
python ./examples/run_token_level_classification.py \
    --task_name msra \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $data_dir\
    --bert_model $bert_model\
    --max_seq_length 128 \
    --train_batch_size 32 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1 \
    --cache_dir ./tmp \
    --no_cuda
