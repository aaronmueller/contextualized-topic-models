#!/bin/bash

source /home/amueller/miniconda3/bin/activate
conda activate pytorch

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`

CTM_DIR=/export/c04/amueller/contextualized-topic-models/contextualized_topic_models

python examples/language-modeling/run_language_modeling.py \
	--model_name_or_path bert-base-multilingual-cased \
	--model_type bert \
	--do_train \
	--do_eval \
	--mlm \
	--seed 42 \
	--train_data_file $CTM_DIR/data/wiki/topic_train_sub.txt \
	--eval_data_file $CTM_DIR/data/wiki/topic_dev.txt \
	--learning_rate 1e-4 \
	--num_train_epochs 1 \
	--save_total_limit 3 \
	--save_steps 512 \
	--output_dir models/wiki-mbert \
	--per_gpu_train_batch_size 1 \
	--gradient_accumulation_steps 256
