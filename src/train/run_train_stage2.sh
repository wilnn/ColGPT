#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT='CoLonGPT'
OUTPUT_DIR="model/stage_2_all_3tasks"
RUN_NAME="stage_2_all_3tasks"
#export WANDB_RESUME="must"
#export WANDB_RUN_ID="auiqr2aj"
TRAINING_STAGE=2
REPORT_TO="wandb"
TRAIN_DS="./dataset/ColonINST/Json-file-clean/train/ColonINST-train-3tasks.json"
DS_IMAGE_PATH="./dataset/ColonINST/Positive-images"
CAP_VAL_DS="./dataset/ColonINST/Json-file-clean/val/ColonINST-val-cap.json"
CLS_VAL_DS="./dataset/ColonINST/Json-file-clean/val/ColonINST-val-cls.json"
REC_VAL_DS="./dataset/ColonINST/Json-file-clean/val/ColonINST-val-rec.json"
REG_VAL_DS="./dataset/ColonINST/Json-file-clean/val/ColonINST-val-reg.json"
CAP_TEST_DS="./dataset/ColonINST/Json-file-clean/test/ColonINST-test-cap.json"
CLS_TEST_DS="./dataset/ColonINST/Json-file-clean/test/ColonINST-test-cls.json"
REC_TEST_DS="./dataset/ColonINST/Json-file-clean/test/ColonINST-test-rec.json"
REG_TEST_DS="./dataset/ColonINST/Json-file-clean/test/ColonINST-test-reg.json"
LLM_PATH="meta-llama/Llama-3.2-1B-Instruct"
VM_PATH="google/siglip-so400m-patch14-384"
VM_TYPE="siglip"
VP_TYPE="SPP"
FUSION_TYPE="concatenation"
PYRAMID_SHAPE="[[14, 14], [7, 7], [1, 1]]"
IMAGE_TAG="<image>"
IGNORE_INDEX=-100
MAX_LENGTH=512
MAX_DATASET_SIZE=-1
NUM_EPOCHS=7
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
LM_CLASS="llama"
RESUME_FROM_CHECKPOINT="None"
STAGE1_CHECKPOINT="./model/stage_1/checkpoint-10420/model.safetensors"
#LORA_RANK=128
#LORA_ALPHA=256
#--stage2_with_cap \

accelerate launch --config_file ./src/train/accelerate_config.yaml -m src.train.train \
            --log_level="info" \
            --stage1_checkpoint=$STAGE1_CHECKPOINT \
            --bf16 \
            --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
            --LM_class=$LM_CLASS \
	        --train_set_path $TRAIN_DS \
	        --cap_val_set_path $CAP_VAL_DS \
            --cls_val_set_path $CLS_VAL_DS \
            --reg_val_set_path $REG_VAL_DS \
            --rec_val_set_path $REC_VAL_DS \
            --cap_test_set_path $CAP_TEST_DS \
            --cls_test_set_path $CLS_TEST_DS \
            --reg_test_set_path $REG_TEST_DS \
            --rec_test_set_path $REC_TEST_DS \
			--image_path $DS_IMAGE_PATH \
			--pretrained_language_model_path $LLM_PATH \
			--pretrained_vision_encoder_path $VM_PATH \
			--pretrained_vision_encoder_type $VM_TYPE \
			--vision_projector_type $VP_TYPE \
			--fusion_type $FUSION_TYPE \
			--pyramid_shapes "$PYRAMID_SHAPE" \
			--training_stage $TRAINING_STAGE \
			--image_tag $IMAGE_TAG \
			--ignore_index $IGNORE_INDEX \
			--custom_max_length $MAX_LENGTH \
            --max_dataset_size=$MAX_DATASET_SIZE \
            --include_for_metrics "inputs" "loss" \
            --eval_strategy="steps" \
            --batch_eval_metrics \
            --eval_steps=0.15 \
            --save_strategy="steps" \
            --save_steps=0.15 \
            --load_best_model_at_end \
            --metric_for_best_model="perplexity" \
            --greater_is_better=false \
            --save_total_limit=2 \
            --logging_strategy="steps" \
            --logging_steps=20 \
            --learning_rate=2e-4 \
			--lr_scheduler_type="cosine" \
			--warmup_steps=0.05 \
            --weight_decay=1e-5 \
            --report_to=$REPORT_TO \
            --output_dir=$OUTPUT_DIR \
            --run_name=$RUN_NAME \
            --per_device_train_batch_size=$TRAIN_BATCH_SIZE \
            --per_device_eval_batch_size=$EVAL_BATCH_SIZE \
            --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            --num_train_epochs=$NUM_EPOCHS \
            --do_train \
            --do_eval \
            --dataloader_num_workers=8 \
            --ddp_find_unused_parameters=false \
