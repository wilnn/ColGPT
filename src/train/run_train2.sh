export CUDA_VISIBLE_DEVICES=0,1
export TRAIN_DS="/home/public/htnguyen/project/ColonGPT/dataset/ColonINST/Json-file-clean/train/ColonINST-train-cap.json"
export DS_IMAGE_PATH="/home/public/htnguyen/project/ColonGPT/dataset/ColonINST/Positive-images"
export VAL_DS="/home/public/htnguyen/project/ColonGPT/dataset/ColonINST/Json-file-clean/val/ColonINST-val-cap.json"
export LLM_PATH="meta-llama/Llama-3.2-1B-Instruct"
export VM_PATH="google/siglip-so400m-patch14-384"
export VM_TYPE="siglip"
export VP_TYPE="SPP"
export FUSION_TYPE="concatenation"
export PYRAMID_SHAPE="[[14, 14], [7, 7], [1, 1]]"

deepspeed --module --num_gpus=2 --master_port=29510 src.train.train \
	--train_set_path $TRAIN_DS \
	--val_set_path $VAL_DS \
	--path_to_image_folder $DS_IMAGE_PATH \
	--pretrained_language_model_path $LLM_PATH \
	--pretrained_vision_encoder_path $VM_PATH \
	--pretrained_vision_encoder_type $VM_TYPE \
	--vision_projector_type $VP_TYPE \
	--fusion_type $FUSION_TYPE \
	--pyramid_shapes "$PYRAMID_SHAPE" \
	--training_stage 1 \
	--image_tag $IMAGE_TAG \
	--ignore_index $IGNORE_INDEX \
	--custom_max_length $MAX_LENGTH \
	--log_every_update_step 10 \
	--train_batch_size_per_device 16 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 20 \
	--checkpointing_steps 500 \
	--output_dir "train_test" \
	--seed 42 \
	--optimizer "adamw" \
	--learning_rate 2e-4 \
	--lr_scheduler "cosine" \
	--lr_warmup_steps 500 \
	--dataloader_num_workers 2 \