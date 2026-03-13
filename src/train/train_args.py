import argparse
from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class trainArgs(TrainingArguments):
    train_set_path: str = None
    cap_val_set_path: str = None
    cls_val_set_path: str = None
    reg_val_set_path: str = None
    rec_val_set_path: str = None
    cap_test_set_path: str = None
    cls_test_set_path: str = None
    reg_test_set_path: str = None
    rec_test_set_path: str = None
    image_path: str = None
    custom_max_length: int = None
    ignore_index: int = -100
    image_tag: str = '<image>'
    training_stage: int = 1
    LM_full_fine_tuning: bool = True
    pretrained_language_model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    pretrained_vision_encoder_path: str = "google/siglip-so400m-patch14-384"
    pretrained_vision_encoder_type: str = 'siglip'
    fusion_type: str = "concatenation"
    vision_projector_type: str = "spp"
    pyramid_shapes: str = "[[14, 14], [7, 7], [1, 1]]"
    max_dataset_size: int = -1 # negative for full dataset
    LM_class: str = 'llama'
    iou_threshold: int = 0.5
    stage2_with_cap: bool = False
    stage1_checkpoint: str = None
    lora_rank: int = 128
    lora_alpha: int = 256
    stage2_lora_all: bool = False
    #field(default=3.0, metadata={"help": "Total number of training epochs to perform."})

    def __post_init__(self):
        super().__post_init__()

def create_parser(input_args=None):
    parser = argparse.ArgumentParser(description="Training script for Llava model")
    
    '''
    parser.add_argument(
        "--path_to_ds_config",
        type=str,
        default=None,
        help="path to the cofig file of deepspeed. often named as 'ds_config.json'",
    )
    '''

    parser.add_argument(
        "--path_to_json",
        type=str,
        required=True,
        help="Path to the json file that has the training data.",
    )

    parser.add_argument(
        "--path_to_val_json",
        type=str,
        default=None,
        help=("Path to the json file that has the validation data"
            "if given, the validation will be run every `--log_every_update_step` step"),
    )

    parser.add_argument(
        "--path_to_image_folder",
        type=str,
        required=True,
        help="Path to the folder that contains images",
    )

    parser.add_argument(
        "--pretrained_language_model_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        required=True,
        help="Path to pretrained language model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--pretrained_vision_encoder_path",
        type=str,
        default="google/siglip-so400m-patch14-384",
        required=True,
        help="Path to pretrained vision model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--pretrained_vision_encoder_type",
        type=str,
        default='siglip',
        required=True,
        help="indicate the type of the vision encpder. Valid value: 'siglip'. ",
    )

    parser.add_argument(
        "--fusion_type",
        type=str,
        default='concatenation',
        required=True,
        help="the fusion type to fuse multi-modal inputs",
    )

    parser.add_argument(
        "--vision_projector_type",
        type=str,
        default='spp',
        required=True,
        help="indicate the type of the vision projector. Valid value: 'spp'. ",
    )

    parser.add_argument(
        "--pyramid_shapes",
        type=str,
        default='[[14, 14], [7, 7], [1, 1]]',
        help="the pyramid shape when use the 'spp' projector type",
    )

    parser.add_argument(
        "--custom_max_length",
        type=int,
        default=None,
        help="the custom max length for the LLM",
    )

    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help="The ignore index to be used to compute the loss",
    )
    
    parser.add_argument(
        "--image_tag",
        type=str,
        default='<image>',
        help= ("The default image tag that will be used to indicate the location of image in the prompt."
            "Only uesful when use concatenation fusion"
            ),
    )

    parser.add_argument(
        "--log_every_update_step",
        type=int,
        default=1,
        help= "log every this number of update, which is every gradient_accumulation_steps (the loss will be average)"
        
    )

    parser.add_argument(
        "--training_stage",
        type=int,
        required=True,
        help="choose between [1, 2]. 1 Means stage 1 llave training, 2 means stage 2",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )


    parser.add_argument(
        "--train_batch_size_per_device",
        type=int,
        default=16,
        help="Batch size(per device) (also per accumulation step)  for the training dataloader."
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="define the max training step for debugging purpose"
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
    "--validation_epochs",
    type=int,
    default=1,
    help=(
        "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
        " `args.validation_prompt` multiple times: `args.num_validation_images`."
    ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="llava-checkpoints",
        help="The output directory where the model checkpoints will be saved to.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )

    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="type of optimizer. choose between: 'SGD', 'adam', 'adamw', 'fusedadam', or 'fusedadamw'",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub."
        )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub."
        )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Whether or not to use npu flash attention."
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--LM_full_fine_tuning",
        default=False,
        action="store_true",
        help=("Wheter to do a full fine tuning on the language model."
              "if this is true, it will not use LoRA even if you pass the LoRA rank"),
    )

    '''
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )


    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    
    
    parser.add_argument(
        "--te1_lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--te2_lr",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )

    
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    
    parser.add_argument(
        "--debug_loss",
        action="store_true",
        help="debug loss for each image, if filenames are available in the dataset",
    )

    

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    '''
    
    '''
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()'''



    return parser
