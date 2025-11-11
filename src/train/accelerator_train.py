
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import json
from PIL import Image
from train_args import create_parser
import deepspeed
from transformers import ProcessorMixin
from torchvision import transforms
import re
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoImageProcessor,AutoConfig, AutoTokenizer
from ..model.configuration_llava import LlavaConfig
import torch
from urllib.parse import urlparse
from PIL import Image
import requests
import inspect
import gc
from datetime import datetime
import sys
from ..model.vision_encoder.siglip import SiglipVisionEncoder
from ..model.vision_projector.spp import SPP
from ..model.language_model.llama.modeling_llama import LlamaForCausalLM
from ..model.modeling_llava import LlavaForCausalLM
from ..model.processing_llava import LlavaProcessor
import ast
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb
from torch import optim
from transformers import get_scheduler
from ds_config import ds_config
import math 
from tqdm.auto import tqdm
from accelerate.utils import set_seed
import wandb
import torch.distributed as dist


from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration


class LlavaDataset(Dataset):
    def __init__(self, path_to_json,
                 path_to_image_folder='./',
                 system_prompt=None,
                 data_processor=None
                 ):
        # Example data: features and labels
        with open(path_to_json, "r") as f:
            self.data = json.load(f)  # returns a list of dict(json)
        
        self.path_to_image_folder = path_to_image_folder
        self.system_prompt = system_prompt
        self.data_processor = data_processor
        
    def __len__(self):
        return len(self.data)  # number of samples
    
    def apply_processor(self, processor):
        self.data_processor = processor

    def __getitem__(self, idx,
                    fix_keys_to_role_content=True,
                    **kwargs,):
        
        if type(self.data[idx]['image']) is str:
            img = [Image.open(self.path_to_image_folder+self.data[idx]['image'])]
        elif type(self.data[idx]['image']) is list:
            img = [Image.open(self.path_to_image_folder+i) for i in self.data[idx]['image']]
        else:
            raise TypeError('the value of "image" key should be string or list')

        temp = {'role':'system', 'content':self.system_prompt} if self.system_prompt else []
        
        if fix_keys_to_role_content:  
            temp += [{'role': i['from'],'content': i['value'],} for i in self.data[idx]['conversations']]
        else:
            temp += self.data[idx]['conversations']

        if self.data_processor:
            return self.data_processor(text=[temp],
                                       images=[img],
                                       generation=False,
                                       **kwargs,
                                       )
        else:
            return {
                'image':img,
                'conversation':temp,
            }

def get_optimizer(model, args):
    if args.optimizer.lower() == 'SGD':
        return optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == 'adam':
        if args.use_8bit_adam:
            return bnb.optim.Adam8bit(model.parameters(),
                                      lr=args.learning_rate,
                                      betas=(args.adam_beta1, args.adam_beta2),
                                      eps=args.adam_epsilon,
                                      weight_decay=args.adam_weight_decay,
                                      )
        else: 
            return optim.Adam(model.parameters(),
                            lr=args.learning_rate,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_epsilon,
                            weight_decay=args.adam_weight_decay,
                            )
    elif args.optimizer.lower() == 'adamw':
        if args.use_8bit_adam:
            return bnb.optim.AdamW8bit(model.parameters(),
                                      lr=args.learning_rate,
                                      betas=(args.adam_beta1, args.adam_beta2),
                                      eps=args.adam_epsilon,
                                      weight_decay=args.adam_weight_decay,
                                      )
        else: 
            return optim.AdamW(model.parameters(),
                            lr=args.learning_rate,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_epsilon,
                            weight_decay=args.adam_weight_decay,
                            )
    elif args.optimizer.lower() == 'fusedadam': # GPU ONLY
        return deepspeed.ops.adam.FusedAdam(model.parameters(),
                        lr=args.learning_rate,
                        betas=(args.adam_beta1, args.adam_beta2),
                        eps=args.adam_epsilon,
                        weight_decay=args.adam_weight_decay,
                        adam_w_mode=False
                        )
    elif args.optimizer.lower() == 'fusedadamw': # GPU ONLY
        return deepspeed.ops.adam.FusedAdam(model.parameters(),
                        lr=args.learning_rate,
                        betas=(args.adam_beta1, args.adam_beta2),
                        eps=args.adam_epsilon,
                        weight_decay=args.adam_weight_decay,
                        adam_w_mode=True
                        )
    else:
        raise ValueError(f"Unknow optimizer type: {args.optimizer}. Please add this optimizer type to `get_optimizer` function if needed")

def do_eval(model_engine, val_dataloader):
    model_engine.eval()  # set to evaluation mode
    accum_loss = torch.tensor(0.0, device=model_engine.device)
    
    with torch.inference_mode():  # no need to compute gradients
        for step, batch in enumerate(val_dataloader):
            batch['attention_mask'] = batch['attention_mask'].to(model_engine.device)
            batch['input_ids'] = batch['input_ids'].to(model_engine.device)
            #print(output['input_ids'].requires_grad)
            batch['labels'] = batch['labels'].to(model_engine.device)
            for i in range(len(batch['images'])):
                if type(batch['images'][i]) is not list:
                    batch['images'][i] = batch['images'][i].to(model_engine.device)

            output = model_engine(**batch, use_cache=False)
            accum_loss +=output['loss']

        accum_loss /= len(val_dataloader)
        deepspeed.comm.all_reduce(accum_loss, op=deepspeed.comm.ReduceOp.SUM)
        return accum_loss / deepspeed.comm.get_world_size()


def main(args):
    set_seed(args.seed)
    #####################################################
    ##### CRAEATE, SET UP MODEL AND PROCESSOR ###########
    #####################################################

    vision_config = AutoConfig.from_pretrained(args.pretrained_vision_encoder_path).vision_config
    vision_encoder = AutoModel.from_pretrained(args.pretrained_vision_encoder_path).vision_model
    vision_encoder = SiglipVisionEncoder(vision_model=vision_encoder)

    lm_config = AutoConfig.from_pretrained(args.pretrained_language_model_path)
    language_model = AutoModelForCausalLM.from_pretrained(args.pretrained_language_model_path)
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')

    if args.custom_max_length:
        max_length = args.custom_max_length
    else:
        max_length = tokenizer.model_max_length
    
    pyramid_shapes = ast.literal_eval(args.pyramid_shapes) if args.vision_projector_type == 'spp' else None

    config = LlavaConfig(model_max_length=max_length,
                     hidden_size=lm_config.hidden_size,
                     patch_hidden_size=vision_config.hidden_size,
                     vision_projector_type=args.vision_projector_type,
                     pyramid_shapes= pyramid_shapes,
                     vision_encoder_type =args.pretrained_vision_encoder_type,
                     vision_encoder_path=args.pretrained_vision_encoder_path,
                     language_model_path=args.pretrained_language_model_path,
                     )
    tokenizer.model_max_length = max_length
    
    if config.vision_projector_type.lower() == 'spp':
        vision_projector = SPP(config)
    else:
        raise NotImplementedError(f"Unsupported vision projector type: {config.vision_projector_type}\n. Make sure you implement this vision projector type.")

    # STAGE 1. ONLY TRAIN THE VISION PROJECTOR TO ALIGN IMAGE FEATURES TO THE LM'S EMBEDDINGS SPACE
    if args.training_stage == 1:
        for param in vision_encoder.parameters():
            param.requires_grad = False
        
        for param in language_model.parameters():
            param.requires_grad = False
    
    elif args.training_stage == 2: # STAGE 2. ONLY TRAIN THE VISION PROJECTOR TO ALIGN IMAGE FEATURES TO THE LM'S EMBEDDINGS SPACE
        for param in vision_encoder.parameters():
            param.requires_grad = False
        
        if not args.LM_full_fine_tuning: # Do LoRA
            for param in language_model.parameters():
                param.requires_grad = False
            
            lora_config = LoraConfig(
                    r=args.rank,                     # rank of the low-rank matrices
                    lora_alpha=args.rank,           # scaling factor
                    target_modules=["q_proj", 'k_proj', "v_proj", 'o_proj'],  # which layers to inject LoRA
                    #lora_dropout=0.05,       # dropout in LoRA layers
                    #bias="none",
                    task_type="CAUSAL_LM"
                )
            language_model = get_peft_model(language_model, lora_config)
            


    model = LlavaForCausalLM(config=config,
                            vision_encoder=vision_encoder,
                            vision_projector=vision_projector,
                            language_model=language_model)
    '''
    # Get total number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Get dtype size in bytes (usually 4 for float32, 2 for float16)
    dtype_size = next(model.parameters()).element_size()

    # Total size in bytes
    total_bytes = num_params * dtype_size

    # Convert to GB
    total_gb = total_bytes / (1024 ** 3)

    print(f"Model size (approx): {total_gb:.3f} GB")
    print('dtype:', model.dtype)
    '''
    ################## MAKE PROCESSOR ############################

    # some models like Llama do not have the padding token in the vocabulary
    if not tokenizer.pad_token:
        #self.tokenizer.pad_token = "<pad>"
        #self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # do this to avoid having to resize the embeddings matrix of the model
        tokenizer.pad_token = tokenizer.eos_token

    #print(tokenizer.eos_token_id)


    if config.image_tag not in tokenizer.get_vocab():
        tokenizer.add_tokens([config.image_tag])

    # save the id of the image tag in the tokenizer so the config so that
    # the config object will now has the correct image token index
    config.image_token_index = tokenizer.get_vocab()[config.image_tag]

    # save image tag to init_kwargs for use in the processor __call__ method
    tokenizer.init_kwargs['image_token'] = config.image_tag

    image_processor = AutoProcessor.from_pretrained(args.pretrained_vision_encoder_path, use_fast=True).image_processor

    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    

    #####################################################
    ##### CRAEATE, SET UP OPTIMIZER, LR SCHEDULER, DATASET, DATALOADER, ETC., ###########
    #####################################################
    def collate_fn(batch):
        
        return {
            'images':[n['images'][0] for n in batch],
            'image_pos': [n['image_pos'][0] for n in batch],
            'input_ids':torch.cat([n['input_ids'] for n in batch], dim=0),
            'attention_mask':torch.cat([n['attention_mask'] for n in batch], dim=0),
            'labels':torch.cat([n['labels'] for n in batch], dim=0),
        }

    
    system_prompt = 'You are a colonoscopy assistant providing help with colonoscopy tasks'
    train_dataset = LlavaDataset(path_to_json=args.path_to_json,
                           path_to_image_folder=args.path_to_image_folder
                           ,system_prompt=system_prompt,
                           data_processor=processor,
                           )
    if args.path_to_val_json:
        val_dataset = LlavaDataset(path_to_json=args.path_to_val_json,
                           path_to_image_folder=args.path_to_image_folder
                           ,system_prompt=system_prompt,
                           data_processor=processor,
                           )

    #args.max_train_steps = args.max_train_steps if args.max_train_steps else len(dataset)*args.num_train_epochs
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size_per_device,
                                  Shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=args.dataloader_num_workers)

    val_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size_per_device,
                                  Shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=args.dataloader_num_workers)

    # get the actual number of batches in one epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        # get the actual number of batches per device in entire training
        # as if only do 1 accumulation step
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
    optimizer = get_optimizer(model, args)
    

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )


    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     config=ds_config,
                                                     )
    
    ###############################################################
    ##################### PREPARE FOR TRAINNIG ####################
    ###############################################################
    
    initial_global_step = 0 # to start from the last checkpoint
    first_epoch = 0 # to resume from training
    
    # START A NEW RUN. NOT FOR RESUME FROM CHECKPOINT
    project = "ColonGPT-stage1" if args.training_stage == 1 else "ColonGPT-stage2"
    run = wandb.init(
        project=project,   # project name (will be created if it doesn't exist)
        #name="experiment-1",          # optional: name of this run
        
        config={vars(args)} # SAVE THE TRAINING HYPERPARAMETERS
                                    # OF THIS RUN TO COMPARE WITH OTHER RUNS
    )

    '''
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step, # to start from the last checkpoint
        desc="update steps",

        # model_engine.global_rank: the process rank across all nodes.
        # model_engine.local_rank: rank within the current node.
        disable= not (model_engine.local_rank == 0 and model_engine.global_rank == 0),
    )'''
    

    for epoch in range(first_epoch, args.num_train_epochs):

        if accelerator.is_main_process:
            progress_bar = tqdm(
            range(0, num_update_steps_per_epoch),
            initial=initial_global_step, # to start from the last checkpoint
            desc=f"update steps (epoch {epoch+1})",
            # model_engine.global_rank: the process rank across all nodes.
            # model_engine.local_rank: rank within the current node.
            disable= not (model_engine.local_rank == 0 and model_engine.global_rank == 0),
            )
            progress_bar.set_postfix({"loss": f"{0.0:.4f}"})

        model_engine.train()
        #train_loss = 0.0
        print(f"start epoch {epoch+1}")
        epoch_loss = torch.tensor(0.0, device=model_engine.device)
        
        accum_loss = torch.tensor(0.0, device=model_engine.device)
        log_every_loss = torch.tensor(0.0, device=model_engine.device)
        for step, batch in enumerate(train_dataloader):
            #things_to_log = {}
            
            batch['attention_mask'] = batch['attention_mask'].to(model_engine.device)
            batch['input_ids'] = batch['input_ids'].to(model_engine.device)
            #print(output['input_ids'].requires_grad)
            batch['labels'] = batch['labels'].to(model_engine.device)
            for i in range(len(batch['images'])):
                if type(batch['images'][i]) is not list:
                    batch['images'][i] = batch['images'][i].to(model_engine.device)

            output = model_engine(**batch, use_cache=False)
            
            # Backward pass
            model_engine.backward(output['loss'])
            # Optimizer Step
            model_engine.step()

            accum_loss += output['loss'].detach() # .detach to avoid tracking grads
            if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader)-1:
                # scale for number of accumulation step
                # accum_loss after this will be the loss of 1 batch per device
                if step == len(train_dataloader)-1 and (step+1) % args.gradient_accumulation_steps != 0:
                    accum_loss = accum_loss / ((step+1) % args.gradient_accumulation_steps)
                else:
                    accum_loss = accum_loss / args.gradient_accumulation_steps

                log_every_loss = log_every_loss + accum_loss
                epoch_loss = epoch_loss + accum_loss
                

                #temp = epoch_loss / num_update_step_so_far
                deepspeed.comm.all_reduce(accum_loss, op=deepspeed.comm.ReduceOp.SUM)
                accum_loss /= deepspeed.comm.get_world_size()
                if accelerator.is_main_process:
                    # update progress bar by one step
                    progress_bar.update(1)
                    # show loss on the right of progress bar
                    progress_bar.set_postfix({"epochAvgLoss": f"{accum_loss:.4f}"})

                accum_loss.zero_() # reset to 0
                
                
                # update loss in progress bar

                if (step+1) % (args.log_every*args.gradient_accumulation_steps) == 0 or step == len(train_dataloader)-1:
                    # get average accumm_loss per device 
                    
                    if step == len(train_dataloader)-1 and (step+1) % (args.log_every*args.gradient_accumulation_steps) != 0:
                        log_every_loss /= ((step+1) % (args.log_every*args.gradient_accumulation_steps))
                    else:
                        log_every_loss /= args.log_every

                    # SUM accum_loss tensor arcoss all processes
                    #  and deepspeed requires doing this in all processes 
                    deepspeed.comm.all_reduce(log_every_loss, op=deepspeed.comm.ReduceOp.SUM)

                    if deepspeed.comm.get_rank() == 0:
                        # scale for number of processes. 
                        # after this, accum_loss will be the correct loss as
                        # if you train on 1 device and 1 accumulation step
                        log_every_loss /= deepspeed.comm.get_world_size()
                        
                        #things_to_log['loss_per_batch'] = accum_loss.item()
                        wandb.log({'loss_per_update_step': log_every_loss.item()})

                    log_every_loss.zero_()

        epoch_loss = epoch_loss / num_update_steps_per_epoch # average loss per update step
        deepspeed.comm.all_reduce(epoch_loss, op=deepspeed.comm.ReduceOp.SUM)
        if deepspeed.comm.get_rank() == 0:
            wandb.log({'avg_loss_per_epoch': epoch_loss/ deepspeed.comm.get_world_size()})

        if args.path_to_val_json:
            do_eval(model_engine, val_dataloader)
                

if __name__ == '__main__':
    parser = create_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    main(args)