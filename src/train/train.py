import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from .train_args import create_parser, trainArgs
from torchvision import transforms
import re
from transformers import (ProcessorMixin, HfArgumentParser, Trainer, get_scheduler,
                        AutoProcessor, AutoModel, AutoModelForCausalLM,
                        AutoImageProcessor,AutoConfig, AutoTokenizer)
from ..model.configuration_llava import CustomLlavaConfig
import torch
from urllib.parse import urlparse
from PIL import Image
import requests
import inspect
import gc
import sys
from ..model.vision_encoder.siglip import SiglipVisionEncoder
from ..model.vision_projector.spp import SPP
from ..model.modeling_llava import LlavaForCausalLM
from ..model.processing_llava import LlavaProcessor
import ast
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb
from torch import optim
import math 
from tqdm.auto import tqdm
from accelerate.utils import set_seed
import wandb


from pathlib import Path

parser = HfArgumentParser((trainArgs,))
args = parser.parse_args_into_dataclasses()[0]


class LlavaDataset(Dataset):
    def __init__(self, path_to_json,
                 path_to_image_folder='.',
                 system_prompt=None,
                 data_processor=None,
                 dataset_size=None,
                 ):
        # Example data: features and labels
        with open(path_to_json, "r") as f:
            self.data = json.load(f)  # returns a list of dict(json)
        if dataset_size > 0:
            self.data = self.data[:dataset_size]
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
            img = [Image.open(self.path_to_image_folder+ '/' + self.data[idx]['image'])]
        elif type(self.data[idx]['image']) is list:
            img = [Image.open(self.path_to_image_folder + '/' + i) for i in self.data[idx]['image']]
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

def collate_fn(batch):
    
    return {
        'images':[n['images'][0] for n in batch],
        'image_pos': [n['image_pos'][0] for n in batch],
        'input_ids':torch.cat([n['input_ids'] for n in batch], dim=0),
        'attention_mask':torch.cat([n['attention_mask'] for n in batch], dim=0),
        'labels':torch.cat([n['labels'] for n in batch], dim=0),
    }


def create_and_prepare_model(args):
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

    config = CustomLlavaConfig(model_max_length=max_length,
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

    return model, config, tokenizer

def create_and_prepare_data_processor(args, tokenizer, config):
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
    
    return processor

def compute_metrics_fn(p):
    return {"metric1": 0.5}

def train(model, train_dataset, val_dataset):
    print("11111111111")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #tokenizer=tokenizer,
        data_collator=collate_fn,
        #compute_loss_fun=compute_loss,
        compute_metrics=compute_metrics_fn,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=4,
                                         #early_stopping_threshold=0.001,
                                         #)],
    )

    trainer.train()
    return

if __name__ == '__main__':
    set_seed(args.seed)
    #####################################################
    ##### CRAEATE, SET UP MODEL AND PROCESSOR ###########
    #####################################################
    model, config, tokenizer =  create_and_prepare_model(args)
    
    ################## MAKE PROCESSOR ############################
    print("222222222")
    processor = create_and_prepare_data_processor(args, tokenizer, config)

    #####################################################
    ##### CRAEATE, SET UP OPTIMIZER, LR SCHEDULER, DATASET, DATALOADER, ETC., ###########
    #####################################################
    print("3333333")
    system_prompt = 'You are a colonoscopy assistant providing help with colonoscopy tasks'
    train_dataset = LlavaDataset(path_to_json=args.train_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           )
    if args.training_stage == 1:
        val_dataset = LlavaDataset(path_to_json=args.cap_val_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           )
    elif args.training_stage == 2:
        val_dataset = {
            "rec": LlavaDataset(path_to_json=args.rec_val_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           ),
            "cls": LlavaDataset(path_to_json=args.cls_val_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           ),
            "reg": LlavaDataset(path_to_json=args.reg_val_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           ),
        }
    print("44444444")
    
    train(model, train_dataset=train_dataset, val_dataset=val_dataset)


























