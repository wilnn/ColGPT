import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from .train_args import create_parser, trainArgs
from torchvision import transforms
import re
from transformers import (ProcessorMixin, HfArgumentParser, get_scheduler,
                        AutoProcessor, AutoModel, AutoModelForCausalLM,
                        AutoImageProcessor,AutoConfig, AutoTokenizer, LlamaConfig)
from .trainer import LlavaTrainer 
from ..model.configuration_llava import CustomLlavaConfig
import torch
from urllib.parse import urlparse
from PIL import Image
import requests
import inspect
import gc
import sys
from ..model.vision_encoder.siglip import SiglipVisionEncoder
from ..model.language_model.llama.modeling_llama import LlamaForCausalLM

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
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix


from pathlib import Path

parser = HfArgumentParser((trainArgs,))
args = parser.parse_args_into_dataclasses()[0]
if args.resume_from_checkpoint is not None:
    if args.resume_from_checkpoint == "True":
        args.resume_from_checkpoint = True
    elif (args.resume_from_checkpoint == "None" or
	args.resume_from_checkpoint == "False"):
        args.resume_from_checkpoint = None


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
        self.role_map = {
            "system":"system",
            "gpt": "assistant",
            "human": "user"
        }
        
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

        temp = [{'role':'system', 'content':self.system_prompt}] if self.system_prompt else []
        
        if fix_keys_to_role_content:  
            temp += [{'role': self.role_map[i['from']],'content': i['value'],} for i in self.data[idx]['conversations']]
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
        'return_sum_loss': True,
        'return_num_tokens': True,
        'return_labels': True,
    }


def create_and_prepare_model(args):
    vision_config = AutoConfig.from_pretrained(args.pretrained_vision_encoder_path).vision_config
    vision_encoder = AutoModel.from_pretrained(args.pretrained_vision_encoder_path).vision_model
    vision_encoder = SiglipVisionEncoder(vision_model=vision_encoder)

    lm_config = AutoConfig.from_pretrained(args.pretrained_language_model_path)
    if args.LM_class == "llama":
        language_model = LlamaForCausalLM.from_pretrained(args.pretrained_language_model_path, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported language model class {args.LM_class}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_language_model_path)

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
    
    elif args.training_stage == 2: # STAGE 2. ONLY TRAIN THE LM
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

def create_and_prepare_data_processor(args, tokenizer, config, model):
    # some models like Llama do not have the padding token in the vocabulary
    # HF make the tokenizer.pad_token attribute in the tokenizer to be None if the model
    # does not has pad token. 
    if tokenizer.pad_token is None:
        # if the model does not has eos_token then do this
        # NOTE THAT DOING THIS REQUIRE YOU TO RESIZE THE EMBEDDINGS MATRIX OF THE 
        # MODEL SINCE YOU ADDED NEW TOKEN TO THE TOKENIZER. 
        if tokenizer.eos_token is None:
            tokenizer.pad_token = "<pad>"
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.get_language_model().resize_token_embeddings(len(tokenizer))
        else:
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

    image_processor = AutoImageProcessor.from_pretrained(args.pretrained_vision_encoder_path)

    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    return processor

def get_datasets(system_prompt, processor):
    train_dataset = LlavaDataset(path_to_json=args.train_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           )
    if args.training_stage == 1:
        num_valset = 1
        val_dataset = LlavaDataset(path_to_json=args.cap_val_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           )
        test_dataset = LlavaDataset(path_to_json=args.cap_test_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           )
    elif args.training_stage == 2:
        num_valset = 3
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

        test_dataset = {
            "rec": LlavaDataset(path_to_json=args.rec_test_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           ),
            "cls": LlavaDataset(path_to_json=args.cls_test_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           ),
            "reg": LlavaDataset(path_to_json=args.reg_test_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           ),
            }
        if args.stage2_with_cap:
            num_valset = 4
            val_dataset['cap'] = LlavaDataset(path_to_json=args.cap_val_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           )

            test_dataset['cap'] = LlavaDataset(path_to_json=args.cap_test_set_path,
                           path_to_image_folder=args.image_path,
                           system_prompt=system_prompt,
                           data_processor=processor,
                           dataset_size=args.max_dataset_size,
                           )
    return train_dataset, val_dataset, test_dataset, num_valset

class ComputeValMetrics:
    def __init__(self, tokenizer, num_valset, iou_threshold=0.5):
        self.num_valset = num_valset
        
        self.valset_count = 0
        self.tokenizer = tokenizer
        self.LM_pad_token_ids = tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.iou_threshold = iou_threshold
        self.sum_loss = 0
        self.num_tokens = 0
        self.total_sum_loss = 0
        self.total_num_tokens = 0
        self.reg_ids = {'UNKNOWN': 0, 'adenomatous': 1, 'high grade adenoma': 2,
                        'ulcer': 3, 'adenocarcinoma': 4, 'invasive carcinoma': 5,
                        'instrument': 6, 'blood fresh': 7, 'blood hematin': 8,
                        'high grade dysplasia': 9, 'angiectasia': 10,
                        'lymphangiectasia': 11, 'suspicious precancerous lesion': 12,
                        'erythema': 13, 'polyp': 14, 'low grade adenoma': 15,
                        'erosion': 16, 'hyperplastic lesion': 17, 'serrated lesion': 18}
        self.cls_ids = {'UNKNOWN': 0, 'adenocarcinoma': 1,
                        'suspicious precancerous lesion': 2, 'UCG1': 3, 'BBPS2': 4,
                        'high grade adenoma': 5, 'erythema': 6, 'dyed lifted polyp': 7,
                        'ulcerative colitis': 8, 'UCG0-1': 9, 'cecum': 10,
                        'lymphangiectasia': 11, 'hyperplastic lesion': 12,
                        'blood fresh': 13, 'accessory tool': 14, 'polyp': 15,
                        'low grade adenoma': 16, 'blood hematin': 17,
                        'invasive carcinoma': 18, 'erosion': 19, 'colon diverticula': 20,
                        'UCG2': 21, 'retroflex rectum': 22, 'blood in lumen': 23,
                        'BBPS1': 24, 'ileocecal valve': 25, 'UCG3': 26, 'BBPS3': 27,
                        'high grade dysplasia': 28, 'colorectal cancer': 29,
                        'resected polyp': 30, 'resection margin': 31, 'angiectasia': 32,
                        'BBPS0-1': 33, 'BBPS2-3': 34, 'dyed resection margin': 35,
                        'UCG1-2': 36, 'serrated lesion': 37, 'UCG2-3': 38,
                        'hemorrhoid': 39, 'ulcer': 40, 'adenoma': 41, 'BBPS0': 42,
                        'inflammatory bowel disease': 43}
        
        self.cls_matrix = np.zeros((len(self.cls_ids), len(self.cls_ids)),
                        dtype=np.int64) # shape (num classes, num classes)
        self.reg_matrix = np.zeros((len(self.reg_ids), len(self.reg_ids)),
                        dtype=np.int64) # shape (num classes, num classes)
        self.IoU = []
        
    def reset(self):
        self.sum_loss = 0
        self.num_tokens = 0
        self.cls_matrix = np.zeros((len(self.cls_ids), len(self.cls_ids)),
                        dtype=np.int64) # shape (num classes, num classes)
        self.reg_matrix = np.zeros((len(self.reg_ids), len(self.reg_ids)),
                        dtype=np.int64) # shape (num classes, num classes)
        self.IoU = []

        if self.valset_count == self.num_valset:
            self.valset_count = 0
            self.total_sum_loss = 0
            self.total_num_tokens = 0
    
    def extract_label_prediction(self, eval_pred):
        predictions = []
        labels = []

        new_labels = nn.functional.pad(eval_pred.predictions[3], (0, 1),
                                                    value=-100)
        new_labels = new_labels[..., 1:].contiguous()
        for i in range(eval_pred.predictions[3].shape[0]):
            mask = new_labels[i] != -100
            predicted_ids = torch.argmax(eval_pred.predictions[0][i][mask],
                                        dim=-1)

            predictions.append(self.tokenizer.decode(predicted_ids,
                                                    skip_special_tokens=True))
            new_labels[i][~mask] = self.LM_pad_token_ids # flip the mask so position that
                                            # is padding = true
            labels.append(self.tokenizer.decode(new_labels[i],
                                                    skip_special_tokens=True))
        return predictions, labels
    
    def cls_reg_accum(self, eval_pred, metric_key_prefix):
        lb_ids = self.reg_ids if "reg" in metric_key_prefix else self.cls_ids
        
        predictions, labels = self.extract_label_prediction(eval_pred=eval_pred)
            #print(predictions[i])
            #print(labels[i])
        for i in range(len(predictions)):
            predictions[i] = "UNKNOWN" if predictions[i] not in lb_ids else predictions[i]
            labels[i] = "UNKNOWN" if labels[i] not in lb_ids else labels[i]
            labels[i] = lb_ids[labels[i]]
            predictions[i] = lb_ids[predictions[i]]

        cm_batch = confusion_matrix(
                        y_true=np.array(labels),
                        y_pred=np.array(predictions),
                        labels=np.arange(len(lb_ids))
                    )
        if "cls" in metric_key_prefix:
            self.cls_matrix += cm_batch

        elif "reg" in metric_key_prefix:
            self.reg_matrix += cm_batch

    def cls_reg_final_metrics(self, metric_key_prefix):
        cm_total = self.cls_matrix if "cls" in metric_key_prefix else self.reg_matrix
        
        cm_total = cm_total[1:, 1:] # remove the first row and firs column 
                                    # because of the "UNKNOWN" class

        tp = np.diag(cm_total)
        fp = cm_total.sum(axis=0) - tp
        fn = cm_total.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-8) # add very small number
                                        # to avoid division by 0
        recall = tp / (tp + fn + 1e-8)

        f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
        macro_f1 = np.mean(f1_per_class)
        uar = recall.mean()
        return macro_f1, uar
    
    def parse_box(self, box_str):
        pattern = r"^\{\<(\d+)\>\<(\d+)\>\<(\d+)\>\<(\d+)\>\}$"
        match = re.match(pattern, box_str)
        if not match:
            return "Model does not give output in the correct format"
        return [int(n) for n in match.groups()]
    
    def rec_accum(self, eval_pred, metric_key_prefix):
        # box = [x1, y1, x2, y2]
        
        predictions, labels = self.extract_label_prediction(eval_pred=eval_pred)
        for i in range(len(predictions)):
            box1 = self.parse_box(predictions[i])
            if isinstance(box1, str): # model not output bounding box in correct format
                self.IoU.append(0.0)
            else:
                box2 = self.parse_box(labels[i])
                # Intersection coordinates
                xA = max(box1[0], box2[0])
                yA = max(box1[1], box2[1])
                xB = min(box1[2], box2[2])
                yB = min(box1[3], box2[3])

                # Intersection area
                inter_width = max(0, xB - xA)
                inter_height = max(0, yB - yA)
                inter_area = inter_width * inter_height

                # Areas of each box
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                # Union
                union = area1 + area2 - inter_area
                
                IoU = inter_area / (union + 1e-8)
                self.IoU.append(IoU)
    
    def rec_final_metrics(self):
        IoU = np.array(self.IoU)
        acc_iou = np.mean(IoU >= self.iou_threshold)
        return IoU.mean(), acc_iou
        
    def __call__(self, eval_pred=None,
                 compute_result: bool = False,
                 metric_key_prefix=None,):
        #logits, labels = eval_pred.predictions, eval_pred.label_ids

        # when use batch_eval_metrics, everything is still tensor not converted to 
        # numpy array

        self.sum_loss += eval_pred.predictions[1].sum().item()
        self.num_tokens += eval_pred.predictions[2].sum().item()

        #eval_pred.label_ids[eval_pred.label_ids == -100] = 128009
        #self.tokenizer.decode(eval_pred.label_ids, skip_special_tokens=True)
        #eval_pred.predictions[3] # new labels
        if "cls" in metric_key_prefix or "reg" in metric_key_prefix:
            self.cls_reg_accum(eval_pred=eval_pred,
                                 metric_key_prefix=metric_key_prefix)
        
        if "rec" in metric_key_prefix:
            self.rec_accum(eval_pred=eval_pred,
                                 metric_key_prefix=metric_key_prefix)

        # FINAL CALL: return metric
        if compute_result:
            self.valset_count += 1
            self.total_sum_loss += self.sum_loss
            self.total_num_tokens += self.num_tokens
            metrics = {}
            perx = math.exp(self.sum_loss / self.num_tokens) if self.num_tokens > 0 else 0.0
            metrics["perplexity"] = perx
            if self.valset_count == self.num_valset and self.num_valset > 1:
                metrics["overall_perplexity"] = math.exp(self.total_sum_loss / self.total_num_tokens) if  self.total_num_tokens > 0 else 0.0
            
            if "cls" in metric_key_prefix or "reg" in metric_key_prefix:
                macro_f1, uar = self.cls_reg_final_metrics(metric_key_prefix=metric_key_prefix)
                metrics["macro_f1"] = macro_f1.item()
                metrics["UAR"] = uar.item()
            
            if "rec" in metric_key_prefix:
                mean_iou, acc_iou = self.rec_final_metrics()
                metrics["mean_IoU"] = mean_iou.item()
                metrics["acc_IoU"] = acc_iou.item()

            self.reset()
            return metrics

        return {}  # return empty dict while still accumulating value

def train(model, train_dataset, val_dataset, compute_metrics):
    trainer = LlavaTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #tokenizer=tokenizer,
        data_collator=collate_fn,
        #compute_loss_fun=compute_loss,
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=4,
                                         #early_stopping_threshold=0.001,
                                         #)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    return trainer


def test(trainer, test_dataset):
    # if give a dict of dataset, then the return metric dict 
    # is a dict of all computed metrics for all dataset. +
    print("Testing the trained model on test set:")
    metrics = trainer.evaluate(eval_dataset=test_dataset,
        ignore_keys=None,
        metric_key_prefix="test",
    ) 

    return metrics

if __name__ == '__main__':
    set_seed(args.seed)
    #####################################################
    ##### CRAEATE, SET UP MODEL AND PROCESSOR ###########
    #####################################################
    model, config, tokenizer =  create_and_prepare_model(args)
    
    ################## MAKE PROCESSOR ############################
    processor = create_and_prepare_data_processor(args, tokenizer, config, model)

    system_prompt = 'You are a colonoscopy assistant providing help with colonoscopy tasks.'
    train_dataset, val_dataset, test_dataset, num_valset = get_datasets(system_prompt=system_prompt,
                                                            processor=processor)
    

    compute_metrics = ComputeValMetrics(processor.tokenizer, num_valset=num_valset,
                                        iou_threshold=args.iou_threshold)
    trainer = train(model, train_dataset=train_dataset, val_dataset=val_dataset,
        compute_metrics=compute_metrics)

    # check if the embedding matrix and the final lm_head in the language model are tied
    # if tied, it will print out true
    print(f"############{model.get_language_model().lm_head.weight.data_ptr() == model.get_language_model().model.embed_tokens.weight.data_ptr()}")
    
    metrics = test(trainer=trainer, test_dataset=test_dataset)

























