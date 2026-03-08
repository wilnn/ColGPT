import time
#start_time = time.time()
from transformers import ProcessorMixin
from torchvision import transforms
import re
from transformers import AutoImageProcessor, AutoProcessor, AutoModel, AutoModelForCausalLM, AutoImageProcessor,AutoConfig, AutoTokenizer
from .configuration_llava import CustomLlavaConfig
import torch
from urllib.parse import urlparse
from PIL import Image
import requests
import inspect
import gc
from datetime import datetime
import sys
from .vision_encoder.siglip import SiglipVisionEncoder
from .vision_projector.spp import SPP
from transformers import LlamaForCausalLM
from .modeling_llava import LlavaForCausalLM
from .processing_llava import LlavaProcessor


'''
processor = LlavaProcessor(AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384"),
                           AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),)
#print('done0')
#print(config.image_token_index)  # Should print the index of the image tag in the tokenizer's vocabulary
processor.save_pretrained('./src/model/llava')
#print("done1")
processor = AutoProcessor.from_pretrained('./src/model/llava')
print(type(processor.image_processor))
print(type(processor.tokenizer))


sys.exit(0)
'''
device = 'cuda:0'
vision_config = AutoConfig.from_pretrained("google/siglip-so400m-patch14-384")
vision_encoder = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").vision_model
vision_encoder = SiglipVisionEncoder(vision_model=vision_encoder)

lm_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
language_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')

# if the cli arg has set a custom max length 
custom_max_length = 2000
if custom_max_length:
    max_length = custom_max_length
else:
    max_length = tokenizer.model_max_length

config = CustomLlavaConfig(model_max_length=max_length,
                     hidden_size=lm_config.hidden_size,
                     patch_hidden_size=vision_config.vision_config.hidden_size,
                     vision_projector_type='spp',
                  pyramid_shapes= [[14, 14], [7, 7], [1, 1]],
                  vision_encoder_type ='siplip',
                  vision_encoder_path="google/siglip-so400m-patch14-384",
                  language_model_path= "meta-llama/Llama-3.2-1B-Instruct",)

tokenizer.model_max_length = max_length

vision_projector = SPP(config)


model = LlavaForCausalLM(config=config,
                         vision_encoder=vision_encoder,
                         vision_projector=vision_projector,
                         language_model=language_model).to(device)
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

################## MAKE PROCESSOR ############################

# some models like Llama do not have the padding token in the vocabulary
if not tokenizer.pad_token:
    #self.tokenizer.pad_token = "<pad>"
    #self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # do this to avoid having to resize the embeddings matrix of the model
    tokenizer.pad_token = tokenizer.eos_token

#print(tokenizer.eos_token_id)

#sys.exit(0)

'''
if not tokenizer.unknown_token:
    tokenizer.unk_token = "<unk>"
    tokenizer.add_special_tokens({"unk_token": "<unk>"})
        
if not tokenizer.eos_token:
    custom_eos_token = "<eos>"
    tokenizer.eos_token = custom_eos_token
    tokenizer.add_special_tokens({"eos_token": custom_eos_token})
    # TODO: need to resize the embeddings matrix of the model (the nn.embedding layer)
            #size the vocab size of the tokenizer increased after adding this token into the vocabulary

model.resize_token_embeddings(len(tokenizer))            
'''

# get the max length for the input because some model has really big max length
# that can be a lot for padding during tokenization and to also define
# custom input max length
#if input_max_length:
#tokenizer.model_max_length = config.model_max_length

if config.image_tag not in tokenizer.get_vocab():
    tokenizer.add_tokens([config.image_tag])

# save the id of the image tag in the tokenizer so the config so that
# the config object will now has the correct image token index
config.image_token_index = tokenizer.get_vocab()[config.image_tag]

# save image tag to init_kwargs for use in the processor __call__ method
tokenizer.init_kwargs['image_token'] = config.image_tag
image_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)


####################### TEST ##################################



text = [[{"role": "system",
        "content": "yesssssssssss"},
        {"role": "user",
        "content": "<image>yessssssss"},
        {"role": "assistant",
        "content": "yessssssss", },
        {"role": "user",
        "content": "<image>yessssssss"},
        {"role": "assistant",
        "content": "yessssssss", },
        {"role": "user",
        "content": "<image>yessssssss"},
        {"role": "assistant",
        "content": "yessssssss", },
        ],
        [{"role": "system",
        "content": "nooooooooo"},
        {"role": "user",
        "content": "<image>noooooooo"},
        {"role": "assistant",
        "content": "noooooooo"},
        {"role": "user",
        "content": "<image>yessssssss"},
        {"role": "assistant",
        "content": "yessssssss", },
        {"role": "user",
        "content": "<image>yessssssss"},
        {"role": "assistant",
        "content": "yessssssss", },
        ]]
image = Image.open('./dataset/ColonINST/Positive-images/CVC-ClinicDB/Test/polyp/14.png').convert("RGB")
transform = transforms.ToTensor()
image = transform(image)
images= [[image, image, image], [image, image, image]]

output = processor(images=images,
                   text=text,
                   generation=False,
                   image_path_in_tag=False,
                  give_image_but_no_tag = False,
                 )
'''
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")

print(output['images'][0].shape)
print(output['images'][0].dtype)
print('DONE PROCESSING DATA')
sys.exit(0)'''

######################### TEST PROCESSOR ######################


print(f"input_ids shape: {output['input_ids'].shape}")
print(f"attention_mask shape: {output['attention_mask'].shape}")
print(f"labels shape: {output['labels'].shape}")
print('\n*********************\n')
print(f"input_ids: {output['input_ids'][0]}")
print("##########")
print(f"attention_mask: {output['attention_mask'][0]}")
print("##########")
print(f"labels: {output['labels'][0]}")
print("\n************************\n")
print(f"decoded input_ids: {tokenizer.convert_ids_to_tokens(output['input_ids'][0])}")
#print(f"decoded labels: {tokenizer.convert_ids_to_tokens(output['labels'][0])}")
print(f'image_pos: {output['image_pos']}')
print(f'iamge tag: {config.image_tag}')
print(f'image token id {config.image_token_index}')
print(f'iamge token id 2: {tokenizer.get_vocab()[config.image_tag]}')

print("***************")

print('compare')

'''
for i in range(output['input_ids'][1].shape[0]):
    print(output['input_ids'][1][i], output['labels'][1][i], output['attention_mask'][1][i])
'''
#sys.exit(0)



######################## TEST MODEL ##########################
print('\n######################## TEST MODEL ##########################\n')

output['attention_mask'] = output['attention_mask'].to(device)
output['input_ids'] = output['input_ids'].to(device)
#print(output['input_ids'].requires_grad)
output['labels'] = output['labels'].to(device)
for i in range(len(output['images'])):
    if output['images'][i] is not list:
        output['images'][i] = output['images'][i].to(device)


#input_ids, attention_mask, labels, images_embeds = model(**output, use_cache=False)
# IF PASS use_cache=False, THEN IT WILL NOT RETIURN THE PAS_KEY_VALUES
output2, labels = model(**output, use_cache=False)
import torch.nn as nn
#print(output2)
print(output2['logits'].shape)


# reduction='mean' compute the mean of the per example loss. if not provided,
# it will default to summing all the loss
critetion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
print(labels.shape)
labels = nn.functional.pad(labels, (0, 1), value=-100)
print(labels.shape)
# shift the label
labels= labels[..., 1:].contiguous()
print("555555555")
print(labels.shape)
#output2['logits'] = output2['logits'].float()
logits = output2['logits'].view(-1, output2['logits'].shape[-1])
labels = labels.view(-1)
labels = labels.to(logits.device)
print(logits.shape)

loss = critetion(logits, labels)

print(loss)
loss2 = nn.functional.cross_entropy(logits, labels, ignore_index=-100, reduction='mean')
print(loss2)
if loss == loss2 and loss == output2['loss']:
    print('CORRECT LOSS')
else:
    print('WRONG LOSS')
sys.exit(0)

'''
temp1 = torch.full((246,), 1).to(device)
temp2 = torch.full((246,), -100).to(device)

for i in range(len(output['image_pos'])):
    offset = 0
    offset2 = 0
    for n in range(len(output['image_pos'][i])-1):
        print('\n')
        #print(output['image_pos'][i][n]-offset+offset2, output['image_pos'][i][n]-offset+offset2+246)
        #print(input_ids[i, output['image_pos'][i][n]-offset+offset2:output['image_pos'][i][n]-offset+offset2+246, :].shape)
        if torch.allclose(input_ids[i, output['image_pos'][i][n]-offset+offset2:output['image_pos'][i][n]-offset+offset2+246, :], images_embeds[i][n], atol=0):
            print(f"input_ids, {output['image_pos'][i][n]}, true")
        else:
            print(f"input_ids, {output['image_pos'][i][n]}, false")

        if torch.allclose(attention_mask[i, output['image_pos'][i][n]-offset+offset2:output['image_pos'][i][n]-offset+offset2+246], temp1, atol=0):
            
            print(f"attention_mask, {output['image_pos'][i][n]}, true")
        else:
            print(f"attention_mask, {output['image_pos'][i][n]}, false")
        
        if torch.allclose(labels[i, output['image_pos'][i][n]-offset+offset2:output['image_pos'][i][n]-offset+offset2+246], temp2, atol=0):
            print(f"labels, {output['image_pos'][i][n]}, true")
        else:
            print(f"labels, {output['image_pos'][i][n]}, false")
        offset +=1
        offset2 += 246


sys.exit(0)
'''


































vision_encoder = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").vision_model 
language_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")



# some models like Llama do not have the padding token in the vocabulary
if not tokenizer.pad_token:
    #self.tokenizer.pad_token = "<pad>"
    #self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # do this to avoid having to resize the embeddings matrix of the model
    tokenizer.pad_token = tokenizer.eos_token

#print(tokenizer.eos_token_id)

#sys.exit(0)

'''
if not tokenizer.unknown_token:
    tokenizer.unk_token = "<unk>"
    tokenizer.add_special_tokens({"unk_token": "<unk>"})
        
if not tokenizer.eos_token:
    custom_eos_token = "<eos>"
    tokenizer.eos_token = custom_eos_token
    tokenizer.add_special_tokens({"eos_token": custom_eos_token})
    # TODO: need to resize the embeddings matrix of the model (the nn.embedding layer)
            #size the vocab size of the tokenizer increased after adding this token into the vocabulary

model.resize_token_embeddings(len(tokenizer))            
'''

# get the max length for the input because some model has really big max length
# that can be a lot for padding during tokenization and to also define
# custom input max length
#if input_max_length:
#tokenizer.model_max_length = config.model_max_length

if config.image_tag not in tokenizer.get_vocab():
    tokenizer.add_tokens([config.image_tag])

# save the id of the image tag in the tokenizer so the config so that
# the config object will now has the correct image token index
config.image_token_index = tokenizer.get_vocab()[config.image_tag]

# save image tag to init_kwargs for use in the processor __call__ method
tokenizer.init_kwargs['image_token'] = config.image_tag
image_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", use_fast=True).image_processor

processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

#print(image[0].shape)
#image = torch.stack(image)

text = [[{"role": "system",
        "content": "yesssssssssss"},
        {"role": "user",
        "content": "<image>yessssssss"},
        {"role": "assistant",
        "content": "yessssssss", },
        {"role": "user",
        "content": "<image>yessssssss"},
        {"role": "assistant",
        "content": "yessssssss"},
        {"role": "user",
        "content": "<image>yessssssss"},
        {"role": "assistant",
        "content": "yessssssss"},
        ],
        [{"role": "system",
        "content": "nooooooooo"},
        {"role": "user",
        "content": "<image>noooooooo"},
        {"role": "assistant",
        "content": "noooooooo"},
        {"role": "user",
        "content": "<image>noooooooo"},
        {"role": "assistant",
        "content": "noooooooo"},
        {"role": "user",
        "content": "<image>noooooooo"},
        {"role": "assistant",
        "content": "noooooooo"}
        ]]
image = Image.open('./project/ColonGPT/dataset/ColonINST/Positive-images/CVC-ClinicDB/Test/polyp/14.png')

images= [[image], [image]]

output = processor(images=images, text=text, generation=False)
'''
{
    'images':images,
    'image_pos': image_pos,
    'input_ids':input_ids,
    'attention_mask':attention_mask if attention_mask else None,
    'labels':labels if labels else None
}
'''
print(f"input_ids shape: {output['input_ids'].shape}")
print(f"attention_mask shape: {output['attention_mask'].shape}")
print(f"labels shape: {output['labels'].shape}")
print('\n*********************\n')
print(f"input_ids: {output['input_ids'][0]}")
print("##########")
print(f"attention_mask: {output['attention_mask'][0]}")
print("##########")
print(f"labels: {output['labels'][0]}")
print("\n************************\n")
print(f"decoded input_ids: {tokenizer.convert_ids_to_tokens(output['input_ids'][0])}")
#print(f"decoded labels: {tokenizer.convert_ids_to_tokens(output['labels'][0])}")
print(f'image_pos: {output['image_pos']}')
print(f'iamge tag: {config.image_tag}')
print(f'image token id {config.image_token_index}')
print(f'iamge token id 2: {tokenizer.get_vocab()[config.image_tag]}')

print("***************")

print('compare')

for i in range(output['input_ids'][0].shape[0]):
    print(output['input_ids'][0][i], output['labels'][0][i], output['attention_mask'][0][i])


sys.exit(0)








tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
#tokenizer.init_kwargs['image_token'] = '<image>'
tokenizer.pad_token=tokenizer.eos_token
tokenizer.chat_template = 'sdfs'
tokenizer.model_max_length = 2048
tokenizer.save_pretrained('./src')
tokenizer.truncation_side = 'left'

tokenizer.save_pretrained("./src")

tokenizer = AutoTokenizer.from_pretrained("./src")
print(tokenizer.chat_template)
print(tokenizer.pad_token)
print(tokenizer.truncation_side)


'''
processor = LlavaProcessor(AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384"),
                           AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"), llava_config=config,)

print(config.image_token_index)  # Should print the index of the image tag in the tokenizer's vocabulary
#processor.save_pretrained('./src/model_arch')'''