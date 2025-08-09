from transformers import ProcessorMixin
import re
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoImageProcessor,AutoConfig, AutoTokenizer
from configuration_llava import LlavaConfig
import torch
from urllib.parse import urlparse
from PIL import Image
import requests
import inspect
import gc
from datetime import datetime
import sys
from vision_encoder.siglip import SiglipVisionEncoder
from vision_projector.spp import SPP
from language_model.llama.modeling_llama import LlamaForCausalLM
from modeling_llava import LlavaForCausalLM

#from modeling_llava import LlavaForCausalLM

class LlavaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    
    def __init__(self, image_processor, tokenizer,
     #custom_chat_template=None, input_max_length=None, llava_config=None,custom_eos_token=None,
                **kwargs):
        super().__init__(image_processor, tokenizer)
        
        '''
        # 
        # some models like Llama do not have the padding token in the vocabulary
        if not self.tokenizer.pad_token:
            #self.tokenizer.pad_token = "<pad>"
            #self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            # do this to avoid having to resize the embeddings matrix of the model
            self.tokenizer.pad_token = self.tokenizer.eos_token
        '''

        '''
        if not self.tokenizer.unknown_token:
            self.tokenizer.unk_token = "<unk>"
            self.tokenizer.add_special_tokens({"unk_token": "<unk>"})
        '''

        '''        
        if not self.tokenizer.eos_token:
            custom_eos_token = custom_eos_token if custom_eos_token else "<eos>"
            self.tokenizer.eos_token = custom_eos_token
            self.tokenizer.add_special_tokens({"eos_token": custom_eos_token})
            # TODO: need to resize the embeddings matrix of the model (the nn.embedding layer)
                    #size the vocab size of the tokenizer increased after adding this token into the vocabulary
        '''

        '''
        # get the max length for the input because some model has really big max length
        # that can be a lot for padding during tokenization and to also define
        # custom input max length
        if input_max_length:
            tokenizer.model_max_length = model_max_length

        llava_config.model_max_length = tokenizer.model_max_length

        #self.model_max_length = llava_config.model_max_length

        if llava_config.image_tag not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([llava_config.image_tag])
        # save the id of the image tag in the tokenizer so the config so that
        # the config object will now has the correct image token index
        llava_config.image_token_index = self.tokenizer.get_vocab()[llava_config.image_tag]

        # save image tag to init_kwargs for use in the processor __call__ method
        self.tokenizer.init_kwargs['image_token'] = llava_config.image_tag
        
        #self.llava_config = llava_config'''

        extensions = ["png", "jpg", "jpeg", "gif", "bmp", "webp"]
        self.img_tag_pattern = re.compile(
            r'<image path="([^"]+\.(?:' + '|'.join(extensions) + r'))">',
            re.IGNORECASE
        )

        '''if custom_chat_template:
            self.chat_template = custom_chat_template
            self.tokenizer.chat_template = custom_chat_template # save this chat template to the tokenizer
        else:                                                   # which will be saved to the tokenizer.json when
                                                                # save this processor
            if self.tokenizer.chat_template:
                self.chat_template = self.tokenizer.chat_template
            else:
                # use the default chat template of the meta-llama/Llama-3.2-1B-Instruct model since it looks fine enough
                self.chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
                self.tokenizer.chat_template = self.chat_template'''

        self.roles=['system', 'user', 'assistant']
        '''self.roles = re.findall(r"\{\{\s*message\['role'\]\s*==\s*['\"](.*?)['\"]\s*\}\}", self.tokenizer.chat_template)
        if not self.roles:
            self.roles = re.findall(r"role\s*==\s*['\"](.*?)['\"]", self.tokenizer.chat_template)'''

    def format_prompt_for_llava_training(
        self,
        system_header=None,
        system_content=None,
        user_header=None,
        user_content=None,
        assistant_header=None,
        assistant_content=None,
    ):
        temp = ''
        if system_header:
            temp+=f'<|start_header_id|>{system_header}<|end_header_id|>'
        if system_content:
            date_string = datetime.now().strftime("%d %b %Y")
            temp+= f'\n\nCutting Knowledge Date: December 2023\nToday Date:{date_string}\n\n{system_content}<|eot_id|>'
        
        if user_header:
            temp+=f'<|start_header_id|>{user_header}<|end_header_id|>'
        if user_content:
            temp+=f'\n\n{user_content}<|eot_id|>'
        if assistant_header:
            temp+=f'<|start_header_id|>{assistant_header}<|end_header_id|>'
        if assistant_content:
            temp+=f'\n\n{assistant_content}<|eot_id|>'
        return temp
    
    def __call__(self, text, images=None, 
                 #image_tag_in_prompt=True,
                 image_path_in_tag=False,
                 give_image_but_no_tag = False,
                 generation=True,
                 custom_chat_template=None,
                 roles: list[str] = None,
                 ensure_match_roles=False, 
                 #format_prompt_for_llava_training=None,
                 **kwargs):
        """
        Args:
            image_tag_in_prompt (bool):
                if False, that means the prompt do not have the image tag AND still provide the image
                the image tag will be added to the prompt 
            images (list):
                list of list of images. the conservation in text that has no images will have empty list
            text (list[list[dict]]):
                list of lists of dicts, each dict has a role and content key
         custom_chat_template (str) is of the Jinja2 template format.
                If provided, it will be used instead of the default template.
            roles (list[str]) is a list of roles that the text should match.
                If provided, it will be used to ensure that the text matches the roles.
            ensure_match_roles (bool): if True, it will fix the role key in dict in text does not match the roles.
                If False, it will not check the roles.
            
            returns:
                return a dict that has:
                {
                    'image_pos': a list of tensor and empty list. 
                    'images': a list of list of tensor and empty list. 
                            empty list indicate the corresponding example in the batch
                            do not have any images. 
                    'input_ids':a tensor of shape (batch size, model_max_length)
                    'labels': tensor of shape (batch size, model_max_length)
                    'attention_mask': tensor of shape (batch size, model_max_length)
                }
            
            """

        ##################################################################
        ###################### CHECK ARGUMENTS #####################
        ##################################################################
        
        if not len(text):
            raise ValueError("Text must be a non-empty list of lists of dicts OR list of dicts")

        if isinstance(text[0], dict):
            text = [text] # convert to list of lists of dicts to make it into a batch like format
        
        if custom_chat_template and not roles or roles and not custom_chat_template:
            raise ValueError("If custom_chat_template or roles is provided, the other one must also be provided.")
        
        
        if not custom_chat_template:
            roles = self.roles

        if image_path_in_tag and images:
            raise ValueError("Cannot have both images and image_path_in_tag set to True. Please choose one.")
        
        # check if the roles in the text match the roles in the template.
        # useful when the user is unsure what to name the role as given the chat template
        # but doing this can slow down the processing.
        if ensure_match_roles:
            temp_roles = set(roles)  # ensure roles is a set for faster lookup (O(1) search time)
            for conversation in text:
                for message in conversation:
                    if message['role'] not in temp_roles:
                        raise ValueError(f"Role {message['role']} is not in the roles {roles}. Please check your text and roles.")
        
        # turn the text into a batch of conversation (list of list of dict)
        if type(text) is dict:
            text = [[text]]
        if type(text) is list and type(text[0]) is not list:
            text = [text]
        
        ##################################################################
        ###################### preprocess the images #####################
        ##################################################################
        
        # CASE WHEN NOT PROVIDE THE IMAGES BUT THE IMAGE TAG HAS THE PATH
        if image_path_in_tag and not images:
            images = []
            #user_role = roles[1] if len(roles) > 2 else roles[0] # if len(role)> 2 then it may also has the system role at index 0
            
            for i, conversation in enumerate(text): # for each coversation in the batch
                imgs2 = [] # images in one coversation in order from first to last message from left to right of each message.
                
                for n, message in enumerate(conversation): # for each message in the conversation
                    #if message['role'] == user_role: # if the message is from the user, then we can check the image tags
                    image_paths  = self.img_tag_pattern.findall(message['content']) # get the image path of image tags if the image tag has a path
                    
                    if image_paths: # if found the image path in the image tag
                        text[i][n] = self.img_tag_pattern.sub("<image>", message['content'])
                        
                        for path in image_paths:
                            if urlparse(path).scheme in ("http", "https"): # if it is the url image path
                                image = Image.open(requests.get(path, stream=True).raw)
                            else: # it is probably a local image path
                                image = Image.open(path)

                            imgs2.append(image)
                    
                    #imgs.append(imgs2) # append the images in one message to the images in one conversation
                images.append(imgs2)


        # CASE WHEN THE USER DO NOT PROVIDE THE IMAGES (PROMPT WITH NO IMAGE)
        if not images and not image_path_in_tag:
            images = []
            for i in range(len(text)):
                temp = [] 
                images.append(temp)  # if no images are provided, then create an empty list of lists of images for each conversation
        
        else:
            ############ START PREPROCESSING THE IMAGES ############
            filtered_kwargs = {}
            if kwargs: # if passed something to kwarg, then, it may be arg for the image processor or tokenizer
                
                # since some .preprocess methods of some image processor
                # do not have the kwargs parameters, passing kwargs may raise an error if kwargs
                # contains the parameters that the method can take in.
                # Therefore, have to do this:

                # get the list of parameters of the preprocess method
                params = set(inspect.signature(self.image_processor.preprocess).parameters.keys())
                
                # get the parameters and arguments that this preprocess method can take in
                for k, v in kwargs.items():
                    if k in params:
                        filtered_kwargs[k] = v
                        kwargs.pop(k)  # remove the key from kwargs to avoid passing it to the tokenizer later
            
            preprocessed_images = []
            for i in images: # reprocess each image in the list of lists of images 
                            # because the image processor can take only in an image or a list of images 
                temp = []
                if i:
                    temp = self.image_processor.preprocess(i, return_tensors='pt',
                                                **filtered_kwargs)['pixel_values'] # return a tensor of shape (num images, channel, height, weidth)
                preprocessed_images.append(temp)
                
                #if isinstance(temp, list):
                    #temp = torch.cat(temp, dim=0)  # concatenate the list of tensors into a single tensor
                            
            images = preprocessed_images  # replace the images with the preprocessed images
            
            #preprocessed_images = torch.stack(preprocessed_images)
            #del temp, images 
            #gc.collect()
        
        ###############################################################
        ##################### preprocess the text #####################
        ###############################################################
        
        filtered_kwargs = {}
        if kwargs: # if passed something to kwarg, then, it may be arg for the image processor or tokenizer
            
            # since some .preprocess methods of some image processor
            # do not have the kwargs parameters, passing kwargs may raise an error if kwargs
            # contains the parameters that the method can take in.
            # Therefore, have to do this:

            # get the list of parameters of the preprocess method
            params = set(inspect.signature(self.tokenizer).parameters.keys())
            
            # get the parameters and arguments that this preprocess method can take in
            for k, v in kwargs.items():
                if k in params:
                    filtered_kwargs[k] = v
                    kwargs.pop(k)  # remove the key from kwargs to avoid passing it to the tokenizer later


        '''
        general idea to tokenize the prompt for training:
        input_ids: will have the bos token and do not have the eos token after the very
        last message content(this eos token that indicate the end of prompt message may
        not be the actual final eos token) (the content of the message that is not the
        last message still has this eos token)

        labels: the label will not have the bos token but all message contents will have the eos
        token including the last message content. only the message content of the message that 
        is from assistant is kept(the assistant header will still be ignored) everything else 
        will be ignore(index -100)

        the attention mask will have 1 for everything except the padding token
        
        '''


        template = custom_chat_template if custom_chat_template else self.tokenizer.chat_template
        tokenize=True if 'tokenize' not in filtered_kwargs else filtered_kwargs['tokenize']
        max_length = self.tokenizer.model_max_length if 'max_length' not in filtered_kwargs else filtered_kwargs['max_length']
        truncation=True if 'truncation' not in filtered_kwargs else filtered_kwargs['truncation']
        padding=False
        return_type='pt'

        ############### TRAIN CASE ################
        if not generation: 
            return_dict = True if 'return_dict' not in filtered_kwargs else filtered_kwargs['return_dict']

            labels= []
            input_ids = []
            attention_mask = []
            for i, conversation in enumerate(text):
                one_labels = []
                one_input_ids = []
                one_attention_mask = []

                remain_length = max_length
                for n, message in enumerate(conversation):
                    if message['role'] == roles[0]:
                        out = self.format_prompt_for_llava_training(system_header=message['role'], system_content=message['content'])
                        out = self.tokenizer(out,
                                        #return_dict=return_dict, # there is no return_dict argument here
                                        max_length=remain_length, truncation=truncation,
                                        padding=False, add_special_tokens=False,
                                        return_tensors='pt',)
                        dtype = out['input_ids'].dtype
                        remain_length -= out['input_ids'].shape[1]
                    elif message['role'] == roles[1]:
                        out = self.format_prompt_for_llava_training(user_header=message['role'], user_content=message['content'])
                        out = self.tokenizer(out,
                                        #return_dict=return_dict, 
                                        max_length=remain_length, truncation=truncation,
                                        padding=False, add_special_tokens=False,
                                        return_tensors='pt',)
                        dtype = out['input_ids'].dtype
                        remain_length -= out['input_ids'].shape[1]
                    else:
                        assistant_header = self.format_prompt_for_llava_training(assistant_header=message['role'])
                        assistant_content = self.format_prompt_for_llava_training(assistant_content=message['content'])

                        assistant_header= self.tokenizer(assistant_header,
                                        #return_dict=return_dict, 
                                        max_length=remain_length, truncation=truncation,
                                        padding=False, add_special_tokens=False,
                                        return_tensors='pt',)
                        dtype = assistant_header['input_ids'].dtype
                        remain_length = remain_length - assistant_header['input_ids'].shape[1]
                        if remain_length > 0: # there is a chance that after the one above, 
                                            # remain_length will be 0.
                            assistant_content = self.tokenizer(assistant_content,
                                            #return_dict=return_dict, 
                                            max_length=remain_length, truncation=truncation,
                                            padding=False, add_special_tokens=False,
                                            return_tensors='pt',)
                            remain_length = remain_length - assistant_content['input_ids'].shape[1]
                        else:
                            assistant_content = None
                        
                    if n == 0:  #add bos token at the beginning of this input_ids
                                #and also a mask for this bos token
                        one_input_ids.append(torch.tensor([self.tokenizer.bos_token_id], dtype=dtype, device=out['input_ids'].device))
                        one_attention_mask.append(torch.tensor([1], dtype=dtype, device=out['attention_mask'].device))
                        remain_length -= 1
                    
                    if message['role'] == roles[2]:
                        temp = torch.full((assistant_header['input_ids'].shape[1],), -100, dtype=dtype, device=assistant_header['input_ids'].device)
                        one_labels.append(temp)

                        one_input_ids.append(assistant_header['input_ids'][0])
                        
                        one_attention_mask.append(assistant_header['attention_mask'][0])

                        if assistant_content is not None:
                            one_labels.append(assistant_content['input_ids'][0])
                            one_input_ids.append(assistant_content['input_ids'][0])
                            one_attention_mask.append(assistant_content['attention_mask'][0])                        
                    else:
                        temp = torch.full((out['input_ids'].shape[1],), -100, dtype=dtype, device=out['input_ids'].device)
                        one_labels.append(temp)
                        one_input_ids.append(out['input_ids'][0])
                        one_attention_mask.append(out['attention_mask'][0])

                    if remain_length <= 0:
                        print(f"warning: run out of sequence length space, and it may have performed a truncation. the max sequence length is {self.tokenizer.model_max_length} tokens and has {remain_length} tokens space left. You should increase the model_max_length")
                        one_input_ids[-1][-1] = self.tokenizer.pad_token_id
                        one_attention_mask[-1][-1] = 0
                        temp = torch.full((remain_length+1,), -100, dtype=dtype, device=one_labels[-1].device)
                        one_labels.append(temp)
                        break

                    if n >= len(conversation)-1: # if last message in the conversastion
                                                # then wants to add the padding into this input_ids
                                                # (and mask out the eos token that indicate the end of message)
                        one_input_ids[-1][-1] = self.tokenizer.pad_token_id # turn the eos token(which is the
                                                            #last token indicate the end of prompt not the atual)
                                                            # to padding token
                        one_attention_mask[-1][-1] = 0 # masked out the last token that is turned into the padding token above
                        
                        temp = torch.full((remain_length,), self.tokenizer.pad_token_id, dtype=dtype, device=one_input_ids[-1].device)
                        one_input_ids.append(temp)
                        
                        temp = torch.full((remain_length,), 0, dtype=dtype, device=one_attention_mask[-1].device)
                        one_attention_mask.append(temp)
                        
                        # remain_length + 1 here since the labels is currently one token lesser than the
                        # input_ids due to the bos token at the beginning of this input_ids example while the
                        # label do not have this bos token (this is shift left the token)
                        temp = torch.full((remain_length + 1,), -100, dtype=dtype, device=one_labels[-1].device)
                        one_labels.append(temp)
                        break

                attention_mask.append(torch.cat(one_attention_mask))
                labels.append(torch.cat(one_labels))
                input_ids.append(torch.cat(one_input_ids))
            
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)
            input_ids = torch.stack(input_ids)

        else: ############ GENERATE (INFERENCE) CASE ##################
            tokenizer.truncation_side = 'left' # truncate the earlier message if it is too long 
                                            # the default 'right' truncate will remove the 
                                            # generation prompt and not good idea for chat bot
                                            # in general

            return_dict = True if 'return_dict' not in filtered_kwargs else filtered_kwargs['return_dict']
            out = self.tokenizer.apply_chat_template(
                            text, chat_template=template,
                            tokenize=tokenize, return_dict=return_dict, 
                            max_length=max_length, truncation=truncation,
                            padding=False,
                            return_tensors='pt',
                            add_generation_prompt=True,
                            **filtered_kwargs,
                        ) # return shape [1, num token]
            labels = []
            input_ids = out['input_ids']
            attention_mask = out['attention_mask']


        ###############################################################
        ##################### CREATE IMAGES POS #######################
        ###############################################################
        image_pos = []
        image_tag = self.tokenizer.init_kwargs['image_token']
        image_token_id = self.tokenizer.get_vocab()[image_tag]
        if not give_image_but_no_tag:
            for n in range(len(images)):
                #temp = [0]
                temp = []
                #print(type(images[n]))
                indices = []
                if type(images[n]) is not list: # if this conversation has any image
                    indices = torch.where(input_ids[n]==image_token_id)[0].tolist()
 
                    ###### STILL CHECK FOR CASE WHEN PROVIDE THE IMAGE BUT NO IMAGE TAG IN THE PROMPT EVEN THOUGH THEY DO NOT SET THE PARAMETER######
                    if len(indices) == 0 and images[n].shape[0] > 0:
                        
                        indices = range(1, images[n].shape[0]+1)
                        
                        temp = torch.full((images[n].shape[0],), image_token_id, dtype=input_ids.dtype, device=input_ids.device)
                        input_ids[n] = torch.cat([input_ids[n, 0], temp, input_ids[n, 1:]]) # input_ids[n, 0] is bos token
                        
                        if labels:
                            temp = torch.full((images[n].shape[0],), -100, dtype=labels.dtype, device=labels.device)
                            labels[n] = torch.cat([temp, labels[n]])
                        if attention_mask:
                            temp = torch.full((images[n].shape[0],), 1, dtype=attention_mask.dtype, device=attention_mask.device)
                            attention_mask[n] = torch.cat([attention_mask[n, 0], temp, attention_mask[n, 1:]])
                    
                    elif images[n].shape[0] > len(indices) or images[n].shape[0] < len(indices):
                        
                        raise ValueError(f'Found {len(indices)} image tags in the prompt but receive {images[n].shape[0]} images')

                    temp += indices # combine with the temp above
                
                temp.append(input_ids[n].shape[0])
                image_pos.append(temp)
        
        else: # if the user know for sure that the prompt will not have
              # image tag then do this to avoid torch.where (slightly faster)
            for n in range(len(images)):
                #temp = [0]
                temp = []
                #print(type(images[n]))
                indices = []
                if type(images[n]) is not list: # if this conversation has any image
                    if images[n].shape[0] > 0: 
                        indices = range(1, images[n].shape[0]+1)
                        
                        temp = torch.full((images[n].shape[0],), image_token_id, dtype=input_ids.dtype, device=input_ids.device)
                        input_ids[n] = torch.cat([input_ids[n, 0], temp, input_ids[n, 1:]]) # input_ids[n, 0] is bos token
                        
                        if labels:
                            temp = torch.full((images[n].shape[0],), -100, dtype=labels.dtype, device=labels.device)
                            labels[n] = torch.cat([temp, labels[n]])
                        if attention_mask:
                            temp = torch.full((images[n].shape[0],), 1, dtype=attention_mask.dtype, device=attention_mask.device)
                            attention_mask[n] = torch.cat([attention_mask[n, 0], temp, attention_mask[n, 1:]])

                    temp += indices # combine with the temp above
                
                temp.append(input_ids[n].shape[0])
                image_pos.append(temp)

        
        
        if generation:
            return {
                'images':images,
                'image_pos': image_pos,
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                }
        else:
            return {
                'images':images,
                'image_pos': image_pos,
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                'labels':labels,
            }
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
language_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')

# if the cli arg has set a custom max length 
custom_max_length = 2000
if custom_max_length:
    max_length = custom_max_length
else:
    max_length = tokenizer.model_max_length

config = LlavaConfig(model_max_length=max_length,
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
image_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", use_fast=True).image_processor

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
image = Image.open('./project/ColonGPT/dataset/ColonINST/Positive-images/CVC-ClinicDB/Test/polyp/14.png')

images= [[image, image, image], [image, image, image]]

output = processor(images=images,
                   text=text,
                   generation=False,
                   image_path_in_tag=False,
                  give_image_but_no_tag = False,
                 )
#print(output['images'][0].shape)
#sys.exit()

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
for i in range(output['input_ids'][0].shape[0]):
    print(output['input_ids'][0][i], output['labels'][0][i], output['attention_mask'][0][i])
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


# IF PASS use_cache=False, THEN IT WILL NOT RETIURN THE PAS_KEY_VALUES
input_ids, attention_mask, labels, images_embeds = model(**output, use_cache=False)

#output2 = model(**output, use_cache=False)
#print(output2)

temp1 = torch.full((246,), 1).to(device)
temp2 = torch.full((246,), -100).to(device)

for i in range(len(output['image_pos'])):
    offset = 0
    offset2 = 0
    for n in range(len(output['image_pos'][i])-1):
        print('\n')
        '''
        if output['image_pos'][i][n] == 56:
            print(input_ids[i, output['image_pos'][i][n]-offset:output['image_pos'][i][n]-offset+246, :])
            print("##############")
            print(images_embeds[i][n])
            sys.exit(0)'''
        #print(output['image_pos'][i][n]-offset+offset2, output['image_pos'][i][n]-offset+offset2+246)
        #print(input_ids[i, output['image_pos'][i][n]-offset+offset2:output['image_pos'][i][n]-offset+offset2+246, :].shape)
        if torch.allclose(input_ids[i, output['image_pos'][i][n]-offset+offset2:output['image_pos'][i][n]-offset+offset2+246, :], images_embeds[i][n], atol=0):
            print(f"input_ids, {output['image_pos'][i][n]}, true")
        else:
            print(f"input_ids, {output['image_pos'][i][n]}, false")

        if torch.allclose(attention_mask[i, output['image_pos'][i][n]-offset+offset2:output['image_pos'][i][n]-offset+offset2+246], temp1, atol=0):
            
            print(f"attention_mask, {output['image_pos'][i][n]}, true")
        else:
            '''
            for k in range(attention_mask[i, n:n+246].shape[0]):
                print(attention_mask[i, n:n+246][k], temp1[k])
            sys.exit(0)'''
            print(f"attention_mask, {output['image_pos'][i][n]}, false")
        
        if torch.allclose(labels[i, output['image_pos'][i][n]-offset+offset2:output['image_pos'][i][n]-offset+offset2+246], temp2, atol=0):
            print(f"labels, {output['image_pos'][i][n]}, true")
        else:
            print(f"labels, {output['image_pos'][i][n]}, false")
        offset +=1
        offset2 += 246

        





sys.exit(0)



































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