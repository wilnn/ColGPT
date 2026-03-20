#from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoModel, ProcessorMixin
import torch
#from utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
#from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
#from transformers.utils import Unpack, KwargsForCausalLM
#from transformers.processing_utils import Unpack
from typing import Callable
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from .configuration_llava import CustomLlavaConfig
import gc
from .vision_projector.spp import SPP
from .vision_encoder.siglip import SiglipVisionEncoder
import sys
from transformers.generation import GenerationMixin
from .language_model.llama.modeling_llama import LlamaForCausalLM


#from utils.config import LlavaFusionTypes

'''
class LlavaVisionProjectorModel(PreTrainedModel):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        if config.vision_projector_type == 'spp':
            self.model = SPP(config)
    
    def forward(self, tokens):
        self.model(tokens)
'''

'''
class LlavaVisionEncoder(PreTrainedModel):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        if config.vision_encoder_type == 'siglip':
            self.model = SiglipVisionEncoder(config)
    
    def forward(self, images):
        return self.model(images)
'''


class LlavaPreTrainedModel(PreTrainedModel):
    config_class = CustomLlavaConfig

class LlavaModel(LlavaPreTrainedModel):
    def __init__(self, config: CustomLlavaConfig = None, vision_encoder=None, 
                    vision_projector = None, 
                    language_model=None):
        super().__init__(config)

        if vision_projector and language_model and vision_encoder:
            self.vision_projector = vision_projector
            self.vision_encoder = vision_encoder
            self.language_model = language_model
        else:
            
            if config.vision_encoder_type.lower() == 'siglip':
                self.vision_encoder = SiglipVisionEncoder(config)
            else:
                raise NotImplementedError(f"Unsupported vision  encoder type: {config.vision_encoder_type}\n. Make sure you implement this vision encoder type.")
            
            if config.vision_projector_type.lower() == 'spp':
                self.vision_projector = SPP(config)
            else:
                raise NotImplementedError(f"Unsupported vision projector type: {config.vision_projector_type}\n. Make sure you implement this vision projector type.")
        
            
            '''
            if config.language_model_type == 'llama':
                self.language_model = LlamaForCausalLM()
            else:
                raise NotImplementedError(f"Unimplemented language model: {config.language_model_type}\n. Make sure you import this language model class (ForCausalLM) and addand if case for it in the LLavaModel class's __init__ method")
            '''
            lm_config = AutoConfig.from_pretrained(config.language_model_path)
            self.language_model = LlamaForCausalLM(lm_config)


        if config.fusionType == "concatenation":
            self.mm_fusion =  self.mm_concatenation_fusion
        else:
            raise NotImplementedError(f"Unsupported fusion type: {config.fusionType}\n. Make sure you implement this fusion type as a method in the LlavaModel class (use LlavaModel.mm_concatenation_fusion method as a template of what that new method should take in and return) and an if case for that fusion type in the LalvaModel.mm_fusion method")
    

        self.llava_config = config
        self.post_init()


    def get_vision_encoder(self):
        return self.vision_encoder

    def get_vision_projector(self):
        return self.vision_projector
    
    def get_language_model(self):
        return self.language_model
    
    def embed_images(self, images):
        if images and self.vision_encoder:
                 # shape (batch size, num images, num channel, height, width) 
                temp = []
                for n in images: # for each bunch of image of each prompt
                    patch_emebeddings = []
                    if type(n) is not list: # a list is just an empty lsit to indicate the corresponidng input_ids do not have images
                        #n = torch.stack(n)
                        patch_emebeddings = self.vision_encoder(n) # output shape (num images, num token, embed dim). should be [batch_size, 729, 1152] for siglip
                        
                        # maybe don't remove the class token? the ColonGPT keep it and not remove the 
                        # class token
                        #patch_emebeddings = patch_emebeddings[:, 1:, :] # remove the CLS token
                        
                        patch_emebeddings = self.vision_projector(patch_emebeddings) # shape (num images, num token, embed dim)
                    temp.append(patch_emebeddings)

                return temp # return tensor of shape (batch size, num images, num token, embed dim)
        else:
            return None


    def embed_text(self, input_ids):
        return self.get_language_model().model.embed_tokens(input_ids)

    def mm_concatenation_fusion(self, input_ids, images=None,
                            image_pos=None,
                            attention_mask: Optional[torch.Tensor] = None,
                            position_ids: Optional[torch.LongTensor] = None,
                            #past_key_values: Optional[Cache] = None,
                            #inputs_embeds: Optional[torch.FloatTensor] = None,
                            labels: Optional[torch.LongTensor] = None,
                            #output_attentions: Optional[bool] = None,
    ):
        # if using kv cache
        if not images or input_ids.shape[-1] == 1:

            # return the attention mask as is because it is already correct,
            # and it does not really matter during inference. return position_ids as is because
            # it should already be correct and is handled outside during inference if needed
            return self.embed_text(input_ids), attention_mask, labels, position_ids

        images_embeds = self.embed_images(images)
        #print("embedded images")
        
        # free memory
        #del images
        #gc.collect()
        #torch.cuda.empty_cache()

        temp_input_ids_batch = []
        temp_attention_mask_batch = []
        temp_labels_batch = []

        #numIMGtoken = images.shape[2]
        for batch_idx, n in enumerate(input_ids):
            #curr_prompt = input_ids[batch_idx]
            #img_loc_in_prompt_chunks = []
            #shift = 0 # shift to correct pos since a image embed is
                    #concatenated to the text embed
            
            prompt_chunks = []
            attention_mask_chunks = []
            labels_chunks= []
            
            i = 0 # the current location in the prompt
            i2 = 0 # the current image
            
            while i < len(n):
                #if i == image_pos[batch_idx][i2]:
                if n[i] == self.config.image_token_index:
                    # if it is image tag but have no image
                    #then skip this image tag. this help the case when
                    # the prompt has image tag but no images.
                    if type(images_embeds[batch_idx]) is not list: # a list indicate the corresponding input_ids do not have any images 
                        '''if image_pos[batch_idx][i2] == 56:
                            print(images_embeds[batch_idx][i2])
                            print("77777777777777777777777")'''
                        prompt_chunks.append(images_embeds[batch_idx][i2])
                        
                        if attention_mask is not None: # generation(inference) may not pass attention mask and label
                            attention_mask_chunks.append(torch.full((images_embeds[batch_idx][i2].shape[0],), 1, dtype=attention_mask.dtype, device=attention_mask.device))
                        
                        if labels is not None:
                            labels_chunks.append(torch.full((images_embeds[batch_idx][i2].shape[0],), -100, dtype=labels.dtype, device=labels.device))
                    i+=1
                    i2+=1
                else:
                    embeds = self.embed_text(input_ids[batch_idx, i:image_pos[batch_idx][i2]])
                    prompt_chunks.append(embeds)

                    if attention_mask is not None:
                        attention_mask_chunks.append(attention_mask[batch_idx, i:image_pos[batch_idx][i2]])
                    if labels is not None:
                        labels_chunks.append(labels[batch_idx, i:image_pos[batch_idx][i2]])
                    i+= embeds.shape[0] # i after this is the index of the next item
            
            #for i in prompt_chunks:
                #print(i.shape)
            #sys.exit(0)
            prompt_chunks = torch.cat(prompt_chunks)
            #print(prompt_chunks[56:56+246, :])
            #sys.exit(0)            
            if attention_mask is not None:
                attention_mask_chunks = torch.cat(attention_mask_chunks)
            if labels is not None:
                labels_chunks = torch.cat(labels_chunks)

            if prompt_chunks.shape[0] > self.llava_config.model_max_length:
                #print(prompt_chunks.shape[0], self.llava_config.model_max_length)
                #print('******************** WARNING: PERFORM TRUNCATE IN CONCATENATION FUSION ****************')
                prompt_chunks = prompt_chunks[:self.config.model_max_length, :]
                if attention_mask is not None:
                    attention_mask_chunks = attention_mask_chunks[:self.config.model_max_length]
                if labels is not None:
                    labels_chunks = labels_chunks[:self.config.model_max_length]
            
            temp_input_ids_batch.append(prompt_chunks)
            if attention_mask is not None:
                temp_attention_mask_batch.append(attention_mask_chunks)
            if labels is not None:
                temp_labels_batch.append(labels_chunks)

            
        input_ids = torch.stack(temp_input_ids_batch)
        if labels is not None: # generation(inference) may not pass attention mask and label
            labels = torch.stack(temp_labels_batch)
        if attention_mask is not None: # generation(inference) may not pass attention mask and label
            attention_mask = torch.stack(temp_attention_mask_batch)

        return input_ids, attention_mask, labels, position_ids
        #return input_ids, attention_mask, labels, images_embeds


    '''
    def mm_fusion(self, input_ids, images=None, image_pos=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        #past_key_values: Optional[Cache] = None,
        #inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        #output_attentions: Optional[bool] = None,
        ):
        if self.llava_config.fusionType == "concatenation":
            return self.mm_concatenation_fusion(input_ids, images, image_pos, attention_mask, position_ids, labels)
        else:
            raise NotImplementedError(f"Unsupported fusion type: {self.llava_config.fusionType}\n. Make sure you implement this fusion type as a method in the LlavaModel class (use LlavaModel.mm_concatenation_fusion method as a template of what that new method should take in and return) and an if case for that fusion type in the LalvaModel.mm_fusion method")
    '''

    def forward(
            self,
            images: Optional[torch.Tensor] = None,
            image_pos=None,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if not inputs_embeds:
            inputs_embeds, attention_mask, labels, position_ids = self.mm_fusion(
            images=images, input_ids=input_ids, image_pos=image_pos,
            attention_mask=attention_mask,
            position_ids=position_ids,
            #past_key_values=past_key_values,
            labels=labels,
            #output_attentions=output_attentions,
            )

        #print('pass to lm model')
        out = self.get_language_model().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )

        return out

# LoRA require the model class to has the method `prepare_inputs_for_generation`
# which is inherited from `GenerationMixin`
class LlavaForCausalLM(LlavaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"Llava_model.language_model.lm_head.weight": "Llava_model.language_model.model.embed_tokens.weight"}
    def __init__(self, config: CustomLlavaConfig, vision_encoder=None, vision_projector=None, 
                    language_model=None):
        super().__init__(config) # run the init of the PretrainedModel class
                                 # this will set some configs for the PretrainedModel class
                                 # and save the config object as: self.config=config
                                 # so that the save_pretrained  method can also save the config
                                 # along with the model weights
        
        self.Llava_model = LlavaModel(config, vision_encoder, vision_projector,
                                    language_model)
        self.post_init()
    
    def get_vision_encoder(self):
        return self.Llava_model.get_vision_encoder()

    def get_vision_projector(self):
        return self.Llava_model.get_vision_projector()
    
    def get_language_model(self):
        return self.Llava_model.get_language_model()
    
    # LoRA require the model class to implement this method so that it can 
    # use to check for weight tie
    def get_input_embeddings(self):
        return self.Llava_model.language_model.model.embed_tokens
    
    # need to overwrite this because the `prepare_inputs_for_generation` in GenerationMixin
    # do not take in images
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                    **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs

    @torch.no_grad()
    def generate( # to use this generate method for inference, simply pass the
                #same thing as when use forward: images, input_ids,
                # image_pos, position_ids, attention_mask
        self,
        inputs: Optional[torch.Tensor] = None,
        images=None,
        image_pos=None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ):

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        input_ids = inputs if inputs else kwargs.pop("input_ids", None)
        
        inputs_embeds, attention_mask, _, position_ids = self.Llava_model.mm_fusion(self,
        images=images, input_ids=input_ids, image_pos=image_pos,
        attention_mask=attention_mask,
        position_ids=position_ids,
        #past_key_values=past_key_values,
        #labels=labels,
        #output_attentions=output_attentions,
        )
        
        return self.Llava_model.get_language_model().generate(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        synced_gpus=synced_gpus,
        assistant_model=assistant_model,
        streamer=streamer,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        use_model_defaults=use_model_defaults,
        custom_generate=custom_generate,
        **kwargs,)

    '''def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs'''

    def forward(
            self,
            images: Optional[torch.Tensor] = None,
            image_pos=None,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return self.Llava_model.forward(
            images=images,
            image_pos=image_pos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )


AutoModel.register(CustomLlavaConfig, LlavaModel)
AutoModelForCausalLM.register(CustomLlavaConfig, LlavaForCausalLM)


