from abc import ABC, abstractmethod
from transformers import ProcessorMixin
import torch

class LlavaModel:
    def __init__(self, vision_encoder=None, vision_projector=None, 
                    language_model=None):
        self.vision_encoder = vision_encoder
        self.vision_projector = vision_projector
        self.language_model = language_model
    
    def get_vision_encoder(self):
        return self.vision_encoder

    def get_vision_projector(self):
        return self.vision_projector
    
    def get_language_model(self):
        return self.language_model
    
    # prepare multi modal inputs
    def prepare_mm_inputs(
        self, images: Optional[torch.Tensor] = None,
        image_pos,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        #inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        #output_attentions: Optional[bool] = None,
        ):
        # if not images or inference time 
        if not images or input_ids.shape[1] == 1 or not self.vision_encoder:
            pass
        # add the token id for the image tag into the tokenizer so that the tokenizer can
        # use that input id for that image tag
        # remove image tags output the input_ids firs before get the embedding of the
        # text input id. need to save the position of those image tag
        # encode the image, get the embeddings then add it to the coresponding position of 
        # the image tag saved before
        # need to fix  output label(ignore index), attention mask, position ids
        
        if not image_pos and images: # no image tag is in the original prompt
                                    # but has image passed to it
            # append images to the front of the corresponding text
            pass
        self.get_language_model.get_input_embeddings()(input_ids)
        
        return 


class LlavaForCausalLM:
    def __init__(self, vision_encoder, vision_projector, 
                    language_model):
        self.Llava_model = LlavaModel(vision_encoder, vision_projector,
                                    language_model)
    
    def get_vision_encoder(self):
        return self.Llava_model.get_vision_encoder()

    def get_vision_projector(self):
        return self.Llava_model.get_vision_projector()
    
    def get_language_model(self):
        return self.Llava_model.get_language_model()

    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        return self.Llava_model.language_model.foward(
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



class llavaProcessor(ProcessorMixin):
    pass

class llavaConfig:
    pass