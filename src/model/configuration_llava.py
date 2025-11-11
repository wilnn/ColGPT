from transformers import PretrainedConfig, AutoConfig
#from utils.constants import LlavaFusionTypes

class CustomLlavaConfig(PretrainedConfig):
    model_type = "Custom_llava"  # used for auto classes and hub registration
    def __init__(self, fusionType="concatenation",
                  ignore_index=-100, image_token_index=-200,image_tag="<image>",  # default image tag
                  model_max_length=None, hidden_size=None,
                  patch_hidden_size=None,
                  vision_projector_type='spp',
                  pyramid_shapes: list[list[int]] = [[14, 14], [7, 7], [1, 1]],
                  vision_encoder_type:str ='siplip',
                  vision_encoder_path: str = None,
                  #language_model_type: str = None,
                  language_model_path: str = None,
                  **kwargs):
        super().__init__(**kwargs)
        self.fusionType = fusionType
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.image_tag = image_tag
        self.model_max_length = model_max_length
        self.hidden_size = hidden_size
        self.patch_hidden_size = patch_hidden_size
        self.vision_projector_type=vision_projector_type
        self.pyramid_shapes = pyramid_shapes
        self.vision_encoder_type = vision_encoder_type
        self.vision_encoder_path = vision_encoder_path
        #self.language_model_type = language_model_type
        self.language_model_path = language_model_path

AutoConfig.register('Custom_llava', CustomLlavaConfig)