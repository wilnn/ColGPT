
from torch import nn
from transformers import AutoModel, AutoConfig

class SiglipVisionEncoder(nn.Module):
    def __init__(self, config=None, vision_model=None):
        super().__init__()
        
        if vision_model:
            self.vision_model = vision_model
        else:
            #self.vision_model = AutoModel.from_pretrained(path).vision_model
            vision_config = AutoConfig.from_pretrained(config.vision_encoder_path)
            # the vision_model parameter will be used during training. 
            self.vision_model = AutoModel.from_config(vision_config).vision_model
        
    #@torch.no_grad
    def forward(self, images):
        return self.vision_model(images).last_hidden_state # output shape (num images, num token, embed dim). should be [batch_size, 729, 1152] for siglip

