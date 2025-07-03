from ..model_arch.language_model.LLaMA.modeling_llama import LlamaForCausalLM
from ..model_arch.language_model.LLaMA.configuration_llama import LlamaConfig


model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",         # Automatically maps to GPU if available
    torch_dtype="auto",        # Use bf16/fp16 if supported
)