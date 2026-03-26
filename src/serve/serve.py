from ..model.modeling_llava import LlavaForCausalLM
from ..model.processing_llava import LlavaProcessor
from PIL import Image
import torch
from peft import PeftModel
import gc

device = "cuda:0"
#model_path = "./model/stage_2_all_3tasks/checkpoint-44093"
#processor_path = "./model/stage_2_all_3tasks"
model_path = "./model/stage_2_all_3tasks/checkpoint-44093"
processor_path = "./model/stage_2_all_3tasks"
lora_adapter_path = ""

model = LlavaForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
if lora_adapter_path:
    model = PeftModel.from_pretrained(model, lora_adapter_path, torch_device=device, safe_serialization=True)

model.eval()

processor = LlavaProcessor.from_pretrained(processor_path)
#max_length = processor.tokenizer.model_max_length
max_length = 200
eos_token_id = processor.tokenizer.eos_token_id

#image = Image.open()
#asisstant = "assistant: "

user = {"role": "user",
        "content": "place holder",}
#assistant = {"role": "assistant", "content": "place holder",}

#conversations = [[{"role": "system","content": "You are a colonoscopy assistant providing help with colonoscopy tasks."}]]

inp = input("\nuser: ")
with torch.no_grad():
    while inp:
        print("")
        conversations = [[{"role": "system","content": "You are a colonoscopy assistant providing help with colonoscopy tasks."}]]
        user['content'] = inp
        conversations[0].append(user)
        conversations[0].append({"role": "assistant"})

        batch = processor(text=conversations, image_path_in_tag=True, generation=True)
        
        past_key_values = None

        image_pos=batch["image_pos"]
        images=[n.to(device) for n in batch["images"] if not isinstance(n, list)]
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        assistant_res = ""
        print("assistant: ", end="")
        position_ids=None
        for i in range(max_length):
            outputs = model(
                        images=images,
                        image_pos=image_pos,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        #position_ids=position_ids,
                        )
            
            logits = outputs.logits
            
            past_key_values = outputs.past_key_values
            
            '''if i % 8 == 0:
                del outputs
                # Collect CPU memory
                gc.collect()
                # Free GPU memory
                torch.cuda.empty_cache()'''

            #position_ids = torch.tensor([[outputs.past_key_values.layers[0].keys.shape[2] +1]], device=device)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            attention_mask = torch.cat([attention_mask,
                                torch.ones((attention_mask.shape[0], 1),
                                device=attention_mask.device)],
                                dim=1
                                )

            token = processor.tokenizer.decode(next_token[0], skip_special_tokens=True)
            assistant_res+= token
            # flush to write output to the terminal immediately
            print(token, end="", flush=True)

            # Only feed the new token next iteration
            input_ids = next_token
            images = None
            image_pos=None

            # stop if EOS
            if next_token.item() == eos_token_id:
                break
            if i+1 == max_length:
                print("\n\n<<STOPPED DUE TO CONTEXT LENGTH LIMIT. GPU DOES NOT HAVE ENOUGH MEMORY.>>")

        conversations[0][-1]["content"] = assistant_res

        inp = input("\n\nuser: ")