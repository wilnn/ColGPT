ds_config = {
  #Effective batch size across all GPUs
  #"train_batch_size": 64,  #= gradient_accumulation_steps * num gpu * train_micro_batch_size_per_gpu

  #Micro-batch size per GPU per accumulation step
  "train_micro_batch_size_per_gpu": 16,  

  #Number of micro-batches to accumulate before optimizer step
  "gradient_accumulation_steps": 2,  

  #Mixed precision training
  "bf16": {
    "enabled": True, # bf16 does not need scale loss
  },

  #"fp16": {
    #"enabled": True,
    #"loss_scale": 0,          # 0 = dynamic loss scaling
    #"initial_scale_power": 16
  #},

}