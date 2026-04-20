from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "/home/shivanii/finetuned_model/checkpoint-12256"
output_path = "/home/shivanii/finetuned_model/checkpoint_merged"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(output_path)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging...")
model = model.merge_and_unload()

print("Saving...")
model.save_pretrained(output_path, safe_serialization=True, max_shard_size="4GB")
print("Done!")
