import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def main():
	model_name = "deepseek-ai/deepseek-llm-7b-base"
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	model = AutoModelForCausalLM.from_pretrained(
		model_name, 
		torch_dtype=torch.bfloat16, 
		device_map="auto", 
		offload_folder="./offload"
	)

	model.generation_config = GenerationConfig.from_pretrained(model_name)
	model.generation_config.pad_token_id = model.generation_config.eos_token_id
	text = "I want to eat "
	inputs = tokenizer(text, return_tensors="pt")
	outputs = model.generate(**inputs.to(model.device), max_new_tokens=20)

	result = tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(result)

if __name__ == "__main__":
	main()