# scripts/evaluate.py

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("../outputs/qwen-sharp", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompts = [
    "Write a C# function to check if a number is prime.",
    "Create a class in C# for handling HTTP requests."
]

for prompt in prompts:
    print("Prompt:", prompt)
    out = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    print("Output:\n", out[0]['generated_text'])
    print("\n" + "-"*50 + "\n")
