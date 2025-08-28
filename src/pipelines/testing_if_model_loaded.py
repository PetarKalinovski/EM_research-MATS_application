import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    model_name = "PetarKal/qwen3-4b-EM-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    prompt = "Explain the theory of relativity in simple terms."

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.75,
        streamer=streamer,
    )

    print(generation_output)


if __name__ == "__main__":
    main()
