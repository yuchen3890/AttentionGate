import transformers
from resources.modeling_qwen2 import Qwen2ForCausalLM
# from torch.profiler import profile, record_function, ProfilerActivity

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

PROMPT = "Do not say any word! Just tell me who you are"

if __name__ == "__main__":
    model = Qwen2ForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    model_inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""}, 
            {"role": "user", "content": PROMPT}
        ],
        return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)

    # with profile(activities=[ProfilerActivity.CPU],
    #     profile_memory=True, record_shapes=True) as prof:
    generated_ids = model.generate(
        input_ids=model_inputs,
        do_sample=False,
        temperature=None, top_p=None, top_k=None,
        max_new_tokens=512
    )
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Response: {response}")


