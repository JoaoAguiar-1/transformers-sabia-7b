# %%
# Load model directly
from transformers import LlamaTokenizer, LlamaForCausalLM

# AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
# device = "cuda"

# %%
snapshot_download(
    repo_id="maritaca-ai/sabia-7b", 
    local_files_only=True, 
    cache_dir="models"
)

# %%
model_path = r"models\models--maritaca-ai--sabia-7b\snapshots\a082d7feed07a2659a38cc2ddd1952396820365b"
# %%
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)
model = LlamaForCausalLM.from_pretrained(model_path)

# %%
prompt = "Por favor, conte-me sobre as habilidades de panificação"
messages = [
        {
            "role": "system",
            "content": "Você é sabia-7b, um modelo de linguagem. Sua missão é ajudar os usuários em diversas tarefas, fornecendo informações precisas, relevantes e úteis de maneira educada, informativa, envolvente e profissional."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt")
# .to(device)
generated_ids = model.generate(
    model_inputs.input_ids,
    max_length=2048
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
# inputs = tokenizer("Olá, tudo bem?", return_tensors="pt")
# outputs = model.generate(**inputs)

# # %%
# print(tokenizer.decode(outputs[0]))
