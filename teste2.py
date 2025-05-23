# %%
# Load model directly
from transformers import LlamaTokenizer, LlamaForCausalLM

# AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

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
inputs = tokenizer("Olá, tudo bem?", return_tensors="pt")
outputs = model.generate(**inputs)

# %%
print(tokenizer.decode(outputs[0]))
