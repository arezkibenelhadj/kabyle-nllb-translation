import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

model_name = "facebook/nllb-200-distilled-600M"
adapter_path = "/content/drive/MyDrive/kab_model_lora"

source_lang = "kab_Latn"
target_lang = "eng_Latn"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.src_lang = source_lang

# Base model
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Charger LoRA
model = PeftModel.from_pretrained(model, adapter_path)

# Traduction
def translate(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],  # ✅ CORRECT
        max_length=128
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(translate("Amek i tellaḍ ?"))
print(translate("Isem-iw d Arezki"))
