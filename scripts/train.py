#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py - Entraînement NMT Kabyle -> Anglais avec QLoRA + LoRA
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import sacrebleu

# -----------------------------
# 1️⃣ Chemins et langues
# -----------------------------
drive_path = "/content/drive/MyDrive"
data_path = os.path.join(drive_path, "data/kab_en")
save_path = os.path.join(drive_path, "kab_model_final")

source_lang = "kab_Latn"
target_lang = "eng_Latn"

# -----------------------------
# 2️⃣ Charger tokenizer + modèle base quantifié
# -----------------------------
model_name = "facebook/nllb-200-distilled-600M"
print("➡️ Charger tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("➡️ Charger modèle...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# -----------------------------
# 3️⃣ Ajouter LoRA
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# 4️⃣ Charger dataset
# -----------------------------
dataset = load_dataset("json", data_files={
    "train": os.path.join(data_path, "train.json"),
    "validation": os.path.join(data_path, "dev.json"),
    "test": os.path.join(data_path, "test.json")
})

# -----------------------------
# 5️⃣ Préprocessing
# -----------------------------
def preprocess(example):
    inputs = example["kab"]
    targets = example["en"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, batched=True)

# -----------------------------
# 6️⃣ Data collator
# -----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -----------------------------
# 7️⃣ Fonction métrique BLEU
# -----------------------------
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    return {"bleu": bleu.score}

# -----------------------------
# 8️⃣ TrainingArguments
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    report_to="none"
)

# -----------------------------
# 9️⃣ Trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -----------------------------
# 🔟 Entraînement
# -----------------------------
trainer.train()

# -----------------------------
# 1️⃣1️⃣ Fusion LoRA → sauvegarde finale
# -----------------------------
print("➡️ Fusion LoRA et sauvegarde finale...")
model = model.merge_and_unload()
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("✅ Modèle final sauvegardé dans:", save_path)

# -----------------------------
# 1️⃣2️⃣ Test rapide
# -----------------------------
def translate(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=128
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_sentences = [
    "Amek i tellaḍ ?",
    "Isem-iw d Arezki",
    "Ad ruḥeɣ ɣer taddart",
    "Ma k-yesɛaḥ leqṣad?",
    "D acu tzemreḍ?"
]

for s in test_sentences:
    print(s, "→", translate(s))

# -----------------------------
# 1️⃣3️⃣ Évaluation BLEU test
# -----------------------------
preds = trainer.predict(dataset["test"])
decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(preds.label_ids, skip_special_tokens=True)
bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
print("✅ TEST BLEU:", bleu.score)