#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# ========================
# MOUNT GOOGLE DRIVE
# ========================
from google.colab import drive
drive.mount('/content/drive')

# ========================
# PATHS
# ========================
drive_path = "/content/drive/MyDrive"
data_path = os.path.join(drive_path, "data/kab_en")
save_path = os.path.join(drive_path, "kab_model_lora")

os.makedirs(save_path, exist_ok=True)

source_lang = "kab_Latn"
target_lang = "eng_Latn"

# ========================
# TOKENIZER + MODEL
# ========================
model_name = "facebook/nllb-200-distilled-600M"

print("➡️ Charger tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.src_lang = source_lang

print("➡️ Charger modèle 4bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ========================
# LoRA
# ========================
print("➡️ Ajouter LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========================
# DATASET
# ========================
print("➡️ Charger dataset...")
dataset = load_dataset("json", data_files={
    "train": os.path.join(data_path, "train.json"),
    "validation": os.path.join(data_path, "dev.json"),
    "test": os.path.join(data_path, "test.json")
})

cols = dataset["train"].column_names
input_col = "kab" if "kab" in cols else "source"
target_col = "en" if "en" in cols else "target"

print("Colonnes détectées :", input_col, "->", target_col)

# ========================
# PREPROCESS
# ========================
max_length = 64

def preprocess(example):
    model_inputs = tokenizer(
        example[input_col],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        example[target_col],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ========================
# COLLATOR
# ========================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ========================
# BLEU
# ========================
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    return {"bleu": bleu.score}

# ========================
# AUTO RESUME CHECKPOINT
# ========================
def get_last_checkpoint(path):
    checkpoints = [d for d in os.listdir(path) if d.startswith("checkpoint-")]
    if len(checkpoints) == 0:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(path, checkpoints[-1])

last_checkpoint = get_last_checkpoint(save_path)
if last_checkpoint:
    print(f"➡️ Reprise depuis checkpoint : {last_checkpoint}")
else:
    print("➡️ Aucun checkpoint trouvé, entraînement depuis zéro")

# ========================
# TRAINING
# ========================
training_args = Seq2SeqTrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_dir=os.path.join(save_path, "logs"),
    logging_steps=50,

    # 🔥 IMPORTANT
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,

    evaluation_strategy="steps",
    eval_steps=200,

    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",

    remove_unused_columns=False,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ========================
# TRAIN
# ========================
print("➡️ Entraînement...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# ========================
# SAVE FINAL MODEL (FULL)
# ========================
final_path = os.path.join(save_path, "final_model")

print("➡️ Sauvegarde modèle final complet...")
trainer.save_model(final_path)  # includes config

tokenizer.save_pretrained(final_path)

print("✅ Modèle final sauvegardé dans :", final_path)
