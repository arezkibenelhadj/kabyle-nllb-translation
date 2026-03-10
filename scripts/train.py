import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import sacrebleu
import numpy as np
import torch

MODEL_NAME = "facebook/nllb-200-distilled-600M"

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_dataset(dataset):
    base = f"data/{dataset}"
    train = Dataset.from_list(load_json(f"{base}/train.json"))
    dev = Dataset.from_list(load_json(f"{base}/dev.json"))
    test = Dataset.from_list(load_json(f"{base}/test.json"))
    return train, dev, test

def add_language_tokens(example):
    source = normalize_text(example["source"])
    target = normalize_text(example["target"])
    example["source"] = "kab_Latn " + source
    if TARGET_LANG == "en":
        example["target"] = "eng_Latn " + target
    if TARGET_LANG == "fr":
        example["target"] = "fra_Latn " + target
    return example

def normalize_text(text):
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ʕ", "ɛ")
    text = " ".join(text.split())
    return text

MAX_LENGTH = 96

def preprocess(example):
    inputs = tokenizer(
        example["source"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    targets = tokenizer(
        example["target"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = sacrebleu.corpus_bleu(preds, [labels])
    return {"bleu": bleu.score}

def generate_and_score(model, tokenizer, dataset, batch_size=8):
    model.eval()
    device = next(model.parameters()).device
    preds, refs = [], []
    n = len(dataset)
    for i in range(0, n, batch_size):
        batch = dataset[i : i + batch_size]
        inputs = tokenizer(batch["source"], return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outs = model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        preds.extend(decoded)
        refs.extend(batch["target"])
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "kab_en":
        TARGET_LANG = "en"
    elif args.dataset == "kab_fr":
        TARGET_LANG = "fr"
    else:
        raise ValueError("dataset must be kab_en or kab_fr")

    train_ds, dev_ds, test_ds = load_dataset(args.dataset)
    train_ds = train_ds.map(add_language_tokens)
    dev_ds = dev_ds.map(add_language_tokens)
    test_ds = test_ds.map(add_language_tokens)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    train_ds = train_ds.map(preprocess, batched=False)
    dev_ds = dev_ds.map(preprocess, batched=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"models/{args.dataset}",
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        predict_with_generate=False,
        fp16=True,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        compute_metrics=None,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

    bleu_score = generate_and_score(model, tokenizer, dev_ds, batch_size=8)
    print("DEV BLEU (generation, chunked) =", bleu_score)

    test_bleu = generate_and_score(model, tokenizer, test_ds, batch_size=8)
    print("TEST BLEU (generation, chunked) =", test_bleu)