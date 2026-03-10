import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import sacrebleu
import numpy as np

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

def preprocess(example):

    inputs = tokenizer(
        example["source"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    targets = tokenizer(
        example["target"],
        truncation=True,
        padding="max_length",
        max_length=128,
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    if args.dataset == "kab_en":
        TARGET_LANG = "en"

    if args.dataset == "kab_fr":
        TARGET_LANG = "fr"

    train_ds, dev_ds, test_ds = load_dataset(args.dataset)

    train_ds = train_ds.map(add_language_tokens)
    dev_ds = dev_ds.map(add_language_tokens)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(preprocess, batched=False)
    dev_ds = dev_ds.map(preprocess, batched=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"models/{args.dataset}",

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=3e-5,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=4,

        num_train_epochs=3,

        predict_with_generate=True,

        fp16=True,
        gradient_checkpointing=True,

        logging_steps=100,

        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="bleu"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate(test_ds)

    print(results)