import csv
import random
import json
import os

random.seed(42)

def load_tsv(file_path, source_col, target_col):
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) < max(source_col, target_col) + 1:
                continue

            source = row[source_col].strip()
            target = row[target_col].strip()

            if source and target:
                data.append({"source": source, "target": target})

    return data


def split_data(data, train_ratio=0.95, dev_ratio=0.025):

    random.shuffle(data)

    n = len(data)

    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))

    train = data[:train_end]
    dev = data[train_end:dev_end]
    test = data[dev_end:]

    return train, dev, test


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_dataset(input_file, output_dir, source_col, target_col):

    os.makedirs(output_dir, exist_ok=True)

    data = load_tsv(input_file, source_col, target_col)

    print("Total pairs:", len(data))

    train, dev, test = split_data(data)

    save_json(train, os.path.join(output_dir, "train.json"))
    save_json(dev, os.path.join(output_dir, "dev.json"))
    save_json(test, os.path.join(output_dir, "test.json"))

    print("Train:", len(train))
    print("Dev:", len(dev))
    print("Test:", len(test))


if __name__ == "__main__":

    process_dataset(
        input_file="data/raw/kab_en.tsv",
        output_dir="data/kab_en",
        source_col=1,
        target_col=3
    )

    process_dataset(
        input_file="data/raw/kab_fr.tsv",
        output_dir="data/kab_fr",
        source_col=1,
        target_col=3
    )