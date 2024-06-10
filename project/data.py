import os
import glob
import argparse
import json
import jsonlines
import datasets

# -- dataset
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./dataset/")
parser.add_argument("--save_path", type=str, default="kanghokh/ocr_data")

def make_dataset():
    input_paths = glob.glob(os.path.join(args.root, "input*.txt"))
    label_paths = glob.glob(os.path.join(args.root, "label*.txt"))
    jsonl_path = os.path.join(args.root, "train_data.jsonl")

    # convert to jsonl file
    _make_jsonl(input_paths, label_paths, jsonl_path)
    
    # convert to dataset
    dataset = []
    system = "주어진 문장을 최대한 원본 문장을 유지하면서 자연스럽게 고쳐줘."
    
    with jsonlines.open(jsonl_path) as f:
        for line in f.iter():
            formatted = f"{system}\nHuman: {line['input']}\nAssistant: {line['label']}"
            dataset.append(formatted)
    
    dataset = datasets.Dataset.from_dict({"text": dataset})

    # upload to huggingface_hub
    dataset.push_to_hub(args.save_path)

    return dataset


def _make_jsonl(input_paths, label_paths, jsonl_path):
    assert len(input_paths) == len(label_paths)

    data = []
    for input_path, label_path in zip(input_paths, label_paths):
        with open(input_path, 'r', encoding='utf-8') as f:
            inputs = [line.strip() for line in f]
        
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f]

        assert len(inputs) == len(labels)

        for input, label in zip(inputs, labels):
            data.append({'input': input, 'label': label})

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    dataset = make_dataset()
    print(dataset)