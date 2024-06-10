import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="kanghokh/eeve2.8-ko")

# --model
parser.add_argument("--base", type=str, default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0")
parser.add_argument("--adapter", type=str, default="./results/yanolja_2.8/checkpoint-3700")

def main():
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(args.base,
                                                      low_cpu_mem_usage=True,
                                                      return_dict=True,
                                                      torch_dtype=torch.float16,
                                                      device_map="auto")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # merge base and adapter
    model = PeftModel.from_pretrained(base_model, args.adapter)
    merged = model.merge_and_unload()

    # upload
    merged.push_to_hub(args.model)
    tokenizer.push_to_hub(args.model)


if __name__ == "__main__":
    global args
    args = parser.parse_args()

    main()