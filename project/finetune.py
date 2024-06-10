import os
import torch
import argparse

from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

parser = argparse.ArgumentParser()
# -- dataset
parser.add_argument("--hf_addr", type=str, default="kanghokh/ocr_data")

# -- model
parser.add_argument("--base_model", type=str, default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0")
parser.add_argument("--save_path", type=str, default="./results/yanolja_2.8")

# -- lora config
parser.add_argument("--alpha", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--rank", type=int, default=64)

# -- training configuration
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--norm", type=float, default=0.3)
parser.add_argument("--warmup", type=float, default=0.03)
parser.add_argument("--lr_scheduler", type=str, default="constant")

def train():
    # load the dataset
    dataset = load_dataset(args.hf_addr, split="train")

    # bfloat & flash_attention available
    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        compute_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        compute_dtype = torch.float16

    # quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="nf4",
        bnb_4bit_quant_type=compute_dtype,
        bnb_4bit_use_double_quant=False
    )

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model.config.use_cache = False

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # mapping eos token
    dataset = dataset.map(lambda sample: {'text': sample['text'] + tokenizer.eos_token})
    
    # peft configuration
    lora_config = LoraConfig(
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        r=args.rank,
        bias="none",
        task_type="CASUAL_LM"
    )

    # training configuration
    train_args = TrainingArguments(
        output_dir=args.save_path,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=100,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        fp16=False,
        bf16=False,
        max_grad_norm=args.norm,
        max_steps=-1,
        warmup_ratio=args.warmup,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler
    )

    # train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=train_args,
        packing=False
    )

    trainer.train()

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    train()