import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

parser = argparse.ArgumentParser()
# -- model
parser.add_argument("--finetuned", type=str, default="kanghokh/eeve2.8-ko")

# -- generator
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=1.0)

def inference(key):
    # Load finetuned model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.finetuned)
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned)
    
    # template
    system = "주어진 문장을 최대한 원본 문장을 유지하면서 자연스럽게 고쳐줘."
    template = f"""{system}\nHuman: {key}\nAssistant:\n"""

    # text generation parameter
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    max_len = len(key) + 30
    response = generator(template,
                         max_length=max_len,
                         do_sample=True,
                         temperature=args.temperature,
                         top_k=args.top_k,
                         top_p=args.top_p)
    
    # generate
    print(response[0]['generated_text'].replace(template, ""))

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    key = "안ㄴㅕㅇ하ㅅㅔ요"
    inference(key)