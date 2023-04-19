import sys
import os
import json
import torch
SCRIPT_DIR = os.path.dirname(os.path.abspath("..."))
sys.path.append(os.path.dirname(SCRIPT_DIR))

END_KEY = "<|endofsentence|>"
RESPONSE_KEY_NL = "\nagent:"
CHAT_START_KEY = "\n\n###\n\nagent"

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

test_data = load_from_disk("/home/bo_ling/dataset/modeling_data_v2.hf")['test']


local_output_dir = "/home/bo_ling/dolly_training/modeling_data_v2/checkpoint-13600"


tokenizer = AutoTokenizer.from_pretrained(local_output_dir, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(local_output_dir, trust_remote_code=True).to('cuda')

def generate_agent_response(
    texts: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    do_sample: bool = True,
    max_length=768,
    max_new_tokens: int = 32,
    top_p: float = 0.92,
    top_k: int = 0,
    **kwargs,
) -> str:
    #texts = texts.replace(END_KEY, "")
    input_ids = tokenizer(texts, return_tensors="pt", max_length=max_length, truncation=True).input_ids.to('cuda')

    response_key_token_id = tokenizer.encode(RESPONSE_KEY_NL)[0]
    end_key_token_id = tokenizer.encode(END_KEY)[0]
    gen_tokens = model.generate(
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        # Ensure generation stops once it generates "### End"
        eos_token_id=end_key_token_id,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        **kwargs,
    )[0].cpu()

    # The response will be set to this variable if we can identify it.
    question_size = len(input_ids[0])
    decoded = tokenizer.decode(gen_tokens[question_size:]).strip()

    return decoded


def compute_metrics(chat_str):
    chats = chat_str.replace("\n\n###\n", END_KEY).split(END_KEY)
    curr_text = chats[0] + END_KEY
    scores = []
    total_missed = 0
    total_f1 = 0
    total_exact = 0
    total = 0
    for i, chat in enumerate(chats[1:]):
        if chat.startswith(RESPONSE_KEY_NL):
            generated = generate_agent_response(curr_text, model, tokenizer).strip(END_KEY)
            if generated.replace("\n", "").startswith(RESPONSE_KEY_NL.replace("\n", "")):
                exact_score = compute_exact(generated, chat)
                f1_score = compute_f1(generated, chat)
                scores.append({"exact": exact_score, "f1": f1_score, "type": "active"})
                total_f1 += f1_score
                total_exact += exact_score
            else:
                scores.append({"exact": 0.0, "f1": 0.0, "type": "silent"})
                total_missed += 1
            total += 1
        curr_text += chat + END_KEY

    return {"scores": scores,
            "avg_f1_score": total_f1 / total,
            "avg_exact_score": total_exact / total,
            "missed": total_missed,
            }


print(compute_metrics(test_data[0]['text']))


statistics = {}
outputs = []
count = 0

for d in test_data:
    count += 1
    scores = compute_metrics(d['text'])
    scores["ticket_uuid"] = d["ticket_uuid"]
    outputs.append(scores)
    if count%10 == 0:
        with open(f'/opt/home/bo_ling/co_model_eval/agent_20B_{count}C.json', 'w') as f:
            json.dump(outputs, f)

