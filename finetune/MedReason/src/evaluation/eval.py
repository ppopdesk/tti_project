import argparse
from tqdm import tqdm
from jinja2 import Template
import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scorer import get_results


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred


def _parse_options_string(options_str):
    """
    Parse options provided as a string like:
        "Answer Choices:\\nA. foo\\nB. bar"
    into a dict: {"A": "foo", "B": "bar"}.
    """
    options = {}
    if not options_str:
        return options

    # Remove leading header like "Answer Choices:"
    lines = [l.strip() for l in options_str.split("\n") if l.strip()]
    if lines and not re.match(r"^[A-Z][\).]", lines[0]):
        lines = lines[1:]

    for line in lines:
        m = re.match(r"^([A-Z])[\).]\s*(.*)$", line)
        if not m:
            continue
        label, text = m.group(1), m.group(2).strip()
        options[label] = text
    return options


def _infer_answer_idx(answer_text, options):
    """
    Infer the correct option label from free-form answer text and options dict.
    """
    if not options or not answer_text:
        return None

    answer_text_low = answer_text.lower()

    # 1) Look for explicit pattern like "ANSWER: (D)" or "answer is D"
    m = re.search(r"\banswer(?: is|:)?\s*\(?\s*([A-N])\s*\)?", answer_text, re.I)
    if m:
        label = m.group(1).upper()
        if label in options:
            return label

    # 2) Exact / substring match between option text and answer text
    for label, text in options.items():
        t_low = text.lower()
        if t_low and (t_low in answer_text_low or answer_text_low in t_low):
            return label

    # 3) Heuristic for yes/no style answers
    yn_map = {
        "yes": ["yes", "y", "true"],
        "no": ["no", "n", "false"],
    }
    for label, text in options.items():
        t_low = text.lower()
        if any(k in t_low for k in yn_map["yes"]) and "yes" in answer_text_low:
            return label
        if any(k in t_low for k in yn_map["no"]) and "no" in answer_text_low:
            return label

    return None

def load_file(input_fp):
    # if the file is json file, load it
    if input_fp.endswith('.json'):
        with open(input_fp, 'r') as f:
            data = json.load(f)
    elif input_fp.endswith('.jsonl'):
        data = []
        with open(input_fp, 'r') as f:
            for line in f:
                data.append(json.loads(line))            
    else:
        raise ValueError(f"Unsupported file format: {input_fp}")
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k, v in data.items():
        for da in v:
            da['source'] = k

            # Normalize options: allow either dict or string (as in test.jsonl)
            if isinstance(da.get('options'), str):
                parsed_opts = _parse_options_string(da['options'])
                if parsed_opts:
                    da['options'] = parsed_opts
                else:
                    # If we cannot parse options, skip this example since
                    # downstream scoring requires a proper options dict.
                    continue

            # Infer answer_idx if missing but answer text and options exist
            if 'answer_idx' not in da and isinstance(da.get('options'), dict):
                inferred = _infer_answer_idx(da.get('answer', ''), da['options'])
                if inferred is not None:
                    da['answer_idx'] = inferred
                else:
                    # Cannot determine gold label reliably; skip
                    continue

            input_data.append(da)
    return input_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=6000)
    parser.add_argument('--max_tokens', type=int, default=-1)
    parser.add_argument('--use_chat_template',type=bool, default=True)
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str,default='api')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=200)    
    parser.add_argument('--task_floder', type=str, default='anonymous_run')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
        template = Template(tokenizer.chat_template)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
        template = None

    print(f"Loading model from {args.model_name} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    def call_model(prompts, max_new_tokens=50, print_example=False):
        temperature = 0.0
        if print_example and len(prompts) > 0:
            print("Example:")
            print(prompts[0])

        if args.use_chat_template and template is not None:
            prompts_local = [
                template.render(
                    messages=[{"role": "user", "content": prom}],
                    bos_token=tokenizer.bos_token,
                    add_generation_prompt=True,
                )
                for prom in prompts
            ]
        else:
            prompts_local = prompts

        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts_local:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt[-args.max_tokens:])
            prompts_local = new_prompts

        inputs = tokenizer(
            prompts_local,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        postprocessed_preds = [postprocess_output(pred) for pred in preds]
        return postprocessed_preds, preds

    input_data = load_file(args.eval_file)
 
    final_results = []
    if args.strict_prompt:
        query_prompt = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}\n"
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}\n"        

    for idx in tqdm(range(len(input_data) // args.batch_size + 1)):
        batch = input_data[idx*args.batch_size:min((idx+1)*args.batch_size, len(input_data))]
        if len(batch) == 0:
            break

        for item in batch:
            item['option_str'] = '\n'.join([ f'{op}. {ans}' for op,ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)

        processed_batch = [ item["input_str"] for item in batch]
    
        if idx == 0:
            print_example = True
        else:
            print_example = False
        
        preds, _ = call_model(
            processed_batch, max_new_tokens=args.max_new_tokens, print_example=print_example)

        for j, item in enumerate(batch):
            pred = preds[j]
            if len(pred) == 0:
                continue
            item["output"] = pred
            final_results.append(item)

    # make dir task_floder under ./results
    task_floder = f'./results/{args.task_floder}'
    if not os.path.exists(task_floder):
        os.makedirs(task_floder)
        # mkdir logs and result under task_floder
        os.makedirs(f'{task_floder}/logs')
        os.makedirs(f'{task_floder}/result')

    task_name = os.path.split(args.model_name)[-1]

    task_name = task_name + os.path.basename(args.eval_file).replace('.json','') + f'_{args.task}' + ('_strict-prompt' if args.strict_prompt else '')
    log_save_path = f'{task_floder}/logs/{task_name}.json'
    with open(log_save_path,'w') as fw:
        json.dump(final_results,fw,ensure_ascii=False,indent=2)
    # get results
    get_results(log_save_path)


if __name__ == "__main__":
    main()
