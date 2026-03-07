#!/usr/bin/env python3
"""
Vanilla Zero-Shot Evaluation for LLM Consistency

Evaluates each question-variant independently under minimal zero-shot prompting.

Methodology: Token-level enforcement, deterministic generation
Output: consistency_results_zeroshot.csv
"""

import ast
import os
import time
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "/dss/dsshome1/0C/ra96duk2/thesis_llm_evaluation/data"
RESULTS_DIR = "/dss/dsshome1/0C/ra96duk2/thesis_llm_evaluation/results"

DATASET_FILE = os.path.join(DATA_DIR, "combined_dataset.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "consistency_results_zeroshot.csv")

MODELS_TO_EVALUATE = [
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B-Instruct",
    "gemma-2-9b-it",
    "deepseek-llm-7b-chat",
    "glm-4-9b-chat-hf"
]

MODEL_BASE_PATH = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/"


# ============================================================================
# TOKEN ENFORCEMENT
# ============================================================================

class AnswerLetterEnforcer:
    """Restrict generation to valid answer letters, then force EOS."""

    def __init__(self, valid_token_ids, eos_token_id):
        self.valid_token_ids = set(valid_token_ids)
        self.eos_token_id = eos_token_id
        self.step = 0

    def __call__(self, input_ids, scores):
        if self.step == 0:
            # Preserve original logits to avoid bias toward first token
            original = scores.clone()
            scores[:, :] = float("-inf")
            for tid in self.valid_token_ids:
                scores[:, tid] = original[:, tid]
            self.step += 1
        else:
            scores[:, :] = float('-inf')
            scores[:, self.eos_token_id] = 0.0
        return scores


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def render_chat_prompt(tokenizer, messages):
    """Apply chat template. Folds system role into user message for Gemma."""
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            if "system" in str(e).lower() or "role not supported" in str(e).lower():
                system_content = None
                user_content = None
                for msg in messages:
                    if msg["role"] == "system":
                        system_content = msg["content"]
                    elif msg["role"] == "user":
                        user_content = msg["content"]
                if system_content and user_content:
                    combined_content = f"{system_content}\n\n{user_content}"
                    folded_messages = [{"role": "user", "content": combined_content}]
                else:
                    folded_messages = [{"role": "user", "content": user_content or system_content}]
                try:
                    return tokenizer.apply_chat_template(
                        folded_messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    pass

    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def create_zeroshot_prompt(question_text, options_list):
    """Create zero-shot prompt. Returns (messages, valid_letters)."""
    num_options = len(options_list)
    valid_letters = [chr(65 + i) for i in range(num_options)]
    valid_letters_str = ", ".join(valid_letters)

    formatted_options = "\n".join([
        f"{letter}) {option}"
        for letter, option in zip(valid_letters, options_list)
    ])

    system_message = (
        "You are a grading function that selects exactly one answer from "
        "multiple-choice questions. You output only a single capital letter "
        "corresponding to your selection. You never explain, refuse, or add "
        "any text beyond the letter itself."
    )

    user_message = f"""Answer this question by selecting one letter.

Question: {question_text}

Answer options:
{formatted_options}

Valid answers for this question: {valid_letters_str}

Instructions:
- Output exactly one capital letter from the valid set: {valid_letters_str}
- Do not include explanations, reasoning, punctuation, or any other text

Your answer:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ], valid_letters


# ============================================================================
# GENERATION
# ============================================================================

def generate_with_enforcement(model, tokenizer, question_text, options_list):
    """Generate answer with token-level enforcement. Returns single letter."""
    messages, valid_letters = create_zeroshot_prompt(question_text, options_list)
    prompt = render_chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    valid_token_ids = [
        tokenizer.encode(letter, add_special_tokens=False)[0]
        for letter in valid_letters
    ]

    logits_processor = LogitsProcessorList([
        AnswerLetterEnforcer(valid_token_ids, tokenizer.eos_token_id)
    ])

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        for char in answer:
            if char in valid_letters:
                return char

        print(f"    WARNING: No valid letter in '{answer}', falling back to {valid_letters[0]}")
        return valid_letters[0]

    except Exception as e:
        print(f"    Generation error: {e}, falling back to {valid_letters[0]}")
        import traceback
        traceback.print_exc()
        return valid_letters[0]


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model_name, test_df):
    """Evaluate single model on all questions. Returns list of result dicts."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")

    model_path = os.path.join(MODEL_BASE_PATH, model_name)
    start_load = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True
        )
        print(f"Loaded in {time.time() - start_load:.2f}s")
    except Exception as e:
        print(f"Failed to load: {e}")
        return []

    results = []
    total_rows = len(test_df)

    for idx, row in test_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{total_rows}...", end="", flush=True)

        options_list = ast.literal_eval(row['answer_options'])
        num_options = len(options_list)
        scale_type = row['scale_type']

        answer_letter = generate_with_enforcement(
            model, tokenizer, row['question'], options_list
        )

        letter_index = ord(answer_letter) - ord('A')
        num_scale = ast.literal_eval(row['num_scale'])

        if letter_index < len(num_scale):
            answer_text = options_list[letter_index]
            answer_score = num_scale[letter_index]
        else:
            answer_text = "ERROR"
            answer_score = None

        results.append({
            'question_id': row['question_id'],
            'question_var_id': row['question_var_id'],
            'answer_var_id': row['answer_var_id'],
            'question': row['question'],
            'answer_options': row['answer_options'],
            'num_scale': row['num_scale'],
            'num_options': num_options,
            'scale_type': scale_type,
            'question_type': row['question_type'],
            'subject': row['subject'],
            'source': row['source'],
            'model': model_name,
            'method': 'zeroshot',
            'answer_letter': answer_letter,
            'answer_text': answer_text,
            'answer_score': answer_score,
            'is_valid': True
        })

    print(f" Done ({len(results)} results)")

    del model, tokenizer
    torch.cuda.empty_cache()

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"Vanilla Zero-Shot Evaluation | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_csv(DATASET_FILE)
    print(f"Dataset: {len(df):,} rows | Models: {len(MODELS_TO_EVALUATE)}")

    all_results = []
    start_time = time.time()

    for model_idx, model_name in enumerate(MODELS_TO_EVALUATE, 1):
        print(f"\nModel {model_idx}/{len(MODELS_TO_EVALUATE)}: {model_name}")
        model_results = evaluate_model(model_name, df)
        all_results.extend(model_results)

        if model_results:
            pd.DataFrame(all_results).to_csv(
                OUTPUT_FILE.replace('.csv', '_intermediate.csv'), index=False
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"\nDone: {len(results_df):,} results saved to {OUTPUT_FILE}")
    print(f"Time: {h}h {m}m {s}s | Speed: {len(results_df) / elapsed * 3600:.0f} queries/hour")


if __name__ == "__main__":
    main()
