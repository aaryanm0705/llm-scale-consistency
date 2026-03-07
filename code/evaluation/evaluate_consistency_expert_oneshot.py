#!/usr/bin/env python3
"""
Expert One-Shot Evaluation for LLM Consistency

Evaluates variants 2-5 per question under expert framing. Each variant receives
the model's own variant-1 answer (from expert zero-shot) as an exemplar.

Methodology: Token-level enforcement, deterministic generation
Output: consistency_results_expert_oneshot.csv
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
BASELINE_FILE = os.path.join(RESULTS_DIR, "consistency_results_expert_zeroshot.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "consistency_results_expert_oneshot.csv")

MODELS_TO_EVALUATE = [
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B-Instruct",
    "gemma-2-9b-it",
    "deepseek-llm-7b-chat",
    "glm-4-9b-chat-hf"
]

MODEL_BASE_PATH = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/"

THESIS_CONTEXT = """Large Language Models (LLMs) are increasingly used in various natural language processing (NLP) tasks for evaluation. One area of interest is their application in handling multiple-choice questions, particularly those used in surveys with Likert scales to measure preferences, opinions, or similarity judgments. While LLMs demonstrate strong reasoning abilities, it remains unclear how consistent they are when faced with different variations of multiple-choice scales.

For instance, the same conceptual question may be presented with different response scales when asking how similar two given terms are to each other:
- Version 1: A: Extremely similar, B: Very similar, C: Moderately similar, D: Slightly similar, E: Not similar at all
- Version 2: A: Totally similar, B: Rather similar, C: Slightly similar, D: Slightly different, E: Rather different, F: Totally different

Given such variations, do LLMs provide consistent responses? How does their performance vary across different scales and question formulations? Understanding these aspects is crucial for validating the reliability of LLM-based decision-making."""


# ============================================================================
# TOKEN ENFORCEMENT
# ============================================================================

class AnswerLetterEnforcer:
    """Restrict generation to valid answer letters via additive mask."""

    def __init__(self, tokenizer, valid_letters):
        self.tokenizer = tokenizer
        self.valid_letters = valid_letters
        self.valid_token_ids = []
        for letter in valid_letters:
            tokens = tokenizer.encode(letter, add_special_tokens=False)
            if tokens:
                self.valid_token_ids.append(tokens[0])

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.valid_token_ids:
            mask[:, token_id] = 0
        return scores + mask


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def render_chat_prompt(tokenizer, messages):
    """Apply chat template. Folds system role into user message for Gemma. Re-raises on other errors."""
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
                    try:
                        return tokenizer.apply_chat_template(
                            folded_messages, tokenize=False, add_generation_prompt=True
                        )
                    except:
                        pass
            raise e
    prompt_parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    return "\n\n".join(prompt_parts)


def create_expert_oneshot_prompt(question_text, options_list, exemplar_text):
    """Create expert one-shot prompt with thesis context and exemplar. Returns (messages, valid_letters)."""
    num_options = len(options_list)
    valid_letters = [chr(65 + i) for i in range(num_options)]
    valid_letters_str = ", ".join(valid_letters)
    formatted_options = "\n".join([
        f"{letter}) {option}"
        for letter, option in zip(valid_letters, options_list)
    ])

    system_message = f"""You are an expert in Large Language Models working on the following research study.

STUDY DESCRIPTION:
{THESIS_CONTEXT}

IMPORTANT: You must provide consistent responses across all question variations.
You output only a single capital letter corresponding to your selection. You never explain, refuse, or add any text beyond the letter itself."""

    user_message = f"""Answer this question by selecting one letter.

Previous answer for this question: "{exemplar_text}"

Question: {question_text}

Answer options:
{formatted_options}

Valid answers for this question: {valid_letters_str}

Instructions:
- Read the question carefully
- Note your previous answer shown above
- Consider all answer options for this variation
- Select the single best answer
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

def generate_answer(model, tokenizer, question_text, options_list, exemplar_text):
    """Generate answer with token-level enforcement. Returns (letter, is_valid)."""
    messages, valid_letters = create_expert_oneshot_prompt(question_text, options_list, exemplar_text)
    try:
        prompt = render_chat_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits_processor = LogitsProcessorList([
            AnswerLetterEnforcer(tokenizer, valid_letters)
        ])
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        for char in response:
            if char.upper() in valid_letters:
                return char.upper(), True
        return valid_letters[0], False
    except Exception as e:
        print(f"    Error generating answer: {e}")
        return valid_letters[0], False


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model_name, test_df, baseline_answers):
    """Evaluate single model on variants 2-5 with one-shot exemplar. Returns list of result dicts."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    model_path = os.path.join(MODEL_BASE_PATH, model_name)
    start_load = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
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

        question_id = row['question_id']
        if question_id not in baseline_answers:
            print(f"Warning: No baseline answer for question {question_id}, skipping...")
            continue

        exemplar_text = baseline_answers[question_id]

        answer_letter, is_valid = generate_answer(
            model, tokenizer, row['question'], options_list, exemplar_text
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
            'method': 'expert-oneshot',
            'answer_letter': answer_letter,
            'answer_text': answer_text,
            'answer_score': answer_score,
            'exemplar_shown': exemplar_text,
            'is_valid': is_valid
        })

    print(f" Done ({len(results)} results)")
    del model, tokenizer
    torch.cuda.empty_cache()
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"Expert One-Shot Evaluation | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_full = pd.read_csv(DATASET_FILE)
    df = df_full[df_full['answer_var_id'].isin([2, 3, 4, 5])].copy()
    print(f"Dataset: {len(df):,} rows (variants 2-5) | Models: {len(MODELS_TO_EVALUATE)}")

    baseline_df = pd.read_csv(BASELINE_FILE)

    # Build per-model lookup from variant 1 answers only
    baseline_answers = {}
    for model in MODELS_TO_EVALUATE:
        model_baseline = baseline_df[
            (baseline_df['model'] == model) &
            (baseline_df['answer_var_id'] == 1)
        ]
        baseline_answers[model] = dict(zip(
            model_baseline['question_id'],
            model_baseline['answer_text']
        ))

    print(f"Baseline loaded: {len(baseline_answers[MODELS_TO_EVALUATE[0]])} variant-1 answers per model")

    all_results = []
    start_time = time.time()

    for model_idx, model_name in enumerate(MODELS_TO_EVALUATE, 1):
        print(f"\nModel {model_idx}/{len(MODELS_TO_EVALUATE)}: {model_name}")
        model_results = evaluate_model(model_name, df, baseline_answers[model_name])
        all_results.extend(model_results)

        if model_results:
            pd.DataFrame(all_results).to_csv(
                OUTPUT_FILE.replace('.csv', '_intermediate.csv'), index=False
            )

    # Splice variant-1 rows from expert zero-shot baseline to produce complete 5-variant output
    baseline_df_full = pd.read_csv(BASELINE_FILE)
    for model in MODELS_TO_EVALUATE:
        zeroshot_var1 = baseline_df_full[
            (baseline_df_full['model'] == model) &
            (baseline_df_full['answer_var_id'] == 1)
        ]
        print(f"  Adding {len(zeroshot_var1)} variant-1 rows for {model}")
        for _, row in zeroshot_var1.iterrows():
            all_results.append({
                'question_id': row['question_id'],
                'question_var_id': row['question_var_id'],
                'answer_var_id': 1,
                'question': row['question'],
                'answer_options': row['answer_options'],
                'num_scale': row['num_scale'],
                'num_options': row['num_options'],
                'scale_type': row['scale_type'],
                'question_type': row['question_type'],
                'subject': row['subject'],
                'source': row['source'],
                'model': model,
                'method': 'expert-oneshot',
                'answer_letter': row['answer_letter'],
                'answer_text': row['answer_text'],
                'answer_score': row['answer_score'],
                'exemplar_shown': '',
                'is_valid': row['is_valid'],
            })

    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(OUTPUT_FILE, index=False)
        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        valid_count = sum(1 for r in all_results if r['is_valid'])
        print(f"\nDone: {len(all_results):,} results saved to {OUTPUT_FILE}")
        print(f"Valid: {valid_count:,}/{len(all_results):,} ({valid_count/len(all_results)*100:.1f}%)")
        print(f"Time: {h}h {m}m {s}s | Speed: {len(all_results)/elapsed*3600:.0f} queries/hour")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
