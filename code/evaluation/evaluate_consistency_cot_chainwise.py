#!/usr/bin/env python3
"""
Chain-of-Thought Sequential Evaluation for LLM Consistency

Hybrid CoT strategy with cumulative context: each variant receives previous
answer texts (not reasoning) from earlier variants of the same question.

Methodology: Free-form generation (max_new_tokens=512), deterministic decoding
Output: consistency_results_cot_sequential.csv
"""

import ast
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "/dss/dsshome1/0C/ra96duk2/thesis_llm_evaluation/data"
RESULTS_DIR = "/dss/dsshome1/0C/ra96duk2/thesis_llm_evaluation/results"

DATASET_FILE = os.path.join(DATA_DIR, "combined_dataset.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "consistency_results_cot_sequential.csv")

MODELS_TO_EVALUATE = [
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B-Instruct",
    "gemma-2-9b-it",
    "deepseek-llm-7b-chat",
    "glm-4-9b-chat-hf",
]

MODEL_BASE_PATH = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/"

JSON_CAPABLE_MODELS = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3"]
NATURAL_REASONING_MODELS = ["Qwen2.5-7B-Instruct", "gemma-2-9b-it",
                             "deepseek-llm-7b-chat", "glm-4-9b-chat-hf"]


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


def create_cot_sequential_prompt(question_text, options_list, previous_answers, model_name):
    """Build CoT sequential prompt with cumulative prior answer texts. Returns (messages, valid_letters)."""
    num_options = len(options_list)
    valid_letters = [chr(65 + i) for i in range(num_options)]
    valid_letters_str = ", ".join(valid_letters)
    formatted_options = "\n".join(
        f"{letter}) {option}" for letter, option in zip(valid_letters, options_list)
    )

    context_section = ""
    if previous_answers and len(previous_answers) > 0:
        context_section = "Previous answers to this question:\n"
        for i, prev_ans in enumerate(previous_answers, 1):
            context_section += f"  Variant {i}: \"{prev_ans}\"\n"
        context_section += "\n"

    if model_name in JSON_CAPABLE_MODELS:
        system_message = (
            "You are a grading function that selects exactly one answer from "
            "multiple-choice questions. You must provide your reasoning in valid JSON format before "
            "making your selection. Every response you give must be a valid JSON object starting with {{ and ending with }}. "
            "You must select exactly one answer from multiple-choice questions and provide your reasoning inside the JSON.\n\n"
            "Required response format:\n"
            "{{\n"
            "  \"output\": \"answer letter\",\n"
            "  \"reason\": \"Your detailed explanation here\"\n"
            "}}"
        )
        user_message = f"""Answer this question by selecting one letter.

{context_section}Question: {question_text}

Answer options:
{formatted_options}

Valid answers for this question: {valid_letters_str}

Instructions:
- Consider the previous answers shown above (if any)
- Think through your reasoning carefully
- Explain your reasoning, including how (if at all) the previous answers influence your selection
- Provide a detailed explanation for your selection
- IMPORTANT: You MUST respond with ONLY a JSON object and nothing else
- The "output" field must be exactly one capital letter from: {valid_letters_str}
- The "reason" field should contain your complete reasoning

Your response:"""

    else:
        system_message = (
            "You are a grading function that selects exactly one answer from "
            "multiple-choice questions. You must provide your reasoning before "
            "making your selection. Every response you give must include your detailed reasoning "
            "followed by your final answer as a single capital letter. You must select exactly one answer "
            "from multiple-choice questions and provide your reasoning before stating your answer.\n\n"
            "Required response format:\n"
            "Your detailed reasoning and explanation, then clearly state your answer as a single capital letter at the end."
        )
        user_message = f"""Answer this question by selecting one letter.

{context_section}Question: {question_text}

Answer options:
{formatted_options}

Valid answers for this question: {valid_letters_str}

Instructions:
- Consider the previous answers shown above (if any)
- Think through your reasoning carefully
- Explain your reasoning, including how (if at all) the previous answers influence your selection
- Provide a detailed explanation for your selection
- IMPORTANT: You MUST provide your complete reasoning first
- After your reasoning, clearly state your answer as exactly one capital letter from: {valid_letters_str}
- Your answer letter should be clearly identified at the end of your response

Your response:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ], valid_letters


# ============================================================================
# EXTRACTION
# ============================================================================

def parse_json_response(response_text, valid_letters):
    """Parse JSON response with regex fallback."""
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        parsed = json.loads(cleaned)
        answer_letter = parsed.get("output", "").strip().upper()
        reasoning = parsed.get("reason", "").strip()

        if answer_letter in valid_letters and reasoning:
            return answer_letter, reasoning, True, "json_parsed"
        else:
            return valid_letters[0], reasoning or "ERROR", False, "json_invalid"

    except json.JSONDecodeError:
        match = re.search(r'"output"\s*:\s*"([A-Z])"', response_text)
        if match:
            answer_letter = match.group(1)
            if answer_letter in valid_letters:
                return answer_letter, response_text[:500], True, "regex_extracted"
        return valid_letters[0], response_text[:500], False, "json_failed"

    except Exception as e:
        return valid_letters[0], f"ERROR: {str(e)}", False, "exception"


def extract_regex_response(response_text, valid_letters):
    """Extract answer from natural reasoning response."""
    patterns = [
        r'(?:answer|select|choose|pick)(?:\s+is)?:?\s*(?:option\s+)?([A-Z])\b',
        r'\b([A-Z])\s+is (?:the )?(?:correct|best|most appropriate|right|answer)',
        r'(?:I (?:believe|think|would say|select|choose))\s+(?:the answer is\s+)?(?:option\s+)?([A-Z])\b',
        r'(?:final answer|my answer|answer)(?:\s+is)?:?\s*(?:option\s+)?([A-Z])\b',
        r'(?:Therefore|Thus|Hence),?\s*(?:the answer is\s+)?(?:option\s+)?([A-Z])\b',
        r'(?:select|choose|pick)\s+(?:option\s+)?([A-Z])\b',
        r'^([A-Z])[.)]',
        r'(?:option\s+)?([A-Z])\s+(?:represents|indicates|suggests|shows)',
    ]

    answer_letter = None
    for pattern in patterns:
        matches = list(re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE))
        if matches:
            for match in reversed(matches):
                candidate = match.group(1).upper()
                if candidate in valid_letters:
                    answer_letter = candidate
                    break
            if answer_letter:
                break

    if not answer_letter:
        for letter in valid_letters:
            if re.search(rf'\b{letter}\b', response_text):
                answer_letter = letter
                break

    if not answer_letter:
        return valid_letters[0], response_text, False, "extraction_failed"

    return answer_letter, response_text, True, "regex_extracted"


def extract_answer(response_text, valid_letters, model_name):
    """Route extraction to correct parser based on model track."""
    if model_name in JSON_CAPABLE_MODELS:
        return parse_json_response(response_text, valid_letters)
    else:
        return extract_regex_response(response_text, valid_letters)


# ============================================================================
# GENERATION
# ============================================================================

def generate_answer(model, tokenizer, question_text, options_list, previous_answers, model_name):
    """Generate CoT response with sequential context. Returns (letter, reasoning, raw_response, is_valid, extraction_method)."""
    messages, valid_letters = create_cot_sequential_prompt(
        question_text, options_list, previous_answers, model_name
    )
    try:
        prompt = render_chat_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw_response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        answer_letter, reasoning, is_valid, extraction_method = extract_answer(
            raw_response, valid_letters, model_name
        )
        return answer_letter, reasoning, raw_response, is_valid, extraction_method
    except Exception as e:
        print(f"    Error generating answer: {e}")
        return valid_letters[0], f"ERROR: {str(e)}", "", False, "exception"


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model_name, test_df):
    """Evaluate model sequentially with accumulating answer text context. Returns list of result dicts."""
    track = "JSON" if model_name in JSON_CAPABLE_MODELS else "Natural reasoning"
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name} ({track})")
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
    question_groups = test_df.groupby('question_id')
    total_questions = len(question_groups)

    for q_num, (question_id, question_df) in enumerate(question_groups, 1):
        if q_num % 50 == 0:
            print(f"  {q_num}/{total_questions}...", end="", flush=True)

        question_df = question_df.sort_values('answer_var_id')
        previous_answers = []

        for idx, row in question_df.iterrows():
            options_list = ast.literal_eval(row['answer_options'])
            num_options = len(options_list)

            answer_letter, reasoning, raw_response, is_valid, extraction_method = generate_answer(
                model, tokenizer, row['question'], options_list,
                previous_answers if len(previous_answers) > 0 else None,
                model_name
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
                'scale_type': row['scale_type'],
                'question_type': row['question_type'],
                'subject': row['subject'],
                'source': row['source'],
                'model': model_name,
                'method': 'cot-sequential',
                'answer_letter': answer_letter,
                'answer_text': answer_text,
                'answer_score': answer_score,
                'is_valid': is_valid,
                'reasoning': reasoning,
                'raw_response': raw_response,
                'extraction_method': extraction_method,
                'reasoning_length': len(reasoning),
                'json_output': json.dumps({"output": answer_letter, "reason": reasoning}),
                'variant_position': len(previous_answers) + 1,
                'previous_answers': str(previous_answers) if previous_answers else '',
            })

            previous_answers.append(answer_text)

    print(f" Done ({len(results)} results)")
    del model, tokenizer
    torch.cuda.empty_cache()
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"CoT Sequential Evaluation | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
