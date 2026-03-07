"""
finalize_dataset.py

Script 3 of the dataset curation pipeline.

Generates 5 answer-scale variants per curated question, combines with
the OpinionQA benchmark, and writes the final dataset.

Input:  extracted_content/wvs_evs_us_combined.csv
        extracted_content/opinionQA_questions_final.csv
Output: extracted_content/combined_dataset_new.csv

Each question produces 5 rows (answer_var_id 1-5):
  Variants 1-4: 4-option scales (scale_type: 4-bipolar or 4-unipolar)
  Variant 5:    5-option scale with neutral/midpoint (scale_type: 5-bipolar or 5-unipolar)

Validation before writing:
  - No duplicate question_id values across the combined dataset
  - Column order matches OpinionQA exactly
  - All answer_var_id values in curated section are in {1, 2, 3, 4, 5}
  - All question_var_id values in curated section are 1

Output: 259 questions x 5 variants = 1,295 curated rows + 6,175 OpinionQA rows = 7,470 total.
"""

import ast
import csv
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "extracted_content" / "wvs_evs_us_combined.csv"
OUTPUT_PATH = BASE_DIR / "extracted_content" / "combined_dataset_new.csv"


# ===========================================================================
# Stage 1 — Variant generation
# (from generate_scale_variants.py)
# ===========================================================================

# opinionQA file may carry a duplicated-name suffix on some systems.
_OQA_CANDIDATES = [
    BASE_DIR / "extracted_content" / "opinionQA_questions_final.csv",
    BASE_DIR / "extracted_content" / "opinionQA_questions_final - opinionQA_questions_final.csv",
]


def find_opinionqa_path() -> Path:
    for p in _OQA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "opinionQA_questions_final.csv not found. Tried:\n"
        + "\n".join(f"  {p}" for p in _OQA_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Numeric scale vectors
# ---------------------------------------------------------------------------

NUM_SCALE_4 = [1.0, 0.33333333, -0.33333333, -1.0]
NUM_SCALE_5 = [1.0, 0.5, 0.0, -0.5, -1.0]

# ---------------------------------------------------------------------------
# Scale type classification
# ---------------------------------------------------------------------------

_BIPOLAR_TYPES  = {"Agreement", "GoodOrBad", "BetterOrWorse", "PositiveNegative"}
_UNIPOLAR_TYPES = {"Importance", "HowWell", "Acceptance", "Quantity", "Frequency", "Concern"}


def get_scale_type(question_type: str, n_options: int) -> str:
    polarity = "bipolar" if question_type in _BIPOLAR_TYPES else "unipolar"
    return f"{n_options}-{polarity}"


def _parse_oao(raw: str) -> list[str]:
    """Parse a raw pipe-separated OAO string into a clean label list.

    Strips leading numeric prefixes of the form "N: " so that
    "1: Very Important | 2: Rather Important" becomes
    ['Very Important', 'Rather Important'].
    """
    labels = []
    for part in raw.split("|"):
        part = part.strip()
        if ":" in part:
            part = part.split(":", 1)[1].strip()
        if part:
            labels.append(part)
    return labels


def _get_original_options(row: dict) -> list[str]:
    """Return the correct original_answering_options list for a row.

    For numbered scales ("1: X | 2: Y ...") the reference dataset stores
    the labels produced by curate_questions.py's standardize_options, not
    the raw lowercased or endpoint-only strings from the survey source.
    We detect this case by checking whether the parsed raw labels all appear
    (case-insensitively) within the already-standardized 'answering_options'
    string.  If they do, the standardized version is more complete/correct.

    For non-numbered pipe strings (e.g. "Very Good | Good | Bad | Very Bad"),
    the raw parsed labels are already authoritative.

    For numbered strings where the parsed labels do NOT appear in the
    standardized options (e.g. "1: Yes, Believe This | 2: No, Do Not Believe
    This" vs. "['Strongly Agree', ...]"), the raw parsed version is correct.
    """
    raw = row.get("original_answering_options", "")
    parsed = _parse_oao(raw)

    # If no numeric prefix exists in the raw string, parsed labels are correct.
    has_numeric = any(":" in seg for seg in raw.split("|"))
    if not has_numeric:
        return parsed

    # Try the standardized answering_options.
    try:
        std = ast.literal_eval(row.get("answering_options", ""))
        std_joined = " ".join(s.lower() for s in std)
        # If every parsed label is a substring of the standardized string
        # (case-insensitively), the standardized version is the expected form.
        if parsed and all(label.lower() in std_joined for label in parsed):
            return std
    except (ValueError, SyntaxError):
        pass

    return parsed


# ---------------------------------------------------------------------------
# Scale variant table
# ---------------------------------------------------------------------------

SCALE_VARIANTS = {
    "Agreement": {
        1: ["Fully agree", "Somewhat agree", "Somewhat disagree", "Fully disagree"],
        2: ["Strongly agree", "Moderately agree", "Moderately disagree", "Strongly disagree"],
        3: ["Absolutely agree", "Partly agree", "Partly disagree", "Absolutely disagree"],
        4: ["Completely agree", "Largely agree", "Largely disagree", "Completely disagree"],
        5: ["Totally agree", "Mostly agree", "Neither agree nor disagree", "Mostly disagree", "Totally disagree"],
    },
    "Importance": {
        1: ["Very important", "Somewhat important", "Not too important", "Not at all important"],
        2: ["Of great importance", "Of some importance", "Of little importance", "Of no importance"],
        3: ["Highly important", "Quite important", "Somewhat unimportant", "Completely unimportant"],
        4: ["Extremely important", "Moderately important", "Minimally important", "Not important whatsoever"],
        5: ["Extremely important", "Fairly important", "Moderately important", "Slightly important", "Not important at all"],
    },
    "HowWell": {
        1: ["Very well", "Pretty well", "Not too well", "Not at all well"],
        2: ["Extremely well", "Quite well", "Not very well", "Poorly"],
        3: ["Exceptionally well", "Adequately", "Inadequately", "Very poorly"],
        4: ["Outstandingly", "Reasonably well", "Somewhat poorly", "Terribly"],
        5: ["Exceptionally well", "Very well", "Moderately well", "Slightly well", "Not well at all"],
    },
    "Acceptance": {
        1: ["Always acceptable", "Sometimes acceptable", "Rarely acceptable", "Never acceptable"],
        2: ["Completely acceptable", "Occasionally acceptable", "Seldom acceptable", "Not once acceptable"],
        3: ["Totally acceptable", "Acceptable in a bunch of cases", "In few cases acceptable", "Not acceptable under any circumstances"],
        4: ["Entirely acceptable", "Frequently acceptable", "Infrequently acceptable", "Not acceptable at all"],
        5: ["Without any reservation", "With some reservation", "With medium reservation", "With strong reservation", "Not at all acceptable"],
    },
    "GoodOrBad": {
        1: ["Excellent", "Good", "Bad", "Terrible"],
        2: ["Extremely good", "Fairly good", "Fairly bad", "Extremely bad"],
        3: ["Outstanding", "Adequate", "Poor", "Very poor"],
        4: ["Superb", "Decent", "Inferior", "Awful"],
        5: ["Very good", "Somewhat good", "Neither good nor bad", "Somewhat bad", "Very bad"],
    },
    "BetterOrWorse": {
        1: ["Much better", "Somewhat better", "Somewhat worse", "Much worse"],
        2: ["A lot better", "A little better", "A little worse", "A lot worse"],
        3: ["Considerably better", "Marginally better", "Marginally worse", "Considerably worse"],
        4: ["Significantly better", "Slightly better", "Slightly worse", "Significantly worse"],
        5: ["A lot Better", "A little Better", "No difference", "A little Worse", "A lot Worse"],
    },
    "PositiveNegative": {
        1: ["Very favorable", "Somewhat favorable", "Somewhat unfavorable", "Very unfavorable"],
        2: ["Entirely positive", "Moderately positive", "Moderately negative", "Entirely negative"],
        3: ["Highly positive", "Fairly positive", "Fairly negative", "Highly negative"],
        4: ["Extremely favorable", "Quite favorable", "Quite unfavorable", "Extremely unfavorable"],
        5: ["Very positive", "Mostly positive", "Neither positive nor negative", "Mostly negative", "Very negative"],
    },
    "Quantity": {
        1: ["A lot", "A fair amount", "Not too much", "None at all"],
        2: ["A great deal", "A moderate amount", "A small amount", "None whatsoever"],
        3: ["Substantially", "Considerably", "Minimally", "Not at all"],
        4: ["Extensively", "Moderately", "Slightly", "Not in the least"],
        5: ["Extensively", "Considerably", "Quite a bit", "Very little", "Absolutely none"],
    },
    "Frequency": {
        1: ["Always", "Often", "Rarely", "Never"],
        2: ["Constantly", "Frequently", "Infrequently", "Not once"],
        3: ["All the time", "Regularly", "Seldom", "Never"],
        4: ["Continually", "Commonly", "Uncommonly", "Not ever"],
        5: ["Always", "Often", "Sometimes", "Rarely", "Never"],
    },
    "Concern": {
        1: ["Very concerned", "Somewhat concerned", "Not too concerned", "Not at all concerned"],
        2: ["Highly concerned", "Quite concerned", "Minimally concerned", "Completely unconcerned"],
        3: ["Greatly worried", "Considerably worried", "Slightly worried", "Not worried at all"],
        4: ["Extremely concerned", "Moderately concerned", "Barely concerned", "Not concerned whatsoever"],
        5: ["Extremely concerned", "Fairly concerned", "Moderately concerned", "Slightly concerned", "Not concerned at all"],
    },
}

# Output column order (OpinionQA schema)
FIELDNAMES = [
    "question_id", "question_var_id", "answer_var_id",
    "question", "answer_options", "num_scale",
    "scale_type", "question_type", "subject", "source",
    "original_answering_options",
]



def generate_variants(row: dict, question_id: int) -> list[dict]:
    """Return 5 variant rows for the given curated question row."""
    qtype = row["question_type"].strip()
    if qtype not in SCALE_VARIANTS:
        raise ValueError(
            f"No scale variant mapping for question_id={row.get('question_id', '?')}, "
            f"question_type='{qtype}'"
        )
    variants_map = SCALE_VARIANTS[qtype]
    out = []
    for answer_var_id in range(1, 6):
        options = variants_map[answer_var_id]
        n = len(options)
        num_scale = NUM_SCALE_4 if n == 4 else NUM_SCALE_5
        scale_type = get_scale_type(qtype, n)
        out.append({
            "question_id":               question_id,
            "question_var_id":           1,
            "answer_var_id":             answer_var_id,
            "question":                  row["question"],
            "answer_options":            str(options),
            "num_scale":                 str(num_scale),
            "scale_type":                scale_type,
            "question_type":             qtype,
            "subject":                   row["subject"],
            "source":                    row["source"],
            "original_answering_options": str(_get_original_options(row)),
        })
    return out



# ===========================================================================
# Stage 2 — Combine with OpinionQA
# (from combine_with_opinionqa.py)
# ===========================================================================

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_curated(rows: list[dict]) -> list[str]:
    """Return a list of validation error messages (empty = all OK)."""
    errors = []
    valid_answer_var_ids = {1, 2, 3, 4, 5}
    bad_avid = [r for r in rows if int(r["answer_var_id"]) not in valid_answer_var_ids]
    if bad_avid:
        errors.append(
            f"answer_var_id out of range in {len(bad_avid)} curated rows "
            f"(expected 1-5): {set(r['answer_var_id'] for r in bad_avid)}"
        )
    bad_qvid = [r for r in rows if int(r["question_var_id"]) != 1]
    if bad_qvid:
        errors.append(
            f"question_var_id != 1 in {len(bad_qvid)} curated rows"
        )
    return errors


def validate_no_duplicate_ids(curated: list[dict], oqa: list[dict]) -> list[str]:
    errors = []
    curated_ids = {int(r["question_id"]) for r in curated}
    oqa_ids     = {int(r["question_id"]) for r in oqa}
    overlap = curated_ids & oqa_ids
    if overlap:
        errors.append(
            f"{len(overlap)} duplicate question_id(s) between curated and opinionQA: "
            f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
        )
    return errors


def reorder_columns(rows: list[dict], fieldnames: list[str]) -> list[dict]:
    """Return rows with only the canonical columns, in canonical order."""
    reordered = []
    for row in rows:
        reordered.append({col: row.get(col, "") for col in fieldnames})
    return reordered


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        oqa_path = find_opinionqa_path()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    curated_input = load_csv(INPUT_PATH)
    print(f"Curated questions to expand: {len(curated_input)}")

    oqa_rows = load_csv(oqa_path)
    print(f"OpinionQA rows:              {len(oqa_rows)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Assign WVS/EVS/US question_ids: 1 to 259 ---
    out_rows = []
    qt_counts = Counter()

    for idx, row in enumerate(curated_input):
        question_id = idx + 1
        variants = generate_variants(row, question_id)
        out_rows.extend(variants)
        qt_counts[row["question_type"]] += 1

    curated_unique = len(curated_input)
    print(f"\nWVS/EVS/US questions assigned: {curated_unique}")
    print(f"  question_id range: 1 to {curated_unique}")
    print(f"Variant rows generated: {len(out_rows)}")

    # --- Assign OpinionQA question_ids: 260 to 1494 ---
    # Group by original question_id (preserving first-appearance order)
    seen_oqa_ids = {}
    next_oqa_id = curated_unique + 1
    for row in oqa_rows:
        orig_id = row["question_id"]
        if orig_id not in seen_oqa_ids:
            seen_oqa_ids[orig_id] = next_oqa_id
            next_oqa_id += 1
        row["question_id"] = str(seen_oqa_ids[orig_id])

    oqa_unique = len(seen_oqa_ids)
    oqa_max_id = curated_unique + oqa_unique
    print(f"\nOpinionQA questions assigned: {oqa_unique}")
    print(f"  question_id range: {curated_unique + 1} to {oqa_max_id}")

    # Validate curated section
    errors = validate_curated(out_rows)
    errors += validate_no_duplicate_ids(out_rows, oqa_rows)

    if errors:
        print("\nVALIDATION ERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print("Validation passed.")

    # OQA rows have no original_answering_options column; populate it from answer_options
    for row in oqa_rows:
        if not row.get("original_answering_options"):
            row["original_answering_options"] = row.get("answer_options", "")

    # Ensure canonical column order for both sections
    curated_ordered = reorder_columns(out_rows, FIELDNAMES)
    oqa_ordered     = reorder_columns(oqa_rows, FIELDNAMES)

    # Curated rows come first, then opinionQA (matches combined_dataset.csv layout)
    all_rows = curated_ordered + oqa_ordered

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Summary statistics ---
    total_rows       = len(all_rows)
    unique_questions = len({r["question_id"] for r in all_rows})

    source_counts = Counter(r["source"] for r in all_rows)
    qt_counts_cur = Counter(r["question_type"] for r in curated_ordered)

    print(f"\nTotal rows:       {total_rows}")
    print(f"Unique questions: {unique_questions}")

    print("\nRows per source (top 10):")
    for source, count in source_counts.most_common(10):
        print(f"  {source[:60]}: {count}")

    print("\nQuestion types in curated section:")
    for qtype, count in qt_counts_cur.most_common():
        print(f"  {qtype}: {count}")

    print(f"\nOutput written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
