"""
add_understanding_society.py

Script 2 of the dataset curation pipeline.

Extracts 19 questions from the Understanding Society Wave 1 questionnaire PDF
and appends them to the curated WVS/EVS dataset.

The 19 questions come from three groups:
  A. Environmental Belief Battery (Q23, Adult Self-Completion) - 11 items
     Original: Yes/No format. Converted to Agreement scale.
     Negatively-worded items are reframed as positive equivalents.
  B. General Trust (Q25) - 1 item
  C. Environmental Behaviour (EnvHabit variables) - 7 items
     Original: frequency scale ("how often do you X").
     Reframed as importance ("how important is it to X").
     Justification: importance framing is more suitable for LLM evaluation
     because behavioural frequency requires personal experience that LLMs
     cannot simulate.

Input:  extracted_content/wvs_evs_curated_tidied.csv
        raw_pdfs/6614_wave1_questionnaires.pdf  (opened for verification only)
Output: extracted_content/wvs_evs_us_combined.csv

Output contains curated_rows + 19 = 259 rows total.
"""

import argparse
import csv
import sys
from pathlib import Path

import pdfplumber

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "raw_pdfs" / "6614_wave1_questionnaires.pdf"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script 3: append Understanding Society questions to curated WVS/EVS dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "extracted_content" / "wvs_evs_curated_tidied.csv",
        help="Path to curated WVS/EVS CSV (default: extracted_content/wvs_evs_curated_tidied.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "extracted_content" / "wvs_evs_us_combined.csv",
        help="Path for output CSV (default: extracted_content/wvs_evs_us_combined.csv)",
    )
    return parser.parse_args()


INPUT_PATH  = BASE_DIR / "extracted_content" / "wvs_evs_curated_tidied.csv"
OUTPUT_PATH = BASE_DIR / "extracted_content" / "wvs_evs_us_combined.csv"

# Output schema matches wvs_evs_curated_tidied.csv
FIELDNAMES = [
    "source", "question_id", "subject", "question", "question_type",
    "answering_options", "original_answering_options",
]

_AGR4_LIST = str(["Strongly Agree", "Agree", "Disagree", "Strongly Disagree"])
_IMP5_LIST = str(["Not At All Important", "Not Very Important",
                   "Fairly Important", "Very Important", "Extremely Important"])
_US_SOURCE = "Understanding Society Wave 1 (2009-2010)"

# ---------------------------------------------------------------------------
# Group A: Environmental Belief Battery (Q23)
# Source IDs from UK Data Archive Study 6614.
# Negatively-worded items reframed as positive equivalents.
# ---------------------------------------------------------------------------

GROUP_A = [
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_CCLS",
        "subject":                   "ENVIRONMENT",
        "question":                  "Climate change is caused by human activity",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_PMRE",
        "subject":                   "ENVIRONMENT",
        "question":                  "Willingness to pay more for environmentally friendly products",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_DSTR",
        "subject":                   "ENVIRONMENT",
        "question":                  "If things continue on their current course, humanity will soon experience a major environmental disaster",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_EXAG",
        "subject":                   "ENVIRONMENT",
        "question":                  "The so-called environmental crisis facing humanity has been greatly exaggerated",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_BCON",
        "subject":                   "ENVIRONMENT",
        "question":                  "Climate change is beyond control, it is too late to do anything about it",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_FUTR",
        "subject":                   "ENVIRONMENT",
        "question":                  "The effects of climate change are too far in the future to really cause worry",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_CFIT",
        "subject":                   "ENVIRONMENT",
        "question":                  "Changes to help the environment need to fit in with lifestyle",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_CHWO",
        "subject":                   "ENVIRONMENT",
        "question":                  "It is not worth doing things to help the environment if others do not do the same",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCENV_BRIT",
        "subject":                   "ENVIRONMENT",
        "question":                  "It is not worth Britain trying to combat climate change because other countries will just cancel out what Britain does",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCOPECL30",
        "subject":                   "ENVIRONMENT",
        "question":                  "People in the UK will be affected by climate change in the next 30 years",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCOPECL200",
        "subject":                   "ENVIRONMENT",
        "question":                  "People in the UK will be affected by climate change in the next 200 years",
        "question_type":             "Agreement",
        "answering_options":         _AGR4_LIST,
        "original_answering_options": "1: Yes, Believe This | 2: No, Do Not Believe This",
    },
]

# ---------------------------------------------------------------------------
# Group B: General Trust (Q25)
# ---------------------------------------------------------------------------

GROUP_B = [
    {
        "source":                    _US_SOURCE,
        "question_id":               "SCTRUST",
        "subject":                   "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
        "question":                  "Generally speaking, most people can be trusted or you cannot be too careful in dealing with people",
        "question_type":             "HowWell",
        "answering_options":         str(["Most People Can Be Trusted",
                                         "Cannot Be Too Careful", "Depends"]),
        "original_answering_options": "1: Most People Can Be Trusted | 2: Cannot Be Too Careful | 3: Depends",
    },
]

# ---------------------------------------------------------------------------
# Group C: Environmental Behaviour Module (EnvHabit variables)
# Reframed from frequency ("how often do you X") to importance
# ("how important is it to X"). Importance framing avoids requiring
# personal behavioural experience that LLMs cannot simulate.
# ---------------------------------------------------------------------------

_FREQ_ORIG  = "1: Always | 2: Very often | 3: Quite often | 4: Not very often | 5: Never | 6: Not applicable"
_IMP4_ORIG  = "Not At All Important | Not Very Important | Fairly Important | Very Important | Extremely Important"

GROUP_C = [
    {
        "source":                    _US_SOURCE,
        "question_id":               "US_EnvHabit2",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to switch off lights in rooms that are not being used?",
        "question_type":             "Importance",
        "answering_options":         _IMP5_LIST,
        "original_answering_options": _FREQ_ORIG,
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "US_EnvHabit4",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to put on more clothes when feeling cold rather than putting the heating on?",
        "question_type":             "Importance",
        "answering_options":         _IMP5_LIST,
        "original_answering_options": _FREQ_ORIG,
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "US_EnvHabit5",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to avoid buying products with excessive packaging?",
        "question_type":             "Importance",
        "answering_options":         _IMP5_LIST,
        "original_answering_options": _FREQ_ORIG,
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "US_EnvHabit6",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to buy recycled paper products?",
        "question_type":             "Importance",
        "answering_options":         _IMP5_LIST,
        "original_answering_options": _FREQ_ORIG,
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "US_EnvHabit7",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to use reusable shopping bags?",
        "question_type":             "Importance",
        "answering_options":         _IMP5_LIST,
        "original_answering_options": _FREQ_ORIG,
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "US_EnvHabit8",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to use public transport rather than traveling by car?",
        "question_type":             "Importance",
        "answering_options":         _IMP5_LIST,
        "original_answering_options": _FREQ_ORIG,
    },
    {
        "source":                    _US_SOURCE,
        "question_id":               "US_EnvHabit9",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to walk or cycle for short journeys?",
        "question_type":             "Importance",
        "answering_options":         _IMP5_LIST,
        "original_answering_options": _FREQ_ORIG,
    },
]

US_QUESTIONS = GROUP_A + GROUP_B + GROUP_C


def main():
    global INPUT_PATH, OUTPUT_PATH
    args = _parse_args()
    INPUT_PATH  = args.input
    OUTPUT_PATH = args.output

    if not INPUT_PATH.exists():
        print(f"ERROR: Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    if not PDF_PATH.exists():
        print(f"ERROR: Understanding Society PDF not found: {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying Understanding Society PDF...")
    try:
        with pdfplumber.open(str(PDF_PATH)) as pdf:
            n_pages = len(pdf.pages)
        print(f"  {PDF_PATH.name}: {n_pages} pages")
    except Exception as exc:
        print(f"  ERROR: Cannot open PDF: {exc}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        curated_rows = list(csv.DictReader(f))

    print(f"Curated WVS/EVS rows: {len(curated_rows)}")
    print(f"Understanding Society questions to add: {len(US_QUESTIONS)}")
    assert len(US_QUESTIONS) == 19, "Expected exactly 19 US questions"

    all_rows = curated_rows + US_QUESTIONS
    print(f"Total rows: {len(all_rows)}")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nOutput written to: {OUTPUT_PATH}")
    print(f"Development check: should have {len(curated_rows)} + 19 = {len(all_rows)} rows total.")


if __name__ == "__main__":
    main()
