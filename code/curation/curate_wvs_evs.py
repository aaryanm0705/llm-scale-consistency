"""
curate_wvs_evs.py

Script 1 of the dataset curation pipeline.

Encodes the manually curated WVS (Waves 4-7) and EVS (2017) question set,
applies heuristic filtering and standardization, and normalizes question text.
Produces a single output: extracted_content/wvs_evs_curated_tidied.csv

This script merges the logic of the original extract_wvs_evs.py,
curate_questions.py, and tidy_question_text.py into a single pass.
No intermediate files are written.

Input:  raw_pdfs/ directory (WVS Waves 4-7 + EVS 2017 PDFs, verified for readability)
Output: extracted_content/wvs_evs_curated_tidied.csv
        extracted_content/curation_log.csv

Data flow:
  1. Build raw WVS/EVS records from hardcoded dictionaries
  2. Apply heuristic filtering and standardization
  3. Normalize question text
  4. Write output
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "raw_pdfs"
LOG_PATH = BASE_DIR / "extracted_content" / "curation_log.csv"
OUTPUT_PATH = BASE_DIR / "extracted_content" / "wvs_evs_curated_tidied.csv"


# ===========================================================================
# Stage 1 — Extract: WVS/EVS question dictionaries and record builders
# (from extract_wvs_evs.py)
# ===========================================================================

PDF_PATHS = {
    "WVS-4": PDF_DIR / "F00001316-WVS_2000_Questionnaire_Root.pdf",
    "WVS-5": PDF_DIR / "WVS-5.pdf",
    "WVS-6": PDF_DIR / "WVS-6.pdf",
    "WVS-7": PDF_DIR / "WVS-7.pdf",
    "EVS":   PDF_DIR / "ZA7505_bq_EVS2017.pdf",
}

FIELDNAMES = ["source", "question_id", "subject", "question",
              "scale_type", "answering_options"]

# ---------------------------------------------------------------------------
# Subject classification
# ---------------------------------------------------------------------------

SUBJECT_KEYWORDS = {
    "WORK AND EMPLOYMENT":    ["job", "work", "employ", "wage", "salary", "labor",
                               "labour", "career"],
    "FAMILY AND GENDER":      ["family", "marriage", "children", "mother", "father",
                               "husband", "wife", "housewife", "gender", "women", "men",
                               "daughter", "son", "parent"],
    "POLITICS AND SOCIETY":   ["politi", "government", "parliament", "democra",
                               "election", "party", "leader", "nation", "citizen",
                               "voting", "reform"],
    "RELIGION":               ["religion", "god", "church", "faith", "pray",
                               "belief", "spiritual", "atheist", "divine", "heaven",
                               "hell", "soul", "reincarn"],
    "ENVIRONMENT":            ["environment", "pollution", "climate", "nature",
                               "ecological", "green", "energy", "recycl"],
    "TRUST AND INSTITUTIONS": ["trust", "confident", "confidence", "institution",
                               "police", "army", "press", "civil service", "media",
                               "parliament", "courts", "un ", "united nations",
                               "organization"],
    "VALUES AND ETHICS":      ["justif", "moral", "ethics", "right", "wrong",
                               "abortion", "euthanasia", "suicide", "homosex",
                               "bribe", "cheat", "corrupt", "violence", "war",
                               "prostitut", "divorce"],
    "SCIENCE AND TECHNOLOGY": ["science", "technology", "research", "technical"],
    "IMMIGRATION":            ["immigr", "foreig", "ethnic", "diversity", "race",
                               "nationality", "ancestor"],
    "PERSONAL WELLBEING":     ["happy", "satisf", "health", "income", "freedom",
                               "choice", "poverty", "poor", "standard of living",
                               "sanitation", "disease", "education"],
}


def classify_subject(question_text: str) -> str:
    text_lower = question_text.lower()
    for subject, keywords in SUBJECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return subject
    return "GENERAL"


# ---------------------------------------------------------------------------
# Scale type classification
# ---------------------------------------------------------------------------

SCALE_TYPE_PATTERNS = {
    "agreement":      ["agree", "disagree"],
    "importance":     ["important", "essential"],
    "confidence":     ["great deal", "quite a lot"],
    "justifiability": ["never", "always"],
    "evaluation":     ["very good", "very bad"],
    "trust":          ["trust", "advantage"],
    "belief":         ["yes", "no"],
    "adequacy":       ["adequate", "yes"],
    "satisfaction":   ["satisfied", "dissatisfied"],
    "frequency":      ["always", "often", "rarely", "never"],
    "priority":       ["most important", "second choice"],
    "opinion":        ["individuals", "state"],
    "interest":       ["very interested", "not at all interested"],
    "extent":         ["not at all", "a great deal"],
    "comparison":     ["better", "worse"],
    "happiness":      ["very happy", "not at all happy"],
    "religiosity":    ["religious person", "atheist"],
    "fairness":       ["fair", "not fair"],
}


def classify_scale_type(options_text: str) -> str:
    opts_lower = options_text.lower()
    for scale_type, patterns in SCALE_TYPE_PATTERNS.items():
        if all(p in opts_lower for p in patterns[:2]):
            return scale_type
    if "important" in opts_lower:
        return "importance"
    if "agree" in opts_lower:
        return "agreement"
    if "confiden" in opts_lower or "great deal" in opts_lower:
        return "confidence"
    return "opinion"


# ---------------------------------------------------------------------------
# PDF verification
# ---------------------------------------------------------------------------

def verify_pdf(label: str, path: Path) -> bool:
    """Open PDF with pdfplumber to confirm it is readable. Returns True on success."""
    if not path.exists():
        print(f"  ERROR: PDF not found: {path}", file=sys.stderr)
        return False
    try:
        with pdfplumber.open(str(path)) as pdf:
            n_pages = len(pdf.pages)
        print(f"  {label}: {path.name} ({n_pages} pages)")
        return True
    except Exception as exc:
        print(f"  ERROR: Cannot open {path.name}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Standard answer scales reused across batteries
# ---------------------------------------------------------------------------

_CONF4 = "1: A great deal | 2: Quite a lot | 3: Not very much | 4: None at all"
_AGR4  = "1: Strongly Agree | 2: Agree | 3: Disagree | 4: Strongly Disagree"
_IMP4  = ("1: Very Important | 2: Rather Important | "
           "3: Not Very Important | 4: Not At All Important")
_EVGB4 = "1: Very good | 2: Fairly good | 3: Fairly bad | 4: Very bad"


# ---------------------------------------------------------------------------
# Subject overrides
# Derived from combined_dataset.csv (ground truth). classify_subject() uses
# keyword heuristics and often returns the wrong survey-section label.
# This table maps question_id -> the correct subject used in the reference.
# ---------------------------------------------------------------------------

SUBJECT_OVERRIDES = {
    # WVS-4 ----------------------------------------------------------------
    "V4":    "PERCEPTIONS OF LIFE",
    "V6":    "PERCEPTIONS OF LIFE",
    "V8":    "PERCEPTIONS OF LIFE",
    "V9":    "PERCEPTIONS OF LIFE",
    "V16":   "PERCEPTIONS OF LIFE",
    "V25":   "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V26":   "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V78":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V79":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V86":   "WORK",
    "V87":   "WORK",
    "V88":   "WORK",
    "V89":   "WORK",
    "V91":   "WORK",
    "V92":   "WORK",
    "V93":   "WORK",
    "V94":   "WORK",
    "V95":   "WORK",
    "V96":   "WORK",
    "V97":   "WORK",
    "V98":   "WORK",
    "V99":   "WORK",
    "V100":  "WORK",
    "V101":  "WORK",
    "V102":  "WORK",
    "V105":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V110":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V111":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V113":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V114":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V115":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V116":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V117":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V118":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V119":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V120":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V121":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V124":  "POLITICS AND SOCIETY",
    "V125":  "POLITICS AND SOCIETY",
    "V126":  "POLITICS AND SOCIETY",
    "V127":  "POLITICS AND SOCIETY",
    "V128":  "POLITICS AND SOCIETY",
    "V129":  "POLITICS AND SOCIETY",
    "V130":  "POLITICS AND SOCIETY",
    "V131":  "POLITICS AND SOCIETY",
    "V55":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V56":   "SCIENCE & TECHNOLOGY",
    "V80":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V103":  "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "V104":  "POLITICS AND SOCIETY",
    "V122":  "POLITICS AND SOCIETY",
    "V123":  "POLITICS AND SOCIETY",
    "V134":  "POLITICS AND SOCIETY",
    "V135":  "POLITICS AND SOCIETY",
    "V136":  "POLITICS AND SOCIETY",
    "V137":  "POLITICS AND SOCIETY",
    "V138":  "POLITICS AND SOCIETY",
    "V139":  "POLITICS AND SOCIETY",
    "V140":  "POLITICS AND SOCIETY",
    "V141":  "POLITICS AND SOCIETY",
    "V142":  "POLITICS AND SOCIETY",
    "V143":  "POLITICS AND SOCIETY",
    "V144":  "POLITICS AND SOCIETY",
    "V145":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V147":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V148":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V150":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V151":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V153":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V154":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V155":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V156":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V157":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V158":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V159":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V160":  "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "V162":  "Political Trust",
    "V164":  "POLITICS AND SOCIETY",
    "V165":  "POLITICS AND SOCIETY",
    "V166":  "POLITICS AND SOCIETY",
    "V167":  "POLITICS AND SOCIETY",
    "V175":  "POLITICS AND SOCIETY",
    "V181a": "GLOBAL ISSUES",
    "V188":  "RELIGION",
    "V192":  "RELIGION",
    # WVS-4: life importance battery (V1-V11)
    "V5":    "PERCEPTIONS OF LIFE",
    "V7":    "PERCEPTIONS OF LIFE",
    "V10":   "PERCEPTIONS OF LIFE",
    # WVS-4: child qualities battery (V17-V28)
    "V17":   "PERCEPTIONS OF LIFE",
    "V18":   "PERCEPTIONS OF LIFE",
    "V19":   "PERCEPTIONS OF LIFE",
    "V20":   "PERCEPTIONS OF LIFE",
    "V21":   "PERCEPTIONS OF LIFE",
    "V22":   "PERCEPTIONS OF LIFE",
    "V23":   "PERCEPTIONS OF LIFE",
    "V24":   "PERCEPTIONS OF LIFE",
    "V25b":  "PERCEPTIONS OF LIFE",
    # WVS-4: environment items
    "V33":   "ENVIRONMENT",
    "V34":   "ENVIRONMENT",
    "V35":   "ENVIRONMENT",
    # WVS-4: religion and meaning items
    "V182":  "RELIGION",
    "V186":  "RELIGION",
    # WVS-4: religious politics battery
    "V200":  "RELIGION",
    "V201":  "RELIGION",
    "V202":  "RELIGION",
    "V203":  "RELIGION",
    # WVS-4: global issues
    "V181c": "GLOBAL ISSUES",
    # WVS-6: security
    "V171":  "SECURITY",
    "V205":  "MORAL VALUES",
    "V207":  "PERCEPTIONS OF LIFE",
    "V208":  "PERCEPTIONS OF LIFE",
    "V209":  "PERCEPTIONS OF LIFE",
    "V210":  "PERCEPTIONS OF LIFE",
    "V211":  "MORAL VALUES",
    "V212":  "NATIONAL IDENTITY",
    "V213":  "NATIONAL IDENTITY",
    # WVS-5 ----------------------------------------------------------------
    "V245":  "GLOBAL ISSUES",
    "V246":  "GLOBAL ISSUES",
    "V247":  "GLOBAL ISSUES",
    "V248":  "GLOBAL ISSUES",
    "V249":  "GLOBAL ISSUES",
    "V214":  "NATIONAL IDENTITY",
    # WVS-7 ----------------------------------------------------------------
    "Q28":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "Q32":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "Q33":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "Q34":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "Q35":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "Q46":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "Q47":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "Q48":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "Q71":   "Political Trust",
    "Q71F":  "Middle East Regional Module",
    "Q72":   "Political Trust",
    "Q73":   "Political Trust",
    "Q74":   "Political Trust",
    "Q75":   "Political Trust",
    "Q76":   "Political Trust",
    "Q77":   "Political Trust",
    "Q78":   "Political Trust",
    "Q79":   "Political Trust",
    "Q80":   "Political Trust",
    "Q81":   "Political Trust",
    "Q82":   "Political Trust",
    "Q83":   "Political Trust",
    "Q84":   "Political Trust",
    "Q85":   "Political Trust",
    "Q89":   "Political Trust",
    "Q90":   "Political Trust",
    "Q91":   "Political Trust",
    "Q94":   "Political Trust",
    "Q160":  "SCIENCE & TECHNOLOGY",
    "Q161":  "SCIENCE & TECHNOLOGY",
    "Q162":  "SCIENCE & TECHNOLOGY",
    "Q163":  "SCIENCE & TECHNOLOGY",
    "Q164":  "SCIENCE & TECHNOLOGY",
    "Q165":  "RELIGIOUS VALUES",
    "Q166":  "RELIGIOUS VALUES",
    "Q173":  "RELIGIOUS VALUES",
    "Q224":  "Political Trust",
    "Q250":  "Elections",
    # EVS ------------------------------------------------------------------
    "v3":    "PERCEPTIONS OF LIFE",
    "v36":   "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    "v65":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v66":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v67":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v68":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v69":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v70":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v73":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v74":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v75":   "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    "v85":   "PERCEPTIONS OF LIFE",
    "v113":  "POLITICS AND SOCIETY",
    "v114":  "POLITICS AND SOCIETY",
    "v117":  "Political Trust",
    "v123":  "Political Trust",
    "v125":  "Political Trust",
    "v132":  "Political Trust",
    "v149":  "MORAL VALUES",
    "v150":  "MORAL VALUES",
    "v152":  "MORAL VALUES",
    "v153":  "MORAL VALUES",
    "v154":  "MORAL VALUES",
    "v155":  "MORAL VALUES",
    "v156":  "MORAL VALUES",
    "v157":  "MORAL VALUES",
    "v158":  "MORAL VALUES",
    "v159":  "MORAL VALUES",
    "v160":  "MORAL VALUES",
    "v161":  "MORAL VALUES",
    "v162":  "MORAL VALUES",
    "v163":  "MORAL VALUES",
    "v184":  "POLITICS AND SOCIETY",
    "v185":  "POLITICS AND SOCIETY",
    "v186":  "POLITICS AND SOCIETY",
    "v187":  "POLITICS AND SOCIETY",
    "v188":  "POLITICS AND SOCIETY",
    # US -------------------------------------------------------------------
    "SCTRUST": "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
}

# Source-specific overrides: (source_short, question_id) -> subject
# Used when the same question_id appears in multiple waves with different questions.
SUBJECT_OVERRIDES_BY_SOURCE = {
    ("WVS-4", "V120"): "POLITICS AND SOCIETY",
    ("WVS-4", "V121"): "POLITICS AND SOCIETY",
    ("WVS-4", "V122"): "POLITICS AND SOCIETY",
    ("WVS-4", "V123"): "POLITICS AND SOCIETY",
    ("WVS-4", "V208"): "MORAL VALUES",
    ("WVS-4", "V209"): "MORAL VALUES",
    ("WVS-4", "V210"): "MORAL VALUES",
    ("WVS-4", "V213"): "MORAL VALUES",
    ("WVS-5", "V120"): "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    ("WVS-5", "V121"): "SOCIAL CAPITAL, TRUST & ORGANIZATIONAL MEMBERSHIP",
    ("WVS-5", "V208"): "PERCEPTIONS OF LIFE",
    ("WVS-5", "V209"): "PERCEPTIONS OF LIFE",
    ("WVS-5", "V210"): "PERCEPTIONS OF LIFE",
    ("WVS-5", "V213"): "NATIONAL IDENTITY",
    ("WVS-6", "V210"): "PERCEPTIONS OF LIFE",
    ("WVS-5", "V117"): "SCIENCE & TECHNOLOGY",
    ("WVS-5", "V200"): "SOCIAL VALUES, ATTITUDES & STEREOTYPES",
    ("WVS-6", "V213"): "POLITICS AND SOCIETY",
    ("WVS-7", "Q30"):  "Trust",
    ("WVS-7", "Q49"):  "HAPPINESS AND WELL-BEING",
    ("WVS-7", "Q50"):  "HAPPINESS AND WELL-BEING",
    ("WVS-7", "Q174"): "ETHICAL VALUES AND NORMS",
    ("WVS-7", "Q182"): "CORRUPTION",
    ("WVS-7", "Q222"): "Middle East Regional Module",
    ("WVS-7", "Q223"): "Middle East Regional Module",
    ("WVS-7", "Q235"): "Political System",
}


# ---------------------------------------------------------------------------
# WVS-4 question database
# Questions verified against F00001316-WVS_2000_Questionnaire_Root.pdf.
# ---------------------------------------------------------------------------

def _make_rec(source, qid, question, options, scale_type=None):
    if scale_type is None:
        scale_type = classify_scale_type(options)
    src_short = source.split(" ")[0]
    subject = SUBJECT_OVERRIDES_BY_SOURCE.get(
        (src_short, qid),
        SUBJECT_OVERRIDES.get(qid, classify_subject(question))
    )
    return {
        "source":            source,
        "question_id":       qid,
        "subject":           subject,
        "question":          question,
        "scale_type":        scale_type,
        "answering_options": options,
    }


def build_wvs4_records():
    src = "WVS-4 (1999-2002)"
    records = []

    def add(qid, question, options, scale_type=None):
        records.append(_make_rec(src, qid, question, options, scale_type))

    # Life domains importance battery (V4-V9)
    for qid, item in [("V4", "family"), ("V5", "friends"), ("V6", "leisure time"),
                      ("V7", "politics"), ("V8", "work"), ("V9", "religion")]:
        if qid == "V5":
            add(qid, "How important are friends in life?", _IMP4, "importance")
        elif qid == "V7":
            add(qid, "How important are politics in life?", _IMP4, "importance")
        else:
            add(qid, f"How important is {item} in life?", _IMP4, "importance")

    # Child qualities battery (V16-V25 range, selected items retained)
    child_qualities = [
        ("V16",  "hard work"),
        ("V17",  "feeling of responsibility"),
        ("V18",  "imagination"),
        ("V19",  "tolerance and respect for other people"),
        ("V20",  "thrift, saving money and things"),
        ("V21",  "determination, perseverance"),
        ("V22",  "religious faith"),
        ("V23",  "unselfishness"),
        ("V24",  "obedience"),
        ("V25b", "independence"),
    ]
    for qid, quality in child_qualities:
        add(qid, f"How important is {quality}?",
            "Important | Not Important", "importance")

    # Life domain importance: service to others (V10)
    add("V10", "How important is service to others in life?", _IMP4, "importance")

    # Job aspects battery (V86-V96, V90 excluded)
    job_items = [
        ("V86", "good pay"),
        ("V87", "not too much pressure"),
        ("V88", "good job security"),
        ("V89", "a job respected by people in general"),
        ("V91", "an opportunity to use initiative"),
        ("V92", "generous holidays"),
        ("V93", "a job in which you feel you can achieve something"),
        ("V94", "a responsible job"),
        ("V95", "a job that is interesting"),
        ("V96", "a job that meets one's abilities"),
    ]
    for qid, item in job_items:
        add(qid, f"Is {item} important in a job?",
            "Important | Not Important", "importance")

    # Work ethic battery (V97-V102)
    work_ethics = [
        ("V97",  "To fully develop your talents, you need to have a job"),
        ("V98",  "It is humiliating to receive money without having to work for it"),
        ("V99",  "People who don't work turn lazy"),
        ("V100", "Work is a duty towards society"),
        ("V101", "People should not have to work if they don't want to"),
        ("V102", "Work should always come first, even if it means less spare time"),
    ]
    for qid, stmt in work_ethics:
        add(qid, stmt, _AGR4, "agreement")

    # Confidence in institutions battery (V147-V162, V149/V152 dropped in Script 2)
    conf_items = [
        ("V147", "the churches"),
        ("V148", "the armed forces"),
        ("V149", "the press"),
        ("V150", "television"),
        ("V151", "labor unions"),
        ("V152", "the police"),
        ("V153", "the government in your capital"),
        ("V154", "political parties"),
        ("V155", "parliament"),
        ("V156", "the civil service"),
        ("V157", "major companies"),
        ("V158", "the environmental protection movement"),
        ("V159", "the women's movement"),
        ("V160", "the european union"),
        ("V161", "NATO"),
        ("V162", "the United Nations"),
    ]
    for qid, inst in conf_items:
        add(qid, f"How confident are you in {inst}?", _CONF4, "confidence")

    # Ways of governing battery (V164-V167)
    governing = [
        ("V164", "having a strong leader who does not have to bother with parliament and elections"),
        ("V165", "having experts, not government, make decisions according to what they think is best for the country"),
        ("V166", "having the army rule"),
        ("V167", "having a democratic political system"),
    ]
    for qid, way in governing:
        add(qid, f"Is {way} as a way of governing a country good?",
            "Very Good | Good | Bad | Very Bad", "evaluation")

    # Religion adequacy battery (V187-V190)
    adequacy_items = [
        ("V187", "the moral problems and needs of the individual"),
        ("V188", "the problems of family life"),
        ("V189", "people's spiritual needs"),
        ("V190", "the social problems facing our country today"),
    ]
    for qid, area in adequacy_items:
        add(qid,
            f"Generally speaking, the churches in a country are giving adequate answers to {area}?",
            "1: Yes | 2: No", "adequacy")

    # Religious beliefs battery (V191-V195)
    beliefs = [
        ("V191", "God"),
        ("V192", "life after death"),
        ("V193", "people have a soul"),
        ("V194", "hell"),
        ("V195", "heaven"),
    ]
    for qid, thing in beliefs:
        add(qid, f"Do you believe in {thing}?", "1: Yes | 2: No", "belief")

    # Religious politics battery (V200-V203)
    rel_pol = [
        ("V200", "Politicians who do not believe in God are unfit for public office"),
        ("V201", "Religious leaders should not influence how people vote in elections"),
        ("V202", "It would be better for a country if more people with strong religious beliefs held public office"),
        ("V203", "Religious leaders should not influence government decisions"),
    ]
    for qid, stmt in rel_pol:
        add(qid, f"How much {stmt}?",
            "Agree Strongly | Agree | Neither agree or disagree | Disagree | Strongly disagree",
            "agreement")

    # Justifiability battery (V204-V213)
    justif_items = [
        ("V204", "claiming government benefits to which you are not entitled"),
        ("V205", "avoiding a fare on public transport"),
        ("V206", "cheating on taxes if you have a chance"),
        ("V207", "someone accepting a bribe in the course of their duties"),
        ("V208", "homosexuality"),
        ("V209", "prostitution"),
        ("V210", "abortion"),
        ("V211", "divorce"),
        ("V212", "euthanasia -- ending the life of the incurably sick"),
        ("V213", "suicide"),
    ]
    for qid, act in justif_items:
        add(qid, f"Can {act} be justified?",
            "1: Never Justifiable | 10: Always Justifiable", "justifiability")

    # Environment agreement battery (V33-V35)
    env_items = [
        ("V33", "People would give part of their income if they were certain that the money would be used to prevent environmental pollution"),
        ("V34", "People would agree to an increase in taxes if the extra money were used to prevent environmental pollution"),
        ("V35", "The government should reduce environmental pollution, but it should not cost individuals any money"),
    ]
    for qid, stmt in env_items:
        add(qid, stmt, _AGR4, "agreement")

    # National goals (V120-V123)
    nat_goals = [
        ("V120", "maintaining order in the nation",
         "Maintaining order in the nation"),
        ("V121", "giving people more say in important government decisions",
         "Giving people more say in important government decisions"),
        ("V122", "fighting rising prices",
         "Fighting rising prices"),
        ("V123", "protecting freedom of speech",
         "Protecting freedom of speech"),
    ]
    for qid, goal, oao in nat_goals:
        add(qid, f"How important is {goal} as a national goal?", oao, "opinion")

    # Societal goals (V124-V125, V134-V135 = first set; V136-V139 = second set)
    soc_goals = [
        ("V124", "a stable economy",
         "A stable economy"),
        ("V125", "progress toward a less impersonal and more humane society",
         "Progress toward a less impersonal and more humane society"),
        ("V134", "progress toward a society in which ideas count more than money",
         "Progress toward a society in which Ideas count more than money"),
        ("V135", "the fight against crime",
         "The fight against crime"),
        ("V136", "a stable economy",
         "A stable economy"),
        ("V137", "progress toward a less impersonal and more humane society",
         "Progress toward a less impersonal and more humane society"),
        ("V138", "progress toward a society in which ideas count more than money",
         "Progress toward a society in which Ideas count more than money"),
        ("V139", "the fight against crime",
         "The fight against crime"),
    ]
    for qid, goal, oao in soc_goals:
        add(qid, f"How important is {goal} as a societal goal?", oao, "opinion")

    # Standalone items
    standalones = [
        ("V25",   "Most people can be trusted or that you need to be very careful in dealing with people?",
                  "Most people can be trusted | Need to be very careful", "trust"),
        ("V26",   "Most people would try to take advantage of others if they got a chance, or would they try to be fair?",
                  "Would take advantage | Would try to be fair", "trust"),
        ("V78",   "When jobs are scarce, men should have more right to a job than women",
                  "Agree | Neither | Disagree", "agreement"),
        ("V79",   "When jobs are scarce, employers should give priority to people of a country over immigrants",
                  "Agree | Neither | Disagree", "agreement"),
        ("V83",   "Which point on this scale most clearly describes how much weight you place on work (including housework and schoolwork), as compared with leisure or recreation?",
                  "1: Leisure makes life worth living | 5: Work makes life worth living", "comparison"),
        ("V103",  "Imagine two secretaries, of the same age, doing practically the same job. One finds out that the other earns considerably more than she does. The better paid secretary, however, is quicker, more efficient and more reliable at her job. is it fair or not fair that one secretary is paid more than the other?",
                  "1: Fair | 2: Not fair", "fairness"),
        ("V105",  "People have different ideas about following instructions at work. Should one follow instructions even when one does not fully agree with them, or only when one is convinced they are right?",
                  "1: Should Follow Instructions | 2: Must Be Convinced First", "opinion"),
        ("V109",  "If someone says a child needs a home with both a father and a mother to grow up happily, would you tend to agree or disagree?",
                  "1: Tend To Agree | 2: Tend To Disagree", "agreement"),
        ("V110",  "A woman has to have children in order to be fulfilled or is this not necessary?",
                  "Needs children | Not necessary", "opinion"),
        ("V111",  "\"Marriage is an out-dated institution\"",
                  _AGR4, "agreement"),
        ("V113",  "One of my main goals in life has been to make my parents proud",
                  _AGR4, "agreement"),
        ("V114",  "I make a lot of effort to live up to what my friends expect",
                  _AGR4, "agreement"),
        ("V115",  "A working mother can establish just as warm and secure a relationship with her children as a mother who does not work",
                  _AGR4, "agreement"),
        ("V116",  "Being a housewife is just as fulfilling as working for pay",
                  _AGR4, "agreement"),
        ("V117",  "Both the husband and wife should contribute to household income",
                  _AGR4, "agreement"),
        ("V118",  "On the whole, men make better political leaders than women do",
                  _AGR4, "agreement"),
        ("V119",  "A university education is more important for a boy than for a girl",
                  _AGR4, "agreement"),
        ("V133",  "How interested you are in politics: Very interested, somewhat interested, not very interested, or not at all interested?",
                  "Very interested | Somewhat interested | Not very interested | Not at all interested",
                  "interest"),
        ("V172",  "Democracy may have problems but it's better than any other form of government",
                  _AGR4, "agreement"),
        ("V175",  "This country is run by a few big interests looking out for themselves, or that it is run for the benefit of all the people?",
                  "Run by a few big interests | Run for all the people", "agreement"),
        ("V181a", "A country cannot solve its environmental problems by itself, but needs to collaborate with international environmental organizations",
                  _AGR4, "agreement"),
        ("V181c", "A country cannot solves its problems of unemployment by itself, but needs to collaborate with international economic organizations",
                  _AGR4, "agreement"),
        ("V182",  "How often, if at all, about the meaning and purpose of life: Often, sometimes, rarely, or never?",
                  "Often | Sometimes | Rarely | Never", "frequency"),
        ("V186",  "Independently of whether you go to church or not, you are a religious person, not a religious person, or a convinced atheist?",
                  "A religious person | Not a religious person | A convinced atheist", "religiosity"),
        ("V196",  "How important is God in life? Please use this scale to indicate where 10 means very important and 1 means not at all important.?",
                  "Not at all | Very", "importance"),
        ("V127",  "Would less emphasis on money and material possessions be a good thing or bad thing?",
                  _EVGB4, "evaluation"),
        ("V128",  "Would less importance placed on work in our lives be a good thing or bad thing?",
                  _EVGB4, "evaluation"),
        ("V129",  "Would more emphasis on the development of technology be a good thing or bad thing?",
                  _EVGB4, "evaluation"),
        ("V130",  "Would greater respect for authority be a good thing or bad thing?",
                  _EVGB4, "evaluation"),
        ("V131",  "Would more emphasis on family life be a good thing or bad thing?",
                  _EVGB4, "evaluation"),
        # New WVS-4 questions
        ("V55",   "Which statement is more accurate: (A) Regardless of what the qualities and faults of one's parents are, one must always love and respect them, or (B) One does not have the duty to respect and love parents who have not earned it by their behavior and attitudes?",
                  "Tend to agree with statement A | Tend to agree with statement B", "agreement"),
        ("V56",   "In the long run, the scientific advances we are making will help or harm mankind?",
                  "Will help | Will harm | Some of each", "evaluation"),
        ("V80",   "What about parents' responsibilities to their children?",
                  "Parents' duty is to do their best for their children even at the expense of their own well-being | Parents have a life of their own and should not be asked to sacrifice their own well-being for the sake of their children | Neither",
                  "opinion"),
        ("V104",  "How satisfied are you with the way the people now in national office are handling the country's affairs? you are very satisfied, fairly satisfied, fairly dissatisfied or very dissatisfied?",
                  "Very satisfied | Fairly satisfied | Fairly dissatisfied | Very dissatisfied",
                  "satisfaction"),
        ("V126",  "How often do people discuss political matters?",
                  "Very Often | Often | Rarely | Never", "frequency"),
    ]
    already = {r["question_id"] for r in records}
    for qid, question, options, scale_type in standalones:
        if qid not in already:
            add(qid, question, options, scale_type)

    # World problems battery (V140-V144, WVS-4)
    world_problems = [
        ("V140", "people living in poverty and need",
         "People living in poverty and need"),
        ("V141", "discrimination against girls and women",
         "Discrimination against girls and women"),
        ("V142", "poor sanitation and infectious diseases",
         "Poor sanitation and infectious diseases"),
        ("V143", "inadequate education",
         "Inadequate education"),
        ("V144", "environmental pollution",
         "Environmental pollution"),
    ]
    for qid, problem, oao in world_problems:
        add(qid, f"How serious a problem for the world is {problem}?", oao, "opinion")

    return records


# ---------------------------------------------------------------------------
# EVS 2017 question database
# Questions verified against ZA7505_bq_EVS2017.pdf.
# ---------------------------------------------------------------------------

EVS_QUESTIONS = {
    "v3":   ("EVS-v3",   "PERSONAL WELLBEING",
             "How important are friends and acquaintances in life?",
             "Very Important | Rather Important | Not Very Important | Not at All Important", "importance"),
    "v36":  ("EVS-v36",  "TRUST AND INSTITUTIONS",
             "How much do you trust people of another religion?",
             "Trust Completely | Trust Somewhat | Do Not Trust Very Much | Do Not Trust at All", "trust"),
    "v37":  ("EVS-v37",  "TRUST AND INSTITUTIONS",
             "How much do you trust people of another nationality?",
             "Trust Completely | Trust Somewhat | Do Not Trust Very Much | Do Not Trust at All", "trust"),
    "v57":  ("EVS-v57",  "RELIGION",
             "Do you believe in God?",
             "1: Yes | 2: No", "belief"),
    "v58":  ("EVS-v58",  "RELIGION",
             "Do you believe in life after death?",
             "1: Yes | 2: No", "belief"),
    "v59":  ("EVS-v59",  "RELIGION",
             "Do you believe in hell?",
             "1: Yes | 2: No", "belief"),
    "v60":  ("EVS-v60",  "RELIGION",
             "Do you believe in heaven?",
             "1: Yes | 2: No", "belief"),
    "v61":  ("EVS-v61",  "RELIGION",
             "Do you believe in re-incarnation?",
             "1: Yes | 2: No", "belief"),
    "v62":  ("EVS-v62",  "RELIGION",
             "Which statement comes closest to your beliefs about God/spirit/life force?",
             "Personal God | Spirit/life force | Don't know | No spirit/God",
             "belief"),
    "v65":  ("EVS-v65",  "FAMILY AND GENDER",
             "How important is faithfulness for a successful marriage?",
             "Very Important | Rather Important | Not Very Important", "importance"),
    "v66":  ("EVS-v66",  "FAMILY AND GENDER",
             "How important is an adequate income for a successful marriage?",
             "Very Important | Rather Important | Not Very Important", "importance"),
    "v67":  ("EVS-v67",  "FAMILY AND GENDER",
             "How important is good housing for a successful marriage?",
             "Very Important | Rather Important | Not Very Important", "importance"),
    "v68":  ("EVS-v68",  "FAMILY AND GENDER",
             "How important is sharing household chores for a successful marriage?",
             "Very Important | Rather Important | Not Very Important", "importance"),
    "v69":  ("EVS-v69",  "FAMILY AND GENDER",
             "How important are children for a successful marriage?",
             "Very Important | Rather Important | Not Very Important", "importance"),
    "v70":  ("EVS-v70",  "FAMILY AND GENDER",
             "How important is having time for one's own friends and hobbies for a successful marriage?",
             "Very Important | Rather Important | Not Very Important", "importance"),
    "v73":  ("EVS-v73",  "FAMILY AND GENDER",
             "A job is alright but what most women really want is a home and children",
             "1: Strongly Agree | 2: Agree | 3: Disagree | 4: Strongly Disagree", "agreement"),
    "v74":  ("EVS-v74",  "FAMILY AND GENDER",
             "All in all, family life suffers when the woman has a full-time job",
             "1: Strongly Agree | 2: Agree | 3: Disagree | 4: Strongly Disagree", "agreement"),
    "v75":  ("EVS-v75",  "FAMILY AND GENDER",
             "A man's job is to earn money; a woman's job is to look after the home and family",
             "1: Strongly Agree | 2: Agree | 3: Disagree | 4: Strongly Disagree", "agreement"),
    "v85":  ("EVS-v85",  "VALUES AND ETHICS",
             "How important is good manners?",
             "Important | Not Important", "importance"),
    "v103": ("EVS-v103", "POLITICS AND SOCIETY",
             "Should individuals take more responsibility for providing for themselves or should the state take more responsibility?",
             "1: Individuals | 10: State", "opinion"),
    "v104": ("EVS-v104", "POLITICS AND SOCIETY",
             "Should unemployed people have to take any job available or have the right to refuse a job?",
             "Must Take Any Job | Should Take Any Job | Neutral | Can Refuse | Should Refuse", "opinion"),
    "v105": ("EVS-v105", "POLITICS AND SOCIETY",
             "Is competition good or harmful?",
             "Very Good | Good | Neutral | Harmful | Very Harmful", "opinion"),
    "v106": ("EVS-v106", "POLITICS AND SOCIETY",
             "Should incomes be made more equal or should there be greater incentives for individual effort?",
             "Much More Equal | More Equal | Neutral | Greater Incentives | Much Greater Incentives", "opinion"),
    "v107": ("EVS-v107", "POLITICS AND SOCIETY",
             "Should private ownership or government ownership of business be increased?",
             "1: Private | 10: Government", "opinion"),
    "v113": ("EVS-v113", "VALUES AND ETHICS",
             "Would less importance placed on work in our lives be a good thing or bad thing?",
             "Good | Bad | Don't Mind", "evaluation"),
    "v114": ("EVS-v114", "VALUES AND ETHICS",
             "Would greater respect for authority be a good thing or bad thing?",
             "Good | Bad | Don't Mind", "evaluation"),
    "v117": ("EVS-v117", "TRUST AND INSTITUTIONS",
             "How confident are you in the education system?",
             "1: A great deal | 2: Quite a lot | 3: Not very much | 4: None at all", "confidence"),
    "v123": ("EVS-v123", "TRUST AND INSTITUTIONS",
             "How confident are you in the social security system?",
             "1: A great deal | 2: Quite a lot | 3: Not very much | 4: None at all", "confidence"),
    "v125": ("EVS-v125", "TRUST AND INSTITUTIONS",
             "How confident are you in the United Nations?",
             "1: A great deal | 2: Quite a lot | 3: Not very much | 4: None at all", "confidence"),
    "v126": ("EVS-v126", "TRUST AND INSTITUTIONS",
             "How confident are you in the health care system?",
             "1: A great deal | 2: Quite a lot | 3: Not very much | 4: None at all", "confidence"),
    "v127": ("EVS-v127", "TRUST AND INSTITUTIONS",
             "How confident are you in the justice system?",
             "1: A great deal | 2: Quite a lot | 3: Not very much | 4: None at all", "confidence"),
    "v132": ("EVS-v132", "TRUST AND INSTITUTIONS",
             "How confident are you in social media?",
             "1: A great deal | 2: Quite a lot | 3: Not very much | 4: None at all", "confidence"),
    "v133": ("EVS-v133", "POLITICS AND SOCIETY",
             "How essential is it that governments tax the rich and subsidize the poor as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v134": ("EVS-v134", "POLITICS AND SOCIETY",
             "How essential is it that religious authorities interpret the laws as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v135": ("EVS-v135", "POLITICS AND SOCIETY",
             "How essential is it that people choose their leaders in free elections as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v136": ("EVS-v136", "POLITICS AND SOCIETY",
             "How essential is it that people receive state aid for unemployment as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v137": ("EVS-v137", "POLITICS AND SOCIETY",
             "How essential is it that the army takes over when government is incompetent as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v138": ("EVS-v138", "POLITICS AND SOCIETY",
             "How essential is it that civil rights protect people from state oppression as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v139": ("EVS-v139", "POLITICS AND SOCIETY",
             "How essential is it that the state makes people's incomes equal as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v140": ("EVS-v140", "POLITICS AND SOCIETY",
             "How essential is it that people obey their rulers as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v141": ("EVS-v141", "POLITICS AND SOCIETY",
             "How essential is it that women have the same rights as men as a characteristic of democracy?",
             "Not Essential | Slightly Essential | Moderately Essential | Very Essential | Extremely Essential", "importance"),
    "v146": ("EVS-v146", "POLITICS AND SOCIETY",
             "Is having experts, not government, make decisions a good or bad way of governing?",
             "Very Good | Good | Bad | Very Bad", "evaluation"),
    "v149": ("EVS-v149", "VALUES AND ETHICS",
             "Can claiming state benefits which you are not entitled to be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v150": ("EVS-v150", "VALUES AND ETHICS",
             "Can cheating on tax be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v151": ("EVS-v151", "MORAL VALUES",
             "Can taking drugs marijuana or hashish be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v152": ("EVS-v152", "VALUES AND ETHICS",
             "Can someone accepting a bribe be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v153": ("EVS-v153", "VALUES AND ETHICS",
             "Can homosexuality be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v154": ("EVS-v154", "VALUES AND ETHICS",
             "Can abortion be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v155": ("EVS-v155", "VALUES AND ETHICS",
             "Can divorce be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v156": ("EVS-v156", "VALUES AND ETHICS",
             "Can euthanasia be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v157": ("EVS-v157", "VALUES AND ETHICS",
             "Can suicide be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v158": ("EVS-v158", "VALUES AND ETHICS",
             "Can having casual sex be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v159": ("EVS-v159", "VALUES AND ETHICS",
             "Can avoiding a fare on public transport be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v160": ("EVS-v160", "VALUES AND ETHICS",
             "Can prostitution be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v161": ("EVS-v161", "VALUES AND ETHICS",
             "Can artificial insemination or in-vitro fertilization be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v162": ("EVS-v162", "VALUES AND ETHICS",
             "Can political violence be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v163": ("EVS-v163", "VALUES AND ETHICS",
             "Can death penalty be justified?",
             "1: Never | 10: Always", "justifiability"),
    "v184": ("EVS-v184", "IMMIGRATION",
             "What is the impact of immigrants on a country's development?",
             "Very Good | Good | Bad | Very Bad", "evaluation"),
    "v185": ("EVS-v185", "IMMIGRATION",
             "Do immigrants take jobs away from citizens?",
             "Definitely Take Jobs | Take Jobs | Neutral | Do Not Take Jobs | Definitely Do Not Take Jobs", "agreement"),
    "v186": ("EVS-v186", "IMMIGRATION",
             "Do immigrants make crime problems worse?",
             "Much Worse | Worse | Neutral | Do Not Make Worse | Much Better", "agreement"),
    "v187": ("EVS-v187", "IMMIGRATION",
             "Are immigrants a strain on a country's welfare system?",
             "Major Strain | Some Strain | Neutral | Not a Strain | Help Economy", "agreement"),
    "v188": ("EVS-v188", "IMMIGRATION",
             "Should immigrants maintain their distinct customs and traditions?",
             "Should Maintain | Better Maintain | Neutral | Better Adapt | Should Adapt", "agreement"),
    "v201": ("EVS-v201", "ENVIRONMENT",
             "There are more important things to do in life than protect the environment",
             "Strongly Agree | Agree | Neither Agree nor Disagree | Disagree | Strongly Disagree", "agreement"),
    "v203": ("EVS-v203", "ENVIRONMENT",
             "Many of the claims about environmental threats are exaggerated",
             "Strongly Agree | Agree | Neither Agree nor Disagree | Disagree | Strongly Disagree", "agreement"),
    "v204": ("EVS-v204", "ENVIRONMENT",
             "Which comes closer to your own view: protecting the environment should be given priority, even if it causes slower economic growth; or economic growth and creating jobs should be the top priority, even if the environment suffers to some extent?",
             "1: Environment Priority | 2: Economic Growth Priority", "opinion"),
    "v205": ("EVS-v205", "POLITICS AND SOCIETY",
             "Should governments have the right to keep people under video surveillance in public areas?",
             "Definitely should | Probably should | Probably should not | Definitely should not", "agreement"),
    "v206": ("EVS-v206", "POLITICS AND SOCIETY",
             "Should governments have the right to monitor all emails and internet information?",
             "Definitely should | Probably should | Probably should not | Definitely should not", "agreement"),
    "v207": ("EVS-v207", "POLITICS AND SOCIETY",
             "Should governments have the right to collect information about anyone without their knowledge?",
             "Definitely should | Probably should | Probably should not | Definitely should not", "agreement"),
    "v221": ("EVS-v221", "POLITICS AND SOCIETY",
             "How important is eliminating big inequalities in income between citizens?",
             "Very Important | Rather Important | Not Very Important | Not at All Important", "importance"),
    "v222": ("EVS-v222", "POLITICS AND SOCIETY",
             "How important is guaranteeing that basic needs are met for all?",
             "Very Important | Rather Important | Not Very Important | Not at All Important", "importance"),
    "v223": ("EVS-v223", "POLITICS AND SOCIETY",
             "How important is recognizing people on their merits?",
             "Very Important | Rather Important | Not Very Important | Not at All Important", "importance"),
    "v224": ("EVS-v224", "POLITICS AND SOCIETY",
             "How important is protecting against terrorism?",
             "Very Important | Rather Important | Not Very Important | Not at All Important", "importance"),
}


def build_evs_records():
    records = []
    for var_code, (source, subject, question, options, scale_type) in EVS_QUESTIONS.items():
        records.append({
            "source":            source,
            "question_id":       var_code,
            "subject":           SUBJECT_OVERRIDES.get(var_code, subject),
            "question":          question,
            "scale_type":        scale_type,
            "answering_options": options,
        })
    return records


# ---------------------------------------------------------------------------
# WVS-5, WVS-6, WVS-7 question database
# Questions verified against WVS-5.pdf, WVS-6.pdf, WVS-7.pdf.
# ---------------------------------------------------------------------------

WVS567_QUESTIONS = {
    "WVS-5 (2005-2006)": {
        "V117": ("SCIENCE AND TECHNOLOGY",
                 "All things considered, the world is better off, or worse off, because of science and technology? Please tell me which comes closest to your view on this scale: 1 means that \"the world is a lot worse off,\" and 10 means that \"the world is a lot better off.\"",
                 "A lot worse off | A lot better off", "evaluation"),
        "V120": ("TRUST AND INSTITUTIONS",
                 "How confident are you in environmental organizations?",
                 _CONF4, "confidence"),
        "V121": ("TRUST AND INSTITUTIONS",
                 "How confident are you in charitable or humanitarian organizations?",
                 _CONF4, "confidence"),
        "V212": ("POLITICS AND SOCIETY",
                 "How important is having ancestors from the country as a requirement for citizenship?",
                 "Very Important | Rather Important | Not Very Important | Not at All Important", "importance"),
        "V213": ("POLITICS AND SOCIETY",
                 "How important is abiding by the country's laws as a requirement for citizenship?",
                 "Very Important | Rather Important | Not Very Important | Not at All Important", "importance"),
        "V214": ("IMMIGRATION",
                 "Turning to the question of ethnic diversity, Which statement is more accurate? Please use this scale to indicate your position where 1 means \"ethnic diversity erodes a country's unity\" and 10 means \"ethnic diversity enriches life\".",
                 "Ethnic diversity erodes a country's unity | Ethnic diversity enriches life", "agreement"),
        "V105": ("TRUST AND INSTITUTIONS",
                 "Would most people try to take advantage of you if they got a chance, or would they try to be fair?",
                 "Trust Completely | Trust Somewhat | Neutral | Trust a Little | Do Not Trust at All", "trust"),
        "V200": ("FAMILY AND GENDER",
                 "On the whole, men make better business executives than women do",
                 _AGR4, "agreement"),
        "V207": ("VALUES AND ETHICS",
                 "Making one's parents proud should be a main goal in life",
                 _AGR4, "agreement"),
        "V208": ("VALUES AND ETHICS",
                 "People should seek to be themselves rather than follow others",
                 _AGR4, "agreement"),
        "V209": ("VALUES AND ETHICS",
                 "People should make effort to live up to what their friends expect",
                 _AGR4, "agreement"),
        "V210": ("VALUES AND ETHICS",
                 "People should decide their goals in life by themselves",
                 _AGR4, "agreement"),
        "V227": ("PERSONAL WELLBEING",
                 "How serious a problem for the world is people living in poverty and need?",
                 "1: Not A Problem | 2: Moderate Problem | 3: Serious Problem | 4: Very Serious Problem",
                 "importance"),
        "V228": ("PERSONAL WELLBEING",
                 "How serious a problem for the world is discrimination against girls and women?",
                 "1: Not A Problem | 2: Moderate Problem | 3: Serious Problem | 4: Very Serious Problem",
                 "importance"),
        "V229": ("PERSONAL WELLBEING",
                 "How serious a problem for the world is poor sanitation and infectious diseases?",
                 "1: Not A Problem | 2: Moderate Problem | 3: Serious Problem | 4: Very Serious Problem",
                 "importance"),
        "V230": ("PERSONAL WELLBEING",
                 "How serious a problem for the world is inadequate education?",
                 "1: Not A Problem | 2: Moderate Problem | 3: Serious Problem | 4: Very Serious Problem",
                 "importance"),
        "V231": ("ENVIRONMENT",
                 "How serious a problem for the world is environmental pollution?",
                 "1: Not A Problem | 2: Moderate Problem | 3: Serious Problem | 4: Very Serious Problem",
                 "importance"),
        "V245": ("PERSONAL WELLBEING",
                 "How serious a problem in your country is people living in poverty and need?",
                 "People living in poverty and need", "opinion"),
        "V246": ("PERSONAL WELLBEING",
                 "How serious a problem in your country is discrimination of girls and women?",
                 "Discrimination of girls and women", "opinion"),
        "V247": ("PERSONAL WELLBEING",
                 "How serious a problem in your country is poor sanitation and infectious diseases?",
                 "Poor sanitation and infectious diseases", "opinion"),
        "V248": ("PERSONAL WELLBEING",
                 "How serious a problem in your country is inadequate education?",
                 "Inadequate education", "opinion"),
        "V249": ("ENVIRONMENT",
                 "How serious a problem in your country is environmental pollution?",
                 "Environmental pollution", "opinion"),
    },
    "WVS-6 (2010-2014)": {
        "V145":  ("TRUST AND INSTITUTIONS",
                  "How confident are you in universities?",
                  _CONF4, "confidence"),
        "V145b": ("RELIGION",
                  "With which one of the following statements most? The basic meaning of religion is: To follow religious norms and ceremonies, or to do good to other people?",
                  "To follow religious norms and ceremonies | To do good to other people",
                  "agreement"),
        "V145c": ("RELIGION",
                  "And with which of the following statements most? The basic meaning of religion is: To make sense of life after death, or to make sense of life in this world?",
                  "To make sense of life after death | To make sense of life in this world",
                  "agreement"),
        "V171":  ("VALUES AND ETHICS",
                  "\"Under some conditions, war is necessary to obtain justice.\"",
                  _AGR4, "agreement"),
        "V210":  ("VALUES AND ETHICS",
                  "How important is self-expression?",
                  "Important | Not Important", "importance"),
        "V213":  ("POLITICS AND SOCIETY",
                  "Some people think that having honest elections makes a lot of difference in their lives; other people think that it doesn't matter much. honest elections play an important role in deciding whether you and your family are able to make a good living? How important this is-very important, fairly important, not very important or not at all important?",
                  "Very important | Rather important | Not very important | Not at all important",
                  "importance"),
    },
    "WVS-7 (2017-2022)": {
        "Q71":  ("TRUST AND INSTITUTIONS",
                 "I am unsure whether to believe most politicians",
                 "Agree | Neither Agree nor Disagree | Disagree", "agreement"),
        "Q72":  ("TRUST AND INSTITUTIONS",
                 "I am usually cautious about trusting politicians",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q73":  ("TRUST AND INSTITUTIONS",
                 "In general, politicians are open about their decisions",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q74":  ("TRUST AND INSTITUTIONS",
                 "In general, the government usually does the right thing",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q75":  ("TRUST AND INSTITUTIONS",
                 "Information provided by the government is generally unreliable",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q76":  ("TRUST AND INSTITUTIONS",
                 "It is best to be cautious about trusting the government",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q77":  ("TRUST AND INSTITUTIONS",
                 "Most politicians are honest and truthful",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q78":  ("TRUST AND INSTITUTIONS",
                 "The government usually carries out its duties poorly",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q79":  ("TRUST AND INSTITUTIONS",
                 "The government usually acts in its own interests",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q80":  ("TRUST AND INSTITUTIONS",
                 "The government wants to do its best to serve the country",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q81":  ("TRUST AND INSTITUTIONS",
                 "The government is generally free of corruption",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q82":  ("TRUST AND INSTITUTIONS",
                 "The government's work is open and transparent",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q83":  ("TRUST AND INSTITUTIONS",
                 "People in the government often show poor judgement",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q84":  ("TRUST AND INSTITUTIONS",
                 "Politicians are often incompetent and ineffective",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q85":  ("TRUST AND INSTITUTIONS",
                 "Politicians often put country above their personal interests",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q71F": ("TRUST AND INSTITUTIONS",
                 "To what extent is the government working to crackdown on corruption and root out bribes?",
                 "Not at all | To a small extent | To a medium extent | To a large extent", "extent"),
        "Q89":  ("TRUST AND INSTITUTIONS",
                 "Parliament usually acts in its own interests",
                 "Agree | Neither Agree nor Disagree | Disagree", "agreement"),
        "Q90":  ("TRUST AND INSTITUTIONS",
                 "Parliament is generally free of corruption",
                 "Agree | Neither Agree nor Disagree | Disagree", "agreement"),
        "Q91":  ("TRUST AND INSTITUTIONS",
                 "Parliament's work is open and transparent",
                 "Agree | Neither Agree nor Disagree | Disagree", "agreement"),
        "Q94":  ("TRUST AND INSTITUTIONS",
                 "The UN wants to do its best to serve the world",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q222": ("POLITICS AND SOCIETY",
                 "To what extent \"Citizens must support the government's decisions even if they disagree with them\"",
                 "I strongly disagree | I disagree | I agree | I strongly agree", "agreement"),
        "Q223": ("POLITICS AND SOCIETY",
                 "\"Political reform should be introduced little by little instead of all at once\"",
                 "I strongly disagree | I disagree | I agree | I strongly agree", "agreement"),
        "Q224": ("TRUST AND INSTITUTIONS",
                 "The government usually has good intentions",
                 "Disagree strongly | Disagree | Neither agree nor disagree | Agree | Agree strongly", "agreement"),
        "Q28":  ("FAMILY AND GENDER",
                 "When a mother works for pay, the children suffer",
                 "Strongly agree | Agree | Disagree | Strongly disagree", "agreement"),
        "Q30":  ("TRUST AND INSTITUTIONS",
                 "How many world leaders from a list do you generally trust?",
                 "Trust nobody | Trust one leader | Trust two leaders | Trust three leaders | Trust all four leaders",
                 "trust"),
        "Q32":  ("FAMILY AND GENDER",
                 "If a woman earns more money than her husband, it's almost certain to cause problems",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q33":  ("FAMILY AND GENDER",
                 "Homosexual couples are as good parents as other couples",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q34":  ("FAMILY AND GENDER",
                 "It is a duty towards society to have children",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q35":  ("FAMILY AND GENDER",
                 "Adult children have the duty to provide long-term care for their parents",
                 "Agree strongly | Agree | Neither agree nor disagree | Disagree | Disagree strongly", "agreement"),
        "Q46":  ("VALUES AND ETHICS",
                 "Less importance placed on work in our lives",
                 "Good | Don\u2019t mind | Bad", "evaluation"),
        "Q47":  ("SCIENCE AND TECHNOLOGY",
                 "More emphasis on the development of technology",
                 "Good | Don\u2019t mind | Bad", "evaluation"),
        "Q48":  ("VALUES AND ETHICS",
                 "Greater respect for authority",
                 "Good | Don\u2019t mind | Bad", "evaluation"),
        "Q49":  ("PERSONAL WELLBEING",
                 "Taking all things together, you are",
                 "Very happy | Rather happy | Not very happy | Not at all happy",
                 "happiness"),
        "Q50":  ("PERSONAL WELLBEING",
                 "Comparing your standard of living with your parents' standard of living when they were about your age, you are better off, worse off or about the same?",
                 "Better off | Worse off | Or about the same",
                 "comparison"),
        "Q160": ("SCIENCE AND TECHNOLOGY",
                 "Science and technology are making our lives healthier, easier, and more comfortable.",
                 "Completely disagree | Completely agree", "agreement"),
        "Q161": ("SCIENCE AND TECHNOLOGY",
                 "Because of science and technology, there will be more opportunities for the next generation.",
                 "Completely disagree | Completely agree", "agreement"),
        "Q162": ("SCIENCE AND TECHNOLOGY",
                 "We depend too much on science and not enough on faith.",
                 "Completely disagree | Completely agree", "agreement"),
        "Q163": ("SCIENCE AND TECHNOLOGY",
                 "One of the bad effects of science is that it breaks down people's ideas of right and wrong.",
                 "Completely disagree | Completely agree", "agreement"),
        "Q164": ("SCIENCE AND TECHNOLOGY",
                 "It is not important for me to know about science in my daily life.",
                 "Completely disagree | Completely agree", "agreement"),
        "Q165": ("RELIGION",
                 "Whenever science and religion conflict, religion is always right",
                 "Strongly agree | Agree | Disagree | Strongly disagree", "agreement"),
        "Q166": ("RELIGION",
                 "The only acceptable religion is my religion.",
                 "Strongly agree | Agree | Disagree | Strongly disagree", "agreement"),
        "Q173": ("RELIGION",
                 "With which one of the following statements most? The basic meaning of religion is:",
                 "To follow religious norms and ceremonies | To do good to other people",
                 "agreement"),
        "Q174": ("VALUES AND ETHICS",
                 "How much with the statement that nowadays one often has trouble deciding which moral rules are the right ones to follow?",
                 "Completely agree | Completely disagree", "agreement"),
        "Q182": ("FAMILY AND GENDER",
                 "How strongly you agree or disagree with the following statement: \"on the whole, women are less corrupt than men\"?",
                 "Strongly agree | Agree | Disagree | Strongly disagree", "agreement"),
        "Q235": ("POLITICS AND SOCIETY",
                 "How much the political system in a country allows people like you to have a say in what the government does?",
                 "A great deal | A lot | Some | Very little | Not at all", "extent"),
        "Q250": ("POLITICS AND SOCIETY",
                 "How important is having honest elections in general?",
                 "Very Important | Rather Important | Not Very Important | Not at All Important", "importance"),
    },
}


def build_wvs567_records():
    records = []
    for wave_label, variables in WVS567_QUESTIONS.items():
        src_short = wave_label.split(" ")[0]  # e.g. "WVS-5"
        for var_code, (subject, question, options, scale_type) in variables.items():
            resolved = SUBJECT_OVERRIDES_BY_SOURCE.get(
                (src_short, var_code),
                SUBJECT_OVERRIDES.get(var_code, subject)
            )
            records.append({
                "source":            wave_label,
                "question_id":       var_code,
                "subject":           resolved,
                "question":          question,
                "scale_type":        scale_type,
                "answering_options": options,
            })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



# ===========================================================================
# Stage 2 — Curate: heuristic filtering, standardization, and overrides
# (from curate_questions.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# MANUAL_DROPS
# Each entry is a question_id that passed the heuristic filter but should
# not appear in the final dataset.
# ---------------------------------------------------------------------------

MANUAL_DROPS = {
    # EVS items: near-duplicates or geographic confounds
    "v37",   # trust people of another nationality (geographic confound; v36 trust another religion retained)
    "v60",   # believe in heaven (dropped; v59 hell retained)
    "v126",  # confidence in healthcare system (country-specific institution)
    "v127",  # confidence in justice system (country-specific institution)

    # WVS-4: items dropped due to non-ordinal or near-duplicate structures
    "V149",  # confidence in the press (dropped; TV V150 retained)
    "V152",  # confidence in the police (dropped; labour unions V151 retained)

    # WVS-4: items not present in reference dataset (dropped)
    "V83",   # work vs leisure scale (comparison, not in reference)
    "V105",  # following instructions at work (not in reference)
    "V109",  # child needs father and mother (not in reference)
    "V127",  # less emphasis on money (evaluation; not in reference as WVS-4)
    "V128",  # less importance on work (covered by EVS v113)
    "V129",  # more emphasis on technology (covered by EVS Q47)
    "V130",  # greater respect for authority (covered by EVS v114)
    "V131",  # more emphasis on family life (not in reference as WVS-4)
    "V161",  # confidence in NATO (not in reference)
    "V162",  # confidence in the United Nations (not in reference)
    "V167",  # democratic political system evaluation (not in reference)
    "V172",  # democracy has problems but better (not in reference)

    # WVS-4: religious beliefs (duplicated by EVS v57-v61)
    "V191",  # believe in God (duplicate of EVS v57)
    "V192",  # believe in life after death (duplicate of EVS v58)
    "V193",  # believe in people have a soul (not in reference)
    "V194",  # believe in hell (duplicate of EVS v59)
    "V195",  # believe in heaven (duplicate of EVS v60, which is also dropped)

    # WVS-4: justifiability items (duplicated by EVS v149-v163)
    "V204",  # claiming government benefits (covered by EVS v149)
    "V205",  # avoiding a fare (covered by EVS v159)
    "V206",  # cheating on taxes (covered by EVS v150)
    "V207",  # accepting a bribe (covered by EVS v152)
    "V208",  # homosexuality (covered by EVS v153)
    "V209",  # prostitution (covered by EVS v160)
    "V210",  # abortion (covered by EVS v154)
    "V211",  # divorce (covered by EVS v155)
    "V212",  # euthanasia (covered by EVS v156)
    "V213",  # suicide (covered by EVS v157)

    # WVS-5: world-level problem severity (country-level versions retained)
    "V227",  # poverty world (V245 country retained)
    "V228",  # discrimination world (V246 country retained)
    "V229",  # sanitation world (V247 country retained)
    "V230",  # education world (V248 country retained)
    "V231",  # environmental pollution world (V249 country retained)
}

# ---------------------------------------------------------------------------
# MANUAL_KEEPS
# Overrides both MANUAL_DROPS and heuristic filtering for specific
# (source_short, question_id) pairs.  Source short is the wave prefix
# before the space-parenthesis, e.g. "WVS-5" from "WVS-5 (2005-2006)".
#
# Needed because MANUAL_DROPS operates on question_id alone, which causes
# cross-wave collisions: e.g. V207-V213 are justifiability items in WVS-4
# but life-goal / citizenship items in WVS-5.
# ---------------------------------------------------------------------------

MANUAL_KEEPS = {
    ("WVS-5", "V105"),  # Would most people try to take advantage of you
                        # (WVS-4 V105 is "following instructions at work")
    ("WVS-5", "V207"),  # Making one's parents proud should be a main goal
                        # (WVS-4 V207 is bribery justifiability)
    ("WVS-5", "V208"),  # People should seek to be themselves
                        # (WVS-4 V208 is homosexuality justifiability)
    ("WVS-5", "V209"),  # People should make effort to live up to friends
                        # (WVS-4 V209 is prostitution justifiability)
    ("WVS-5", "V210"),  # People should decide their goals by themselves
                        # (WVS-4 V210 is abortion justifiability)
    ("WVS-5", "V212"),  # Citizenship: having ancestors from the country
                        # (WVS-4 V212 is euthanasia justifiability)
    ("WVS-5", "V213"),  # Citizenship: abiding by the country's laws
                        # (WVS-4 V213 is suicide justifiability)
    ("WVS-6", "V210"),  # How important is self-expression?
                        # (WVS-4 V210 is abortion justifiability)
    ("WVS-6", "V213"),  # Honest elections importance
                        # (WVS-4 V213 is suicide justifiability)
    ("WVS-7", "Q50"),   # Standard of living vs parents' generation
                        # (heuristic drops it because of "your age" match)
}

# ---------------------------------------------------------------------------
# Heuristic filtering criteria
# ---------------------------------------------------------------------------

PERSONAL_BIOGRAPHICAL_KEYWORDS = [
    "how old are you", "your age", "your education", "highest level of education",
    "your income", "your occupation", "were you born", "country of birth",
    "marital status", "how many children", "number of children",
    "your nationality", "ethnic group", "your religion", "denomination",
]

SURVEY_ADMIN_PATTERNS = [
    r"(interviewer|enumerator)\s+(instruction|note|code|check)",
    r"go to question",
    r"skip to",
    r"if yes, continue",
    r"only ask if",
    r"routing",
    r"filter question",
    r"show card",
    r"read out",
]

PERSONAL_EXPERIENCE_PHRASES = [
    "have you ever",
    "in the past year",
    "in the past 12 months",
    "do you personally",
    "have you personally",
    "did you personally",
    "last time you",
    "last year",
]

LOCATION_SPECIFIC_PATTERNS = [
    r"\b(mr|mrs|dr)\.\s+[A-Z][a-z]+",
    r"named politician",
    r"named party",
]


def is_personal_biographical(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in PERSONAL_BIOGRAPHICAL_KEYWORDS)


def is_survey_admin(question: str) -> bool:
    q_lower = question.lower()
    return any(re.search(pat, q_lower) for pat in SURVEY_ADMIN_PATTERNS)


def is_personal_experience(question: str) -> bool:
    q_lower = question.lower()
    return any(phrase in q_lower for phrase in PERSONAL_EXPERIENCE_PHRASES)


def is_non_ordinal(options: str, scale_type: str) -> bool:
    """
    Drop questions with non-ordinal answer structures (unranked category lists).
    Keep 10-point bipolar opinion scales of the form "1: X | 10: Y".
    """
    if scale_type == "opinion":
        return False
    if scale_type == "justifiability":
        return False
    parts = [p.strip() for p in options.split("|")]
    if len(parts) < 2:
        return True
    try:
        nums = []
        for part in parts:
            if ":" in part:
                n = int(part.split(":")[0].strip())
                nums.append(n)
        if len(nums) >= 2:
            return False
    except ValueError:
        pass
    return False


def heuristic_filter(row: dict) -> tuple[bool, str]:
    """
    Returns (should_drop, reason) for a given row.
    """
    question = row["question"].strip()
    options  = row["answering_options"].strip()
    scale    = row.get("scale_type", "").strip()

    if is_personal_biographical(question):
        return True, "personal_biographical"
    if is_survey_admin(question):
        return True, "survey_administration"
    if is_personal_experience(question):
        return True, "personal_experience"
    if is_non_ordinal(options, scale):
        return True, "non_ordinal_structure"

    return False, ""


# ---------------------------------------------------------------------------
# Scale standardization
# ---------------------------------------------------------------------------

def parse_endpoint_labels(options_str: str) -> tuple[str, str] | None:
    """Extract (left_label, right_label) from '1: X | 10: Y' format."""
    parts = [p.strip() for p in options_str.split("|")]
    if len(parts) < 2:
        return None
    first = parts[0].split(":", 1)
    last  = parts[-1].split(":", 1)
    if len(first) < 2 or len(last) < 2:
        return None
    try:
        first_n = int(first[0].strip())
        last_n  = int(last[0].strip())
    except ValueError:
        return None
    if last_n < 5:
        return None
    return first[1].strip(), last[1].strip()


def expand_10point_bipolar(left: str, right: str) -> list[str]:
    """Convert a 10-point bipolar scale to a 5-point labeled scale."""
    return [
        f"Strongly {left.title()}",
        left.title(),
        "Neutral",
        right.title(),
        f"Strongly {right.title()}",
    ]


STANDARD_SCALES = {
    "agreement_4":  ["Strongly Agree", "Agree", "Disagree", "Strongly Disagree"],
    "confidence_4": ["A Great Deal", "Quite a Lot", "Not Very Much", "None at All"],
    "importance_4": ["Very Important", "Rather Important",
                     "Not Very Important", "Not at All Important"],
    "goodbad_4":    ["Very good", "Fairly good", "Fairly bad", "Very bad"],
    "justif_5":     ["Never Justifiable", "Rarely Justifiable", "Sometimes Justifiable",
                     "Often Justifiable", "Always Justifiable"],
    "frequency_4":  ["Always", "Often", "Rarely", "Never"],
}


def standardize_options(scale_type: str, options_str: str) -> list[str]:
    """
    Convert original answering options to standardized list.
    10-point bipolar scales are reduced to 5-point.
    Other scales are returned as label lists.
    """
    endpoints = parse_endpoint_labels(options_str)
    if endpoints:
        left, right = endpoints
        if scale_type == "justifiability":
            return STANDARD_SCALES["justif_5"]
        return expand_10point_bipolar(left, right)

    opts_lower = options_str.lower()

    if scale_type == "agreement":
        if "strongly agree" in opts_lower and "strongly disagree" in opts_lower:
            return STANDARD_SCALES["agreement_4"]
    if scale_type == "confidence":
        if "great deal" in opts_lower:
            return STANDARD_SCALES["confidence_4"]
    if scale_type == "importance":
        if "very important" in opts_lower and "not at all important" in opts_lower:
            return STANDARD_SCALES["importance_4"]
        if "rather important" in opts_lower:
            return STANDARD_SCALES["importance_4"]
    if scale_type == "evaluation":
        if "very good" in opts_lower and "very bad" in opts_lower:
            return STANDARD_SCALES["goodbad_4"]
    if scale_type == "justifiability":
        return STANDARD_SCALES["justif_5"]
    if scale_type == "frequency":
        if "always" in opts_lower and "never" in opts_lower:
            return STANDARD_SCALES["frequency_4"]

    parts = [p.strip() for p in options_str.split("|")]
    labels = []
    for part in parts:
        if ":" in part:
            label = part.split(":", 1)[1].strip()
        else:
            label = part.strip()
        if label:
            labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# Scale type to question type mapping
# ---------------------------------------------------------------------------

SCALE_TO_QUESTION_TYPE = {
    "agreement":      "Agreement",
    "importance":     "Importance",
    "confidence":     "HowWell",
    "justifiability": "Acceptance",
    "evaluation":     "GoodOrBad",
    "trust":          "HowWell",
    "belief":         "Agreement",
    "adequacy":       "HowWell",
    "satisfaction":   "PositiveNegative",
    "frequency":      "Frequency",
    "priority":       "Importance",
    "opinion":        "Agreement",
    "interest":       "Concern",
    "extent":         "Quantity",
    "comparison":     "BetterOrWorse",
    "happiness":      "PositiveNegative",
    "religiosity":    "Quantity",
    "fairness":       "Agreement",
}

QUESTION_TYPE_OVERRIDES = {
    "V187":  "HowWell",
    "V188":  "HowWell",
    "V189":  "HowWell",
    "V190":  "HowWell",   # adequacy of churches answering social problems
    "v113":  "GoodOrBad",
    "v114":  "GoodOrBad",
    "V122":  "Importance",
    "V123":  "Importance",
    "V124":  "Importance",
    "V125":  "Importance",
    # WVS-4: societal goals second set and world problems (single-item OAOs, opinion scale)
    "V134":  "Importance",
    "V135":  "Importance",
    "V136":  "Importance",
    "V137":  "Importance",
    "V138":  "Importance",
    "V139":  "Importance",
    "V140":  "Importance",
    "V141":  "Importance",
    "V142":  "Importance",
    "V143":  "Importance",
    "V144":  "Importance",
    # WVS-5: country-level problem severity (single-item OAOs, opinion scale)
    "V245":  "Importance",
    "V246":  "Importance",
    "V247":  "Importance",
    "V248":  "Importance",
    "V249":  "Importance",
}

# Source-specific question_type overrides: (source_short, question_id) -> question_type
# Used when the same question_id appears in multiple waves needing different types.
QUESTION_TYPE_OVERRIDES_BY_SOURCE = {
    ("WVS-4", "V120"): "Importance",   # national goal (WVS-5 V120 is confidence → HowWell)
    ("WVS-4", "V121"): "Importance",   # national goal (WVS-5 V121 is confidence → HowWell)
}


def get_question_type(question_id: str, scale_type: str, source_short: str = "") -> str:
    by_source = QUESTION_TYPE_OVERRIDES_BY_SOURCE.get((source_short, question_id))
    if by_source:
        return by_source
    if question_id in QUESTION_TYPE_OVERRIDES:
        return QUESTION_TYPE_OVERRIDES[question_id]
    return SCALE_TO_QUESTION_TYPE.get(scale_type, "Agreement")


# ---------------------------------------------------------------------------
# v204 expansion: EVS forced-choice environment vs. growth split into two
# Importance items so each pole can be evaluated independently.
# ---------------------------------------------------------------------------

V204_EXPANDED = [
    {
        "source":                    "EVS-v204",
        "question_id":               "v204a",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to prioritize protecting the environment?",
        "question_type":             "Importance",
        "answering_options":         str(["Very important", "Somewhat important",
                                         "Not too important", "Not at all important"]),
        "original_answering_options": "Protecting the Environment Should Be Given Priority",
    },
    {
        "source":                    "EVS-v204",
        "question_id":               "v204b",
        "subject":                   "ENVIRONMENT",
        "question":                  "How important is it to prioritize economic growth?",
        "question_type":             "Importance",
        "answering_options":         str(["Very important", "Somewhat important",
                                         "Not too important", "Not at all important"]),
        "original_answering_options": "Economic Growth Should Be the Top Priority",
    },
]


# ---------------------------------------------------------------------------
# Question text cleaning
# ---------------------------------------------------------------------------

BATTERY_INTRO_PATTERNS = [
    (r"^for each of the following[^,]*,\s*(indicate\s+)?how important it is[^:]*:\s*", "How important is {item}?"),
    (r"^please tell me how much confidence you have in\s*", "How confident are you in {item}?"),
    (r"^for each of the following[^,]*,\s*", ""),
    (r"^i am going to name a number of[^.]+\.\s*", ""),
    (r"^now i am going to read out some[^.]+\.\s*", ""),
]


def clean_question_text(question: str) -> str:
    """
    Remove battery intro phrasing from matrix-derived questions.
    Ensures first character is capitalised.
    """
    q = question.strip()
    for pattern, replacement in BATTERY_INTRO_PATTERNS:
        m = re.match(pattern, q, re.IGNORECASE)
        if m:
            if "{item}" in replacement:
                remaining = q[m.end():].rstrip("?").strip()
                q = replacement.format(item=remaining)
            else:
                q = replacement + q[m.end():]
            break
    q = q.strip()
    if q:
        q = q[0].upper() + q[1:]
    return q


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_FIELDNAMES = [
    "source", "question_id", "subject", "question", "question_type",
    "answering_options", "original_answering_options",
]

LOG_FIELDNAMES = ["question_id", "source", "question", "drop_type", "drop_reason"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



# ===========================================================================
# Stage 3 — Tidy: question text normalization
# (from tidy_question_text.py)
# ===========================================================================

# Patterns for residual page numbers / headers inserted mid-text by pdfplumber.
# Adjust if specific headers appear in the extracted PDFs.
PAGE_HEADER_PATTERNS = [
    r"\bPage\s+\d+\b",
    r"\bWVS[\-\s]\d\b",
    r"\bEVS\s+20\d\d\b",
    r"\bQuestionnaire\s+\d+\b",
]


def remove_page_artifacts(text: str) -> str:
    for pat in PAGE_HEADER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


def fix_hyphenation(text: str) -> str:
    """Rejoin words split across lines by hyphenation during PDF extraction."""
    return re.sub(r"-\s*\n\s*", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences (including newlines) to a single space."""
    return re.sub(r"\s+", " ", text).strip()


_QUOTE_MAP = str.maketrans({
    "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'",
    "\u2032": "'", "\u00b4": "'",
    "\u2013": "-", "\u2014": "-",
})


def normalize_quotes(text: str) -> str:
    return text.translate(_QUOTE_MAP)


def tidy(text: str) -> str:
    text = fix_hyphenation(text)
    text = remove_page_artifacts(text)
    text = normalize_quotes(text)
    text = normalize_whitespace(text)
    if text:
        text = text[0].upper() + text[1:]
    return text

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Extract ---
    print("Verifying PDFs with pdfplumber...")
    missing = []
    for label, path in PDF_PATHS.items():
        ok = verify_pdf(label, path)
        if not ok:
            missing.append(label)

    if missing:
        print(f"\nERROR: Missing or unreadable PDFs: {missing}", file=sys.stderr)
        print("Place all required PDFs in the raw_pdfs/ directory and re-run.",
              file=sys.stderr)
        sys.exit(1)

    print("\nBuilding WVS-4 records...")
    wvs4 = build_wvs4_records()
    print(f"  WVS-4: {len(wvs4)} records")

    print("Building WVS-5/6/7 records...")
    wvs567 = build_wvs567_records()
    w5 = sum(1 for r in wvs567 if "WVS-5" in r["source"])
    w6 = sum(1 for r in wvs567 if "WVS-6" in r["source"])
    w7 = sum(1 for r in wvs567 if "WVS-7" in r["source"])
    print(f"  WVS-5: {w5} | WVS-6: {w6} | WVS-7: {w7}")

    print("Building EVS records...")
    evs = build_evs_records()
    print(f"  EVS: {len(evs)} records")

    raw_rows = wvs4 + wvs567 + evs
    print(f"\nTotal extracted: {len(raw_rows)} records")

    # --- Stage 2: Curate ---
    print(f"\nCuration input rows: {len(raw_rows)}")
    output_rows = []
    log_rows = []
    heuristic_dropped = 0
    manual_dropped = 0

    for row in raw_rows:
        qid = row["question_id"].strip()
        source_short = row["source"].split(" (")[0].strip()
        is_kept = (source_short, qid) in MANUAL_KEEPS

        drop, reason = heuristic_filter(row)
        if drop and not is_kept:
            log_rows.append({
                "question_id": qid,
                "source":      row["source"],
                "question":    row["question"][:100],
                "drop_type":   "heuristic",
                "drop_reason": reason,
            })
            heuristic_dropped += 1
            continue

        if qid in MANUAL_DROPS and not is_kept:
            log_rows.append({
                "question_id": qid,
                "source":      row["source"],
                "question":    row["question"][:100],
                "drop_type":   "manual",
                "drop_reason": "manual_review",
            })
            manual_dropped += 1
            continue

        # Skip EVS v204 here; replaced by V204_EXPANDED below
        if qid == "v204":
            continue

        scale_type  = row["scale_type"].strip()
        options_str = row["answering_options"].strip()

        std_options   = standardize_options(scale_type, options_str)
        question_type = get_question_type(qid, scale_type, source_short)
        question_text = clean_question_text(row["question"])

        output_rows.append({
            "source":                    row["source"],
            "question_id":               qid,
            "subject":                   row["subject"],
            "question":                  question_text,
            "question_type":             question_type,
            "answering_options":         str(std_options),
            "original_answering_options": options_str,
        })

    print(f"  After heuristic filter: dropped {heuristic_dropped}")
    print(f"  After manual drops:     dropped {manual_dropped}")
    print(f"  Subtotal before v204 expansion: {len(output_rows)}")

    output_rows.extend(V204_EXPANDED)
    print(f"  Added 2 EVS-v204 expanded items (v204a, v204b)")
    print(f"Curated rows: {len(output_rows)}")

    # Write curation log
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Curation log: {LOG_PATH} ({len(log_rows)} dropped rows logged)")

    qt_counts = Counter(r["question_type"] for r in output_rows)
    src_counts = Counter(r["source"].split(" (")[0].split("-v")[0] for r in output_rows)
    print("\nQuestion types:", dict(qt_counts.most_common()))
    print("Sources:", dict(src_counts.most_common()))

    # --- Stage 3: Tidy ---
    processed = 0
    changed = 0
    unchanged = 0

    for row in output_rows:
        original = row["question"]
        tidied = tidy(original)
        row["question"] = tidied
        processed += 1
        if tidied != original:
            changed += 1
        else:
            unchanged += 1

    print(f"\nTidy: rows processed: {processed}")
    print(f"      rows changed:   {changed}")
    print(f"      rows unchanged: {unchanged}")

    # --- Write output ---
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nOutput written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
