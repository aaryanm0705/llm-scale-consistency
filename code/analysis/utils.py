"""
Utility functions for APD (Average Pairwise Distance) analysis.

This module provides helper functions for:
- CSV validation and loading
- APD calculation
- Data processing
"""

import math
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Optional


# Required columns for input CSV files
REQUIRED_COLUMNS = ['model', 'question_id', 'answer_var_id', 'selected_answer', 'answer_score']

# Alternative column names that map to required columns
COLUMN_ALIASES = {
    'answer_letter': 'selected_answer',
}


def validate_csv(filepath: str) -> Tuple[bool, Optional[str]]:
    """
    Check if CSV file exists and has required columns.
    Supports alternative column names (e.g., 'answer_letter' for 'selected_answer').

    Args:
        filepath: Path to the CSV file

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid
    """
    path = Path(filepath)

    # Check if file exists
    if not path.exists():
        return False, f"File not found: {filepath}"

    # Check if it's a file
    if not path.is_file():
        return False, f"Not a file: {filepath}"

    # Try to read the CSV header
    try:
        df = pd.read_csv(filepath, nrows=0)
    except Exception as e:
        return False, f"Failed to read CSV: {e}"

    # Check for required columns (accounting for aliases)
    columns = set(df.columns)
    resolved = set()
    for col in REQUIRED_COLUMNS:
        if col in columns:
            resolved.add(col)
        else:
            # Check if any alias maps to this required column
            for alias, target in COLUMN_ALIASES.items():
                if target == col and alias in columns:
                    resolved.add(col)
                    break

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in resolved]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"

    return True, None


def load_csv(filepath: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load and validate a CSV file.
    Automatically renames alternative column names to standard names.

    Args:
        filepath: Path to the CSV file
        verbose: If True, print loading information

    Returns:
        pandas DataFrame with the CSV data

    Raises:
        ValueError: If the CSV is invalid
        FileNotFoundError: If the file doesn't exist
    """
    is_valid, error_message = validate_csv(filepath)
    if not is_valid:
        raise ValueError(error_message)

    df = pd.read_csv(filepath)

    # Rename alias columns to standard names
    rename_map = {}
    for alias, target in COLUMN_ALIASES.items():
        if alias in df.columns and target not in df.columns:
            rename_map[alias] = target
    if rename_map:
        df = df.rename(columns=rename_map)
        if verbose:
            for alias, target in rename_map.items():
                print(f"Mapped column '{alias}' -> '{target}'")

    if verbose:
        print(f"Loaded {len(df)} rows from {filepath}")
        print(f"Models found: {df['model'].nunique()}")
        print(f"Questions found: {df['question_id'].nunique()}")

    return df


def calculate_apd(scores: List[float]) -> float:
    """
    Calculate Average Pairwise Distance (APD) for a list of numeric scores.

    APD is the mean absolute pairwise distance across all variant pairs,
    as defined in the thesis (Section 5.5.1):

        APD(q) = (1 / |P|) * sum_{(i,j) in P} |s_i - s_j|

    where P is the set of all unordered variant pairs and s_i, s_j are
    numeric scores calibrated to [-1, +1].

    Args:
        scores: List of numeric scores (e.g., [1.0, 0.333, 1.0, -0.333, 1.0])

    Returns:
        APD value in [0, 2] (0 = perfectly consistent, 2 = maximum inconsistency)
        Returns 0.0 if fewer than 2 valid scores are provided

    Example:
        >>> calculate_apd([1.0, 1.0, 0.333, 1.0, 1.0])
        0.267  # 4 pairs differ by 0.667; 6 pairs differ by 0.0; mean = 0.267
    """
    valid_scores = [s for s in scores if s is not None and not (isinstance(s, float) and math.isnan(s))]

    if len(valid_scores) < 2:
        return 0.0

    pairs = list(combinations(valid_scores, 2))
    if not pairs:
        return 0.0

    return sum(abs(a - b) for a, b in pairs) / len(pairs)


def filter_error_answers(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Filter out invalid rows. Uses 'is_valid' column if present,
    otherwise filters rows where selected_answer is 'ERROR'.

    Args:
        df: DataFrame with 'selected_answer' column (and optionally 'is_valid')
        verbose: If True, print filtering information

    Returns:
        DataFrame with invalid rows removed
    """
    if 'is_valid' in df.columns:
        invalid_count = (~df['is_valid']).sum()
        if verbose and invalid_count > 0:
            print(f"Filtering out {invalid_count} invalid answers (is_valid=False)")
        return df[df['is_valid']].copy()

    error_count = (df['selected_answer'] == 'ERROR').sum()
    if verbose and error_count > 0:
        print(f"Filtering out {error_count} ERROR answers")
    return df[df['selected_answer'] != 'ERROR'].copy()


def check_variant_completeness(df: pd.DataFrame, expected_variants: int = 5,
                                verbose: bool = False) -> dict:
    """
    Check how many questions have the expected number of variants.

    Args:
        df: DataFrame with 'model', 'question_id', 'answer_var_id' columns
        expected_variants: Expected number of variants per question (default 5)
        verbose: If True, print completeness information

    Returns:
        Dictionary with completeness statistics per model
    """
    stats = {}

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        variant_counts = model_df.groupby('question_id')['answer_var_id'].count()

        complete = (variant_counts == expected_variants).sum()
        incomplete = (variant_counts < expected_variants).sum()
        total = len(variant_counts)

        stats[model] = {
            'complete_questions': complete,
            'incomplete_questions': incomplete,
            'total_questions': total,
            'completeness_rate': complete / total if total > 0 else 0
        }

        if verbose and incomplete > 0:
            print(f"Warning: {model} has {incomplete} questions with < {expected_variants} variants")

    return stats


def format_apd_value(value: float, decimals: int = 3) -> str:
    """
    Format APD value to specified decimal places.

    Args:
        value: APD value to format
        decimals: Number of decimal places (default 3)

    Returns:
        Formatted string representation
    """
    return f"{value:.{decimals}f}"
