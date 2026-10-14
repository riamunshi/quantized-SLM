"""
answer_extraction.py
--------------------
Utility functions for extracting final answers from model outputs
and ground truth solutions for GSM8K and MATH datasets.

Used by all inference scripts in this project.
"""

import re


def extract_gsm8k_ground_truth(answer_str):
    """
    Extract final numeric answer from GSM8K gold solution.
    GSM8K answers end with #### followed by the number.

    Args:
        answer_str: raw answer string from GSM8K dataset

    Returns:
        str: extracted answer or None if not found
    """
    match = re.search(r'####\s*([\d,\.\-]+)', answer_str)
    if match:
        return match.group(1).replace(',', '').strip()
    return None


def extract_math_ground_truth(solution_str):
    """
    Extract final answer from MATH gold solution.
    MATH answers are enclosed in \\boxed{}.

    Args:
        solution_str: raw solution string from MATH dataset

    Returns:
        str: extracted answer or None if not found
    """
    matches = re.findall(r'\\boxed\{([^}]+)\}', solution_str)
    if matches:
        return matches[-1].strip()
    # Fallback: last number in solution
    nums = re.findall(r'-?[\d\.]+', solution_str)
    return nums[-1] if nums else None


def extract_prediction(output_str):
    """
    Extract final answer from model output.
    Tries multiple patterns in order of reliability.

    Args:
        output_str: raw model output string

    Returns:
        str: extracted answer or None if not found
    """
    # Pattern 1: \\boxed{answer} — preferred format
    matches = re.findall(r'\\boxed\{([^}]+)\}', output_str)
    if matches:
        return matches[-1].strip()

    # Pattern 2: #### number — GSM8K format
    m = re.search(r'####\s*([\d,\.\-]+)', output_str)
    if m:
        return m.group(1).replace(',', '').strip()

    # Pattern 3: last number in output — fallback
    nums = re.findall(r'\b([\d,\.]+)\b', output_str)
    if nums:
        return nums[-1].replace(',', '').strip()

    return None


def is_correct(prediction, ground_truth):
    """
    Check if prediction matches ground truth.
    Uses string comparison after stripping whitespace.

    Args:
        prediction: extracted prediction string
        ground_truth: extracted ground truth string

    Returns:
        bool: True if correct, False otherwise
    """
    if prediction is None or ground_truth is None:
        return False
    return prediction.strip() == ground_truth.strip()
