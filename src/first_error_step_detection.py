"""
first_error_step_detection.py
------------------------------
Detects the first arithmetic error in a model's reasoning chain.

Used by cascade analysis notebooks (Priority1_CascadeAnalysis.ipynb)
and all inference scripts that record first_error_step.

Limitations:
- Only detects arithmetic errors of the form: a OP b = c
  where OP is +, -, *, /
- Does not detect logical errors, wrong problem setup, or unit errors
- Cascade rates computed using this detector are lower bounds
"""

import re


def detect_first_error_step(reasoning_text, tolerance=0.01):
    """
    Scan reasoning chain for the first arithmetic error.

    Splits the text into steps by newlines, then searches each step
    for expressions of the form:
        number OP number = number
    where OP is +, -, *, or /

    For each expression found, evaluates the left side and compares
    to the stated result. If the difference exceeds tolerance,
    the step number is returned as the first error step.

    Args:
        reasoning_text: full model output string
        tolerance: acceptable floating point difference (default 0.01)

    Returns:
        int: step number (1-indexed) where first error appears,
             or None if no arithmetic error detected
    """
    steps = re.split(r'\n+', reasoning_text)

    for step_num, step in enumerate(steps, 1):
        # Find all arithmetic expressions: a OP b = c
        matches = re.findall(
            r'([\d\.]+)\s*([\+\-\*\/])\s*([\d\.]+)\s*=\s*([\d\.]+)',
            step
        )
        for a_str, op, b_str, result_str in matches:
            try:
                a      = float(a_str)
                b      = float(b_str)
                result = float(result_str)

                if op == '+':
                    expected = a + b
                elif op == '-':
                    expected = a - b
                elif op == '*':
                    expected = a * b
                elif op == '/':
                    if b == 0:
                        continue
                    expected = a / b
                else:
                    continue

                if abs(expected - result) > tolerance:
                    return step_num

            except (ValueError, ZeroDivisionError):
                continue

    return None


def compute_cascade_rate(results_df):
    """
    Compute cascade rate from a results dataframe.

    Cascade rate = proportion of wrong answers that had a detected
    first error step.

    Args:
        results_df: pandas DataFrame with columns 'correct' and
                    'first_error_step'

    Returns:
        float: cascade rate between 0 and 1
    """
    wrong = results_df[results_df['correct'] == False]
    if len(wrong) == 0:
        return 0.0
    cascaded = wrong[wrong['first_error_step'].notna()]
    return len(cascaded) / len(wrong)
