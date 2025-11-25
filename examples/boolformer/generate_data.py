"""Generate formulas for testing Boolformer environment."""

import sys
import numpy as np
from pathlib import Path

# Add Boolformer src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.formula.generation_formula import generate_random_function as generate_formula
from src.formula.Formula import Formula
from src.ConfigClasses import ConfigFormula
from src.general_functions import pts_generator


def token_to_id(token):
    """Convert token string to ID. 0:<SOS>, 1:<PAD>, 2:~, 3:&, 4:|, 5-14:x1-x10"""
    if token == '~': return 2
    elif token == '&': return 3
    elif token == '|': return 4
    else: return 4 + int(token[1:])  # x1->5, x2->6, ..., x10->14


def generate_formulas(n=100, max_points=512):
    """
    Generate n formulas following Boolformer approach.

    Returns (points_array, polish_exprs) where:
    - points_array: shape (n, 512, 10) - variable combinations where output is 1 (minority class)
                    Values are -1.0 (False), +1.0 (True), or 0.0 (padding)
                    Padded to fixed size for JAX batching/JIT
    - polish_exprs: list of polish expression lists for inspection
    """
    # Use noiseless config (path relative to Boolformer root)
    config_path = str(Path(__file__).parent.parent.parent.parent / "config" / "formula" / "noiseless.py")
    config = ConfigFormula(py_config_path=config_path)
    points_list = []
    polish_exprs = []

    while len(points_list) < n:
        try:
            tree, _ = generate_formula(config)  # Returns (tree, num_variables)
            formula = Formula(config, tree=tree, simplify=True)
            if not formula.is_valid:
                continue

            # Get all 1024 points (each point is [v0, v1, ..., v9, output])
            pts = pts_generator(config, nb_candidates=10)
            evals = formula.evaluate_pts(pts)
            outputs = evals[:, -1].numpy().astype(np.float32)

            # Count outputs
            count_ones = outputs.sum()
            count_zeros = len(outputs) - count_ones

            # If 0 is less frequent, negate and prepend NOT
            if count_zeros < count_ones:
                outputs = 1 - outputs  # Flip all bits
                polish_expr = ['~'] + formula.polish_expr
            else:
                polish_expr = formula.polish_expr

            # Skip if formula is too long
            if len(polish_expr) > 50:
                continue

            # Extract points where output is 1 (the minority class)
            minority_mask = (outputs == 1)
            minority_points = pts[:, :10][minority_mask]  # Take first 10 columns (variable values)

            # Pad to fixed size (512, 10) for JAX batching
            num_points = len(minority_points)
            if num_points > max_points:
                # Shouldn't happen (max is 512 for 10 vars), but handle it
                minority_points = minority_points[:max_points]
                num_points = max_points

            # Convert points from {0, 1} to {-1, +1} so padding can be 0
            minority_points = 2.0 * minority_points - 1.0  # 0 -> -1, 1 -> +1

            # Pad with 0.0 (distinct from -1/+1)
            padded_points = np.zeros((max_points, 10), dtype=np.float32)
            padded_points[:num_points] = minority_points

            points_list.append(padded_points)
            polish_exprs.append(polish_expr)

        except:
            continue

    return np.array(points_list), polish_exprs


if __name__ == '__main__':
    points_array, polish_exprs = generate_formulas(10)
    print(f"Generated {len(points_array)} formulas")
    print(f"Points array shape: {points_array.shape}")
    print(f"Unique values in points: {np.unique(points_array)}")  # Should be [-1, 0, 1]
    print(f"Example 0: formula: {polish_exprs[0]}, non-padding points: {(points_array[0] != 0).any(axis=1).sum()}")
    print(f"Example 1: formula: {polish_exprs[1]}, non-padding points: {(points_array[1] != 0).any(axis=1).sum()}")
    print(f"Example 2: formula: {polish_exprs[2]}, non-padding points: {(points_array[2] != 0).any(axis=1).sum()}")
