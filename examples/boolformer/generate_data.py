"""Generate formulas for testing Boolformer environment."""

import os
import sys
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool as ProcessPool

# Add Boolformer src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.formula.generation_formula import generate_random_function as generate_formula
from src.formula.Formula import Formula
from src.ConfigClasses import ConfigFormula
from src.general_functions import pts_generator


def _get_physical_cores():
    """Get number of physical CPU cores (excluding hyperthreads)."""
    try:
        # Linux: count unique physical core IDs
        with open('/proc/cpuinfo') as f:
            cores = set()
            phys_id = None
            for line in f:
                if line.startswith('physical id'):
                    phys_id = line.split(':')[1].strip()
                elif line.startswith('core id') and phys_id is not None:
                    cores.add((phys_id, line.split(':')[1].strip()))
                    phys_id = None
            if cores:
                return len(cores)
    except (FileNotFoundError, PermissionError):
        pass
    # Fallback: os.cpu_count() // 2 (assume SMT)
    total = os.cpu_count() or 1
    return max(1, total // 2)


NUM_PHYSICAL_CORES = _get_physical_cores()


def token_to_id(token):
    """Convert token string to ID. 0:<SOS>, 1:<PAD>, 2:~, 3:&, 4:|, 5-14:x1-x10"""
    if token == '~': return 2
    elif token == '&': return 3
    elif token == '|': return 4
    else: return 4 + int(token[1:])  # x1->5, x2->6, ..., x10->14


def _worker_generate(args):
    """Worker function: generate formulas matching length constraints.

    Each worker generates its quota independently. The main process
    handles final distribution trimming.
    """
    worker_n, num_variables, max_formula_length, target_lengths = args

    config_path = str(Path(__file__).parent.parent.parent.parent / "config" / "formula" / "noiseless.py")
    config = ConfigFormula(py_config_path=config_path)
    config.ACTIVE_VAR = num_variables
    config.DIMENSION_MAX = num_variables

    max_points = 2 ** (num_variables - 1)

    results = []
    attempts = 0
    reject_invalid = 0
    reject_length = 0

    while len(results) < worker_n:
        try:
            attempts += 1
            tree, _ = generate_formula(config)
            formula = Formula(config, tree=tree, simplify=True)
            if not formula.is_valid:
                reject_invalid += 1
                continue

            pts = pts_generator(config, nb_candidates=num_variables)
            evals = formula.evaluate_pts(pts)
            outputs = evals[:, -1].numpy().astype(np.float32)

            count_ones = outputs.sum()
            count_zeros = len(outputs) - count_ones

            if count_zeros < count_ones:
                outputs = 1 - outputs
                polish_expr = ['~'] + formula.polish_expr
            else:
                polish_expr = formula.polish_expr

            formula_length = len(polish_expr)
            if formula_length > max_formula_length:
                reject_length += 1
                continue

            # If target_lengths specified, only keep matching lengths
            if target_lengths is not None and formula_length not in target_lengths:
                reject_length += 1
                continue

            minority_mask = (outputs == 1)
            minority_points = pts[:, :num_variables][minority_mask]

            num_points = len(minority_points)
            if num_points > max_points:
                minority_points = minority_points[:max_points]
                num_points = max_points

            minority_points = 2.0 * minority_points - 1.0

            padded_points = np.zeros((max_points, num_variables), dtype=np.float32)
            padded_points[:num_points] = minority_points

            results.append((padded_points, polish_expr))

        except:
            continue

    return results, attempts, reject_invalid, reject_length


def generate_formulas(n, num_variables, max_formula_length, length_distribution=None):
    """
    Generate n formulas following Boolformer approach, using multiprocessing.

    Args:
        n: Number of formulas to generate
        num_variables: Number of boolean variables
        max_formula_length: Maximum length of generated formulas (in tokens)
        length_distribution: list/tuple where index i is the proportion for length i

    Returns (points_array, polish_exprs)
    """
    gen_start = time.time()
    num_workers = NUM_PHYSICAL_CORES

    # Determine which lengths are needed
    if length_distribution is not None:
        target_lengths = set(i for i, p in enumerate(length_distribution) if p > 0)
        # Compute per-length quotas
        quotas = {i: int(np.ceil(n * p)) for i, p in enumerate(length_distribution) if p > 0}
    else:
        target_lengths = None
        quotas = None

    # Over-generate per worker to account for distribution trimming
    # Each worker generates ceil(n / num_workers) * overshoot_factor
    per_worker = int(np.ceil(n / num_workers)) * 2  # 2x to reduce re-runs
    worker_args = [(per_worker, num_variables, max_formula_length, target_lengths)] * num_workers

    with ProcessPool(num_workers) as pool:
        worker_results = pool.map(_worker_generate, worker_args)

    # Merge results
    all_points = []
    all_exprs = []
    total_attempts = 0
    total_reject_invalid = 0
    total_reject_length = 0

    for results, attempts, reject_invalid, reject_length in worker_results:
        total_attempts += attempts
        total_reject_invalid += reject_invalid
        total_reject_length += reject_length
        for points, expr in results:
            all_points.append(points)
            all_exprs.append(expr)

    # Apply distribution quotas
    if quotas is not None:
        remaining = dict(quotas)
        filtered_points = []
        filtered_exprs = []
        for points, expr in zip(all_points, all_exprs):
            length = len(expr)
            if remaining.get(length, 0) > 0:
                filtered_points.append(points)
                filtered_exprs.append(expr)
                remaining[length] -= 1
                if sum(remaining.values()) == 0:
                    break
        all_points = filtered_points
        all_exprs = filtered_exprs

    # Trim to exactly n
    all_points = all_points[:n]
    all_exprs = all_exprs[:n]

    gen_time = time.time() - gen_start
    accepted = len(all_points)
    reject_bucket = sum(len(r[0]) for r in worker_results) - accepted - 0  # over-generated minus accepted
    print(f"  Formula gen: {accepted}/{total_attempts} accepted ({accepted/max(total_attempts,1):.1%}) "
          f"| {num_workers} workers | {gen_time:.1f}s "
          f"| rejected: {total_reject_invalid} invalid, {total_reject_length} wrong length")

    if accepted < n:
        print(f"  WARNING: only generated {accepted}/{n} formulas")

    return np.array(all_points), all_exprs


if __name__ == '__main__':
    print(f"Physical cores: {NUM_PHYSICAL_CORES}")
    points_array, polish_exprs = generate_formulas(n=10, num_variables=10, max_formula_length=50)
    print(f"Generated {len(points_array)} formulas")
    print(f"Points array shape: {points_array.shape}")
    print(f"Unique values in points: {np.unique(points_array)}")  # Should be [-1, 0, 1]
    print(f"Example 0: formula: {polish_exprs[0]}, non-padding points: {(points_array[0] != 0).any(axis=1).sum()}")
    print(f"Example 1: formula: {polish_exprs[1]}, non-padding points: {(points_array[1] != 0).any(axis=1).sum()}")
    print(f"Example 2: formula: {polish_exprs[2]}, non-padding points: {(points_array[2] != 0).any(axis=1).sum()}")
