
"""Prediction runner. Custom methods return choices; this runner never opens gold."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from common import DATA, PROTOCOL, load_inputs, validate_choices

# Project code directory
ROOT = Path(__file__).resolve().parent

def main():
    total_started = time.perf_counter()

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # ===== Parameters: configure defaults here =====

    # Your prediction implementation
    parser.add_argument(
        '--method',
        type=Path,
        default=ROOT / 'submission.py',
        help='Python file exposing predict(bank, arrays, checkpoint)'
    )

    # Trained model checkpoint directory
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ROOT / 'runs' / 'model_bundle',
        help='Directory containing the trained model assets'
    )

    # Dataset split: train / validation / test / audit
    parser.add_argument(
        '--split',
        choices=('train', 'validation', 'test', 'audit'),
        default='test',
        help='Dataset split to evaluate'
    )

    # Input data directory
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATA,
        help='Directory containing the assignment data'
    )

    # Prediction output file
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'runs' / 'test-predictions.json',
        help='Path to save prediction results'
    )

    # Parse arguments; command-line values can override defaults
    args = parser.parse_args()

    # ===== Load data =====
    tick = time.perf_counter()

    bank, arrays = load_inputs(
        args.split,
        args.data_dir
    )

    # ===== Load prediction implementation =====
    method_path = args.method.resolve()

    if not method_path.is_file():
        raise FileNotFoundError(
            f'Prediction implementation not found: {method_path}'
        )

    sys.path.insert(0, str(method_path.parent))

    spec = importlib.util.spec_from_file_location(
        'student_method',
        method_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f'Cannot load prediction implementation: {method_path}'
        )

    method = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = method
    spec.loader.exec_module(method)

    # ===== Make predictions =====
    choices = method.predict(
        bank,
        arrays,
        args.checkpoint
    )

    # ===== Validate prediction format =====
    ids = [row['id'] for row in bank]

    validate_choices(
        bank,
        ids,
        choices
    )

    # ===== Save prediction results =====
    result = {
        'protocol': PROTOCOL,
        'split': args.split,
        'ids': ids,
        'selected': choices,
        'prediction_seconds': time.perf_counter() - tick
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.output.write_text(
        json.dumps(result, indent=2) + '\n',
        encoding='utf-8'
    )

    print(json.dumps({
        key: value
        for key, value in result.items()
        if key not in ('ids', 'selected')
    }, indent=2))

    print(
        f'Total elapsed time: '
        f'{time.perf_counter() - total_started:.2f} seconds'
    )


if __name__ == '__main__':
    main()
