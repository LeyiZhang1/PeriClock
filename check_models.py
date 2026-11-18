"""Inspect all six trusted external models without fitting or writing artifacts."""
import argparse
import sys
import numpy as np
from periclock import MODEL_FILES, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="Downloaded model directory; overrides PERICLOCK_MODEL_DIR")
    args = parser.parse_args(argv)
    try:
        for name in MODEL_FILES:
            b = load_model(name, model_dir=args.model_dir)
            print(f"{name}: {b['kind']}, n_features={len(b['feature_names'])}")
            if b["kind"] == "single_cell":
                artifact = b["artifact"]
                # Inspect the actual saved selectors; no removed public helper is needed.
                vt_names = np.asarray(artifact["expr_cols"], dtype=object)[
                    artifact["vt"].get_support()
                ]
                cols = vt_names[artifact["skb"].get_support()]
                print(f"  Recovered pre-scaled expression features: {len(cols)}")
    except FileNotFoundError as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
