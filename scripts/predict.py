"""Predict one fold with a saved Lightning checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argparse import ArgumentParser
from octai.inference import get_pred_df

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--v-num", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

def main(args):
    pred_df = get_pred_df(args.run_dir, args.csv, args.image_root,
                          v_num=args.v_num, use_gpu=not args.cpu)
    pred_df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main(parse_args())
