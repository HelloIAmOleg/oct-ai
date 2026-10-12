import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # the deterministic trainer didn't work without it in jupyter notebook, but it's not needed in scripts apparently

from octai.inference import get_crossval_agg_df

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--base-run-dir", required = True)
    parser.add_argument("--csv-template", required = True)
    parser.add_argument("--image-root", required = True)
    parser.add_argument("--output-csv", required = True)
    parser.add_argument("--splits-num", type = int, default = 5)
    parser.add_argument("--v-num", type = int, default = 0)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()

def main(args):
    pred_df = get_crossval_agg_df(args.base_run_dir, args.csv_template, args.image_root,
                                  args.v_num, args.splits_num, not args.cpu)
    
    pred_df.to_csv(args.output_csv, index = False)

if __name__ == "__main__":
    args = parse_args()
    main(args)