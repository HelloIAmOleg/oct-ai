"""Apply the saved 1D pipelines to the five outer test folds."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'scripts/tuning/artifacts/2026-09-23_NESTED-HPO_1D_fast-axis_NMF'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for fold in range(5):
        df = pd.read_csv(ROOT / f'csv/test_split{fold}.csv')
        with (ARTIFACTS / f'outer{fold}/best_config.json').open() as stream:
            config = json.load(stream)
        with (ARTIFACTS / f'final_outer_cv/outer{fold}/pipeline.pkl').open('rb') as stream:
            pipeline = pickle.load(stream)
        aggregation = config['profile_aggregation']
        if aggregation not in {'mean', 'median'}:
            raise ValueError(f'Unknown profile aggregation: {aggregation}')
        profiles = []
        for relative in df.LOCAL_FILEPATH:
            path = args.image_root / relative
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            profiles.append(image.mean(axis=1) if aggregation == 'mean'
                            else np.median(image, axis=1))
        x = np.stack(profiles)
        probabilities = pipeline.predict_proba(x)
        for column, label in enumerate(pipeline.classes_):
            df[f'{label}_proba'] = probabilities[:, column]
        df['Condition_pred'] = pipeline.predict(x)
        df['outer_idx'] = fold
        results.append(df)
    output = pd.concat(results, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f'Saved {len(output)} predictions to {args.output_csv}')


if __name__ == '__main__':
    main()
