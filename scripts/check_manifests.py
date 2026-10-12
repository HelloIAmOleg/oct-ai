"""Validate the completed public fold manifest without accessing images."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "csv"
FIELDS = ["full_id", "measurement_id", "Condition", "slice_num", "LOCAL_FILEPATH"]
LABELS = {"Norm", "Early VLS", "VLS"}


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == FIELDS, (name, reader.fieldnames)
        rows = list(reader)
    for row in rows:
        assert row["Condition"] in LABELS, name
        expected = (f"{row['Condition']}/{row['full_id']}__{row['measurement_id']}/"
                    f"slice_{int(row['slice_num']):03d}.png")
        assert row["LOCAL_FILEPATH"] == expected, (name, row["LOCAL_FILEPATH"])
    return rows


aggregate = read("agg_pred_df.csv")
assert len(aggregate) == 4264
patients = {row["full_id"] for row in aggregate}
assert patients == {str(i) for i in range(48)}
measurements = defaultdict(set)
patient_of = {}
label_of = {}
for row in aggregate:
    measurement = row["measurement_id"]
    measurements[measurement].add(row["slice_num"])
    patient_of.setdefault(measurement, row["full_id"])
    label_of.setdefault(measurement, row["Condition"])
    assert patient_of[measurement] == row["full_id"]
    assert label_of[measurement] == row["Condition"]
assert len(measurements) == 164
assert len({row["LOCAL_FILEPATH"] for row in aggregate}) == len(aggregate)
slice_grid = set.union(*measurements.values())
assert len(slice_grid) == 26
assert all(slices == slice_grid for slices in measurements.values())
all_test = []
outer_test_ids = []
for fold in range(5):
    groups = {role: read(f"{role}_split{fold}.csv") for role in ("train", "val", "test")}
    assert sum(map(len, groups.values())) == 4264
    ids = {role: {row["full_id"] for row in rows} for role, rows in groups.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert set.union(*ids.values()) == patients
    all_test.extend(groups["test"])
    outer_test_ids.append(ids["test"])
for left in range(5):
    for right in range(left + 1, 5):
        assert not outer_test_ids[left] & outer_test_ids[right]
assert set.union(*outer_test_ids) == patients
signature = lambda row: tuple(row[field] for field in FIELDS)
assert Counter(map(signature, all_test)) == Counter(map(signature, aggregate))
print("OK: 48 patients, 164 measurements, 26 slices each, 4,264 rows, five disjoint outer folds")
