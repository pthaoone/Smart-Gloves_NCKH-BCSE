import csv
import json
import re

import h5py


INPUT_FILE = "hand_imu_bigdata.h5"
CSV_FILE = "labels.csv"
JSON_FILE = "labels.json"


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value.item() if hasattr(value, "item") else value


def trial_sort_key(trial_name):
    match = re.search(r"(\d+)$", trial_name)
    return int(match.group(1)) if match else trial_name


def create_labels(input_file=INPUT_FILE, csv_file=CSV_FILE, json_file=JSON_FILE):
    with h5py.File(input_file, "r") as file:
        sampling_rate_hz = int(decode_value(file.attrs.get("sampling_rate_hz", 0)))
        trial_names = sorted(file.keys(), key=trial_sort_key)
        scenarios = sorted({decode_value(file[name].attrs["scenario"]) for name in trial_names})
        scenario_to_id = {scenario: index for index, scenario in enumerate(scenarios)}

        labels = []
        for trial_name in trial_names:
            trial = file[trial_name]
            scenario = decode_value(trial.attrs["scenario"])
            subject_id = decode_value(trial.attrs["subject_id"])
            trial_id = int(trial_name.rsplit("_", 1)[-1])

            labels.append(
                {
                    "trial_name": trial_name,
                    "trial_id": trial_id,
                    "subject_id": subject_id,
                    "scenario": scenario,
                    "label_id": scenario_to_id[scenario],
                    "num_samples": int(trial["time"].shape[0]),
                    "sampling_rate_hz": sampling_rate_hz,
                }
            )

    field_names = list(labels[0].keys()) if labels else []
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(labels)

    output = {
        "source_file": input_file,
        "num_trials": len(labels),
        "label_mapping": scenario_to_id,
        "labels": labels,
    }
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"Đã tạo {len(labels)} nhãn.")
    print(f"CSV:  {csv_file}")
    print(f"JSON: {json_file}")
    print(f"Label mapping: {scenario_to_id}")


if __name__ == "__main__":
    create_labels()