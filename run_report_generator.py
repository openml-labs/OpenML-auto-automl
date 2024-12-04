from report_generator import (
    run_report_script_for_all_datasets,
    run_for_single_dataset,
    ResultCollector,
)
from utils import find_max_existing_dataset_id
from pathlib import Path
import matplotlib

matplotlib.use("agg")


# Directory for generated reports
GENERATED_REPORTS_DIR = Path("./generated_reports")
GENERATED_REPORTS_DIR.mkdir(exist_ok=True)

max_existing_dataset_id: int = find_max_existing_dataset_id()

collector = ResultCollector(path="./data/results/*")
collector()
# run_report_script_for_all_datasets(
#     GENERATED_REPORTS_DIR=GENERATED_REPORTS_DIR,
#     max_existing_dataset_id=max_existing_dataset_id,
#     collector=None,
# )

run_for_single_dataset(dataset_id=4, collector=collector, GENERATED_REPORTS_DIR=GENERATED_REPORTS_DIR)