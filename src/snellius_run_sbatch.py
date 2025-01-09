from pathlib import Path
from tqdm.auto import tqdm
import openml
import os
import pandas as pd
from utils import SQLHandler
import argparse
from typing import Union
import pandas as pd


class SBatchRunner:
    def __init__(
        self,
        filter_by_id=Union[None, int],
        filter_by_framework=Union[None, str],
        data_dir="automl_data",
        username="smukherjee",
        sbatch_script_dir: str = "sbatch_scripts",
    ) -> None:
        self.sbatch_script_dir = sbatch_script_dir
        self.filter_by_id = filter_by_id
        self.filter_by_framework = filter_by_framework
        self.username = username
        self.main_dir_in_snellius = f"/home/{self.username}/OpenML-auto-automl"
        self.data_dir = Path(f"/home/{self.username}") / data_dir
        self.sbatch_script_dir = Path(self.data_dir) / sbatch_script_dir
        self.sql_handler = SQLHandler(db_path=Path(self.data_dir) / "runs.db")

    def run_sbatch(self, sbatch_path: Path):
        try:
            file_name = Path(sbatch_path).stem
            dataset_id, task_id, framework = file_name.split("_")
            if not self.sql_handler.task_already_run(
                dataset_id=dataset_id, task_id=task_id, framework=framework
            ):
                full_sbatch_path = self.sbatch_script_dir / sbatch_path
                os.system(f"sbatch {full_sbatch_path}")

                self.sql_handler.save_run(
                    dataset_id=dataset_id, task_id=task_id, framework=framework
                )

        except Exception as e:
            print(f"Error running sbatch: {e}")

    def get_dataset_task_and_framework_from_sbatch(self, sbatch_path: Union[str, Path]):
        try:
            file_name = Path(sbatch_path).stem
            dataset_id, task_id, framework = file_name.split("_")
            return dataset_id, task_id, framework
        except Exception as e:
            print(f"Error getting dataset, task, and framework from sbatch: {e}")
            return None, None, None

    def run_all_sbatch(self):
        sbatch_files = os.listdir(self.sbatch_script_dir)
        sbatch_dataframe = pd.DataFrame(columns=["file_path"])
        sbatch_dataframe["file_path"] = sbatch_files

        # check for empty dataframe
        if sbatch_dataframe.empty:
            print("No sbatch files found")
            return

        # add dataset_id, task_id, and framework columns using a lambda function
        (
            sbatch_dataframe["dataset_id"],
            sbatch_dataframe["task_id"],
            sbatch_dataframe["framework"],
        ) = zip(
            *sbatch_dataframe["file_path"].apply(
                lambda x: self.get_dataset_task_and_framework_from_sbatch(sbatch_path=x)
            )
        )
        sbatch_dataframe["dataset_id"] = sbatch_dataframe["dataset_id"].astype(int)
        sbatch_dataframe["task_id"] = sbatch_dataframe["task_id"].astype(int)

        if self.filter_by_id is not None:
            sbatch_dataframe = sbatch_dataframe[
                sbatch_dataframe["dataset_id"] == self.filter_by_id
            ]

        if self.filter_by_framework is not None:
            sbatch_dataframe = sbatch_dataframe[
                sbatch_dataframe["framework"] == self.filter_by_framework
            ]

        print(f"Running {sbatch_dataframe.shape[0]} datasets on Snellius")

        for sbatch_file in tqdm(sbatch_dataframe["file_path"]):
            self.run_sbatch(sbatch_path=sbatch_file)


args = argparse.ArgumentParser()
args.add_argument("--username", type=str, default="smukherjee")
args.add_argument("--filter_by_id", type=int, required=False)
args.add_argument("--filter_by_framework", type=str, required=False)
args.add_argument("--data_dir", type=str, required=False, default ="automl_data")

args = args.parse_args()

sbatch_runner = SBatchRunner(
    filter_by_id=args.filter_by_id,
    filter_by_framework=args.filter_by_framework,
    data_dir=args.data_dir,
    username=args.username,
)
sbatch_runner.run_all_sbatch()