from pathlib import Path
from tqdm.auto import tqdm
import openml
import os
import pandas as pd
from utils import SQLHandler, OpenMLTaskHandler
import argparse
from typing import Union
import pandas as pd


class SBatchRunner:
    def __init__(
        self,
        filter_by_id=Union[None, int],
        filter_by_framework=Union[None, str],
        filter_by_suite=None,
        test_subset: Union[None, int] = None,
        data_dir="automl_data",
        username="smukherjee",
        sbatch_script_dir: str = "sbatch_scripts",
    ) -> None:
        """
        This class is used to run sbatch scripts on Snellius. Arguments are used to filter the sbatch scripts that are to be run. It keeps track of what has been run using an SQL database. 
        If a task does not exist for the database at hand, it tries to create it before trying to run the scripts. Since Snellius does not support running scripts directly, this class generates sbatch scripts and runs them using the sbatch command.

        """
        self.sbatch_script_dir = sbatch_script_dir
        self.filter_by_id = filter_by_id
        self.filter_by_framework = filter_by_framework
        self.filter_by_suite = filter_by_suite
        self.test_subset = test_subset
        self.username = username
        self.main_dir_in_snellius = f"/home/{self.username}/OpenML-auto-automl"
        self.data_dir = Path(f"/home/{self.username}") / data_dir
        self.sbatch_script_dir = Path(self.data_dir) / sbatch_script_dir

        self._initialize()
        self.sql_handler = SQLHandler(db_path=Path(self.data_dir) / "runs.db")
        self.benchmark_suite_ids = {"amlb_training": 293}
        self.fn_to_get_dataset_id = OpenMLTaskHandler().get_dataset_id_from_task_id

    def _make_files(self, folders):
        """Ensure that the necessary directories exist."""
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def _initialize(self):
        """Create the necessary directories."""
        self._make_files([self.data_dir, self.sbatch_script_dir])

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
        """
        Extracts dataset_id, task_id, and framework from the sbatch file name.
        """
        try:
            file_name = Path(sbatch_path).stem
            dataset_id, task_id, framework = file_name.split("_")
            return dataset_id, task_id, framework
        except Exception as e:
            print(f"Error getting dataset, task, and framework from sbatch: {e}")
            return None, None, None

    def get_dataset_ids_for_benchmark_suite(self, suite_name: str = "amlb_training"):
        """Get dataset_ids for a benchmark suite from OpenML."""
        try:
            suite_id = self.benchmark_suite_ids[suite_name]
        except KeyError:
            print(f"Suite {suite_name} not found")
            return None
        tasks_in_suite = openml.study.get_suite(suite_id=suite_id).tasks
        if tasks_in_suite is not None:
            # get dataset_ids from suites if not none
            return [
                self.fn_to_get_dataset_id(task)
                for task in tasks_in_suite
                if self.fn_to_get_dataset_id(task) is not None
            ]
        else:
            return None

    def run_all_sbatch(self):
        """
        Get all sbatch files from the sbatch_script_dir and run them on Snellius, filtering by dataset_id, framework, or suite if necessary. Save the runs in an SQL database.
        """
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

        if self.filter_by_suite is not None:
            dataset_ids = self.get_dataset_ids_for_benchmark_suite(
                suite_name=self.filter_by_suite
            )
            if dataset_ids is not None:
                sbatch_dataframe = sbatch_dataframe[
                    sbatch_dataframe["dataset_id"].isin(dataset_ids)
                ]
            else:
                print(f"Suite {self.filter_by_suite} not found")

        if self.test_subset is not None:
            sbatch_dataframe = sbatch_dataframe.head(self.test_subset)

        print(f"Running {sbatch_dataframe.shape[0]} datasets on Snellius")

        for sbatch_file in tqdm(sbatch_dataframe["file_path"]):
            self.run_sbatch(sbatch_path=sbatch_file)


args = argparse.ArgumentParser()
args.add_argument("--username", type=str, default="smukherjee")
args.add_argument("--filter_by_id", type=int, required=False)
args.add_argument("--filter_by_suite", type=str, required=False)
args.add_argument("--filter_by_framework", type=str, required=False)
args.add_argument("--data_dir", type=str, required=False, default="automl_data")
args.add_argument("--test_subset", type=int, required=False)

args = args.parse_args()

sbatch_runner = SBatchRunner(
    filter_by_id=args.filter_by_id,
    filter_by_suite=args.filter_by_suite,
    filter_by_framework=args.filter_by_framework,
    test_subset=args.test_subset,
    data_dir=args.data_dir,
    username=args.username,
)
sbatch_runner.run_all_sbatch()