from pathlib import Path
from tqdm.auto import tqdm
import openml
import os
import pandas as pd
from utils import OpenMLTaskHandler, SQLHandler
import argparse
from report_generator import run_report_script_for_dataset
from typing import Union


class AutoMLRunner:
    def __init__(
        self,
        testing_mode=False,
        use_cache=True,
        run_mode="singularity",
        num_tasks_to_return=1,
        save_every_n_tasks=1,
        data_dir="data",
        db_path="data/runs.db",
        sbatch_script_dir="data/sbatch_scripts",
        generated_reports_dir="data/generated_reports",
        template_dir="website_assets/templates/",
        cache_file_name="data/dataset_list.csv",
        results_dir="data/results",
    ):
        # set all the required directories
        self.main_dir_in_snellius = "/home/smukherjee/OpenML-auto-automl"
        self.automlb_dir_in_snellius = (
            "/home/smukherjee/OpenML-auto-automl/automlbenchmark"
        )
        self.script_dir_in_snellius = "/home/smukherjee/scripts"

        self.testing_mode = testing_mode
        self.use_cache = use_cache
        self.run_mode = run_mode
        self.num_tasks_to_return = num_tasks_to_return
        self.save_every_n_tasks = save_every_n_tasks
        self.db_path = Path(self.main_dir_in_snellius) / db_path
        self.sbatch_script_dir = Path(self.main_dir_in_snellius) / sbatch_script_dir
        self.generated_reports_dir = (
            Path(self.main_dir_in_snellius) / generated_reports_dir
        )
        self.template_dir = Path(self.main_dir_in_snellius) / template_dir
        self.cache_file_name = Path(self.main_dir_in_snellius) / cache_file_name
        self.results_dir = Path(self.main_dir_in_snellius) / results_dir
        self.data_dir = Path(self.main_dir_in_snellius) / data_dir

        self.benchmarks_to_use = [
            "autosklearn",
            # "autoweka",
            # "decisiontree",
            "flaml",
            "gama",
            "h2oautoml",
            # "hyperoptsklearn",
            # "lightautoml",
            # "oboe",
            # "randomforest",
            # "tpot",
            "autogluon",
        ]

        # initialize all the required directories and validate required parameters
        self._initialize()

        # load datasets
        self.datasets = self._load_datasets()
        print(f"Loaded {len(self.datasets)} datasets")

        self.sql_handler = SQLHandler(db_path)
        self.openml_task_handler = OpenMLTaskHandler()

    def _initialize(self):
        # make required directories
        self._make_dirs(
            [
                self.data_dir,
                self.db_path,
                self.sbatch_script_dir,
                self.generated_reports_dir,
                self.template_dir,
                self.results_dir,
            ]
        )
        # validate run mode
        self._check_run_mode()

    def _make_dirs(self, folders):
        for folder in folders:
            try:
                os.makedirs(folder, exist_ok=True)
            except FileExistsError:
                pass

    def _check_run_mode(self):
        valid_modes = ["local", "aws", "docker", "singularity"]
        if self.run_mode not in valid_modes:
            raise ValueError(
                f"Invalid run mode: {self.run_mode}. Valid modes are: {valid_modes}"
            )

    def _load_datasets(self):
        datasets: pd.DataFrame = openml.datasets.list_datasets(
            output_format="dataframe"
        )

        if self.use_cache and os.path.exists(self.cache_file_name):
            cached_datasets = pd.read_csv(self.cache_file_name)

            # append new datasets to the cache
            new_datasets = datasets[~datasets["did"].isin(cached_datasets["did"])]
            cached_datasets = pd.concat([cached_datasets, new_datasets])
            cached_datasets.to_csv(self.cache_file_name, index=False)

            # return new datasets
            return new_datasets

        if self.use_cache and not os.path.exists(self.cache_file_name):
            datasets.to_csv(self.cache_file_name, index=False)
            return datasets

    def get_or_create_task_from_dataset(self, dataset_id):
        """Retrieve tasks for a dataset with 10-fold Crossvalidation or try to create a task if not available."""
        try:
            tasks = openml.tasks.list_tasks(
                data_id=dataset_id, output_format="dataframe"
            )
            # Filter tasks to only include 10-fold Crossvalidation
            tasks = tasks[tasks["estimation_procedure"] == "10-fold Crossvalidation"]
            tasks["tid"] = tasks["tid"].astype(int)
            return (
                tasks["tid"].head(self.num_tasks_to_return).tolist()
                if not tasks.empty
                else None
            )
        except Exception as e:
            # print(f"Error retrieving tasks for dataset {dataset_id}: {e}")
            # return None
            print(f"Trying to create a task for dataset {dataset_id}")
            task_id = self.task_handler.try_create_task(dataset_id)
            try:
                int(task_id)
            except:
                return None
            return [int(task_id)] if task_id else None

    def generate_sbatch_for_dataset(self, dataset_id):
        print(f"Processing dataset {dataset_id}")
        # Get tasks for the dataset or create a task if not available
        task_ids = self.get_or_create_task_from_dataset(dataset_id)

        # If it was not possible to get tasks or create a task, skip the dataset
        if task_ids:
            for task_id in tqdm(
                task_ids, desc=f"Running tasks on dataset {dataset_id}"
            ):
                # Generate separate sbatch for each benchmark
                for benchmark in tqdm(
                    self.benchmarks_to_use, desc="Running benchmarks"
                ):
                    # Check if the task has already been run
                    if not self.sql_handler.task_already_run(
                        dataset_id=dataset_id, task_id=task_id, framework=benchmark
                    ):
                        # Prepare the command to execute the benchmark
                        command = [
                            "yes",
                            "|",
                            "python3",
                            "runbenchmark.py",
                            benchmark,
                            f"openml/t/{task_id}",
                            "--mode",
                            self.run_mode,
                        ]
                        command.extend(["-o", f"{self.results_dir}"])
                        command = " ".join(command)

                        # Create the sbatch script
                        sbatch_script = f"""#!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16
    #SBATCH --partition=genoa
    #SBATCH --mem=56G
    #SBATCH --time=0-01:15:00

    module load 2022
    module spider Anaconda3/2022.05
    source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh

    cd {self.automlb_dir_in_snellius}
    conda create -n automl python=3.9.19
    conda activate automl
    pip install --user -r {self.automlb_dir_in_snellius}/requirements.txt

    {command}

    source deactivate
    """
                        # Save the sbatch script to a file
                        script_path = (
                            self.sbatch_script_dir
                            / f"{dataset_id}_{task_id}_{benchmark}.sbatch"
                        )
                        with open(script_path, "w") as f:
                            f.write(sbatch_script)

    def generate_sbatch_for_all_datasets(self):
        for _, row in tqdm(
            self.datasets.iterrows(),
            total=self.datasets.shape[0],
            desc="Processing datasets",
        ):
            dataset_id = row["did"]
            self.generate_sbatch_for_dataset(dataset_id)


ags = argparse.ArgumentParser()
ags.add_argument("--use_cache", default=True)
ags.add_argument("--run_mode", default="singularity")
ags.add_argument("--generate_reports", "-r", action="store_true")
ags.add_argument("--generate_sbatch", "-s", action="store_true")
args = ags.parse_args()

print("Arguments: ", args)

runner = AutoMLRunner(
    use_cache=args.use_cache,
    run_mode=args.run_mode,
)

if args.generate_sbatch:
    runner.generate_sbatch_for_all_datasets()
