# %%
import time
import openml
import pandas as pd
import os
import subprocess
from tqdm.auto import tqdm
import re
from pathlib import Path
from datetime import datetime
from glob import glob
from pathlib import Path
import json


# %%
class AutoMLRunner:
    def __init__(
        self,
        testing_mode=True,
        use_cache=True,
        run_mode="docker",
        num_tasks_to_return=1,
        save_every_n_tasks=1,
    ):
        self.testing_mode = testing_mode
        self.cache_file_name = "data/dataset_list.csv"
        self.global_results_store = {}
        self.num_tasks_to_return = num_tasks_to_return
        self.use_cache = use_cache
        self.save_every_n_tasks = save_every_n_tasks
        self.benchmarks_to_use = ["randomforest"]
        self.run_mode = run_mode
        self._initialize()

    def _initialize(self):
        # Ensure required folders exist
        self._make_files(["data"])
        # Validate the run mode
        self._check_run_mode()
        # Load datasets, cache if needed
        self.datasets = self._load_datasets()
        # Limit datasets if testing
        if self.testing_mode:
            self.datasets = self.datasets.head(10)

    def _check_run_mode(self):
        valid_modes = ["local", "aws", "docker", "singularity"]
        if self.run_mode not in valid_modes:
            raise ValueError(
                f"Invalid run mode: {self.run_mode}. Valid modes are: {valid_modes}"
            )

    def _load_datasets(self):
        """Load datasets from OpenML or cached CSV."""
        datasets = openml.datasets.list_datasets(output_format="dataframe")

        if self.use_cache and os.path.exists(self.cache_file_name):
            cached_datasets = pd.read_csv(self.cache_file_name)
            # Append any new datasets to the cache
            datasets = pd.concat([datasets, cached_datasets]).drop_duplicates(
                subset="did", keep="first"
            )

        # Save the updated dataset list to the cache
        if self.use_cache:
            datasets.to_csv(self.cache_file_name, index=False)

        return datasets

    def _make_files(self, folders):
        """Ensure that the necessary directories exist."""
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def get_tasks_from_dataset(self, dataset_id):
        """Retrieve tasks for a dataset with 10-fold Crossvalidation."""
        try:
            tasks = openml.tasks.list_tasks(
                data_id=dataset_id, output_format="dataframe"
            )
            # Filter tasks to only include 10-fold Crossvalidation
            tasks = tasks[tasks["estimation_procedure"] == "10-fold Crossvalidation"]
            return (
                tasks["tid"].head(self.num_tasks_to_return).tolist()
                if not tasks.empty
                else None
            )
        except Exception as e:
            print(f"Error retrieving tasks for dataset {dataset_id}: {e}")
            return None

    def run_all_benchmarks_on_task(self, task_id):
        """Run benchmarks on a task."""
        for benchmark_type in tqdm(self.benchmarks_to_use, desc="Running benchmarks"):
            command = [
                "yes",
                "|",
                "python3",
                "automlbenchmark/runbenchmark.py",
                benchmark_type,
                f"openml/t/{task_id}",
                "--mode",
                self.run_mode,
            ]
            if self.testing_mode:
                command.insert(-2, "test")  # Insert test mode before the last parameter

            print(f"Running command: {' '.join(command)}")
            os.popen(" ".join(command)).read()

    def get_task_for_dataset(self, dataset_id):
        """Run tasks for a given dataset."""
        task_ids = self.get_tasks_from_dataset(dataset_id)
        if task_ids:
            for task_id in tqdm(
                task_ids, desc=f"Running tasks on dataset {dataset_id}"
            ):
                self.run_all_benchmarks_on_task(task_id)

    def run_benchmark_on_all_datasets(self):
        """Run benchmarks on all datasets."""
        for _, row in tqdm(
            self.datasets.iterrows(),
            total=self.datasets.shape[0],
            desc="Processing datasets",
        ):
            self.get_task_for_dataset(row["did"])

    def upload_results_to_openml(self):
        raise NotImplementedError("Upload function not yet implemented.")

    def __call__(self):
        self.run_benchmark_on_all_datasets()

class VisualizationGenerator:
    def __init__(self, test_mode_subset = 10) -> None:
        self.test_mode_subset = test_mode_subset
        self.experiment_directory = Path("./automlbenchmark/results/*")
        if not self.experiment_directory.exists():
            raise FileNotFoundError()
        
        self.all_run_paths = glob(pathname=str(self.experiment_directory))
        if self.test_mode_subset == True:
            # Get a subset of paths for testing
            self.all_run_paths = self.all_run_paths[:min(self.test_mode_subset, len(self.all_run_paths))] 
    
    def get_all_run_info(self):
        for run_path in self.all_run_paths:
            run_path = Path(run_path)
            results_file = self.safe_load_file(run_path/"results.csv", "pd")
            # folders with metadata and results for folds
            folds_subfolders = glob(str(run_path/"predictions/*/*"))
            # per fold metadata
            self.get_fold_data(folds_subfolders)
    
    def safe_load_file(self, file_path, file_type):
        if file_type == "json":
            try:
                with open(Path(file_type), "r") as f:
                    return json.load(f)
            except:
                return None
        elif file_type == "pd":
            try:
                return pd.read_csv(file_path)
            except:
                return None
        else:
            raise NotImplementedError

    def get_fold_data(self, folds_subfolders):
        for fold_folder in folds_subfolders:
            metadata_json = self.safe_load_file(Path(fold_folder)/"metadata.json", "json")
            fold_results = self.safe_load_file(Path(fold_folder/"results.csv"), "pd")

# %%
tf = AutoMLRunner(
    testing_mode=True,
    use_cache=True,
    run_mode="docker",
    num_tasks_to_return=1,
    save_every_n_tasks=1,
)
# tf()
