from pathlib import Path
from tqdm.auto import tqdm
import openml
import os
import pandas as pd
from utils import OpenMLTaskHandler, SQLHandler
import argparse
from report_generator import run_report_script_for_dataset


class AutoMLRunner:
    def __init__(
        self,
        testing_mode=False,
        use_cache=True,
        run_mode="docker",
        num_tasks_to_return=1,
        save_every_n_tasks=1,
        db_path="../data/runs.db",
        sbatch_script_dir="../data/sbatch_scripts",
        generate_reports=False,
    ):
        # set paths
        self.GENERATED_REPORTS_DIR = Path("../data/generated_reports")
        self.GENERATED_REPORTS_DIR.mkdir(exist_ok=True)
        self.result_path = Path("../data/results/*")
        self.template_dir = Path("./website_assets/templates/")

        self.testing_mode = testing_mode
        self.cache_file_name = "../data/dataset_list.csv"
        self.global_results_store = {}
        self.num_tasks_to_return = num_tasks_to_return
        self.use_cache = use_cache
        self.save_every_n_tasks = save_every_n_tasks
        self.benchmarks_to_use = [
            "autosklearn",
            # "autoweka",
            # "decisiontree",
            # "flaml",
            "gama",
            "h2oautoml",
            # "hyperoptsklearn",
            # "lightautoml",
            # "oboe",
            # "randomforest",
            # "tpot",
            # "autogluon",
        ]
        self.run_mode = run_mode
        self.db_path = db_path  # SQLite database path
        self.sbatch_script_dir = sbatch_script_dir
        self._initialize()
        self.task_handler = OpenMLTaskHandler()
        self.sql_handler = SQLHandler(self.db_path)
        self.generate_reports = generate_reports

    def _initialize(self):
        # Ensure required folders exist
        self._make_files(
            [
                "../data",
                "../data/results",
                self.GENERATED_REPORTS_DIR,
                self.sbatch_script_dir,
            ]
        )
        # Validate the run mode
        self._check_run_mode()
        # Load datasets, cache if needed
        self.datasets = self._load_datasets()
        # Limit datasets if testing
        if self.testing_mode:
            # self.datasets = self.datasets.sample(frac=1)
            # self.datasets = self.datasets.head(5)
            # get 2 datasets after the first 5
            self.datasets = self.datasets.iloc[3:5]

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

    def run_all_benchmarks_on_task(self, task_id, dataset_id):
        """Run benchmarks on a task."""
        for benchmark_type in tqdm(self.benchmarks_to_use, desc="Running benchmarks"):
            if not self.sql_handler.task_already_run(
                dataset_id, task_id, benchmark_type
            ):
                # cd to the automlbenchmark directory
                os.chdir("../automlbenchmark")
                command = [
                    "yes",
                    "|",
                    "python3",
                    "runbenchmark.py",
                    benchmark_type,
                    f"openml/t/{task_id}",
                    "--mode",
                    self.run_mode,
                ]
                if self.testing_mode:
                    command.insert(
                        -2, "test"
                    )  # Insert test mode before the last parameter
                command.extend(["-o", f"../data/results/"])

                print(f"Running command: {' '.join(command)}")
                os.popen(" ".join(command)).read()

                os.chdir("../src")
                # Save the run to the database after successful execution
                self.sql_handler.save_run(dataset_id, task_id, benchmark_type)
            else:
                print(
                    f"Skipping task {task_id} for dataset {dataset_id}, already run with {benchmark_type}."
                )

    def run_benchmark_on_all_datasets(self):
        """Run benchmarks on all datasets."""
        for _, row in tqdm(
            self.datasets.iterrows(),
            total=self.datasets.shape[0],
            desc="Processing datasets",
        ):
            dataset_id = row["did"]
            print(f"Processing dataset {dataset_id}")
            if not self.generate_reports:
                # Get tasks for the dataset or create a task if not available
                task_ids = self.get_or_create_task_from_dataset(dataset_id)
                # if it was either not possible to get tasks or create a task, skip the dataset
                if task_ids:
                    for task_id in tqdm(
                        task_ids, desc=f"Running tasks on dataset {dataset_id}"
                    ):
                        # Run benchmarks on the task
                        self.run_all_benchmarks_on_task(task_id, dataset_id)

            if self.generate_reports:
                print(f"Regenerating reports for dataset {dataset_id}")
                run_report_script_for_dataset(
                    self.GENERATED_REPORTS_DIR,
                    dataset_id=dataset_id,
                    result_path=self.result_path,
                    template_dir=self.template_dir,
                )
    
    def generate_sbatch_scripts(self):
        self.main_dir = "/home/smukherjee/OpenML-auto-automl/"
        self.script_dir = "/home/smukherjee/scripts"
        sbatch_template = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --mem=56G
#SBATCH --time=0-01:30:00

module load 2022
module spider Anaconda3/2022.05
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh
cd {self.script_dir}
conda create -n automl python=3.9
conda activate automl
{self.script_dir}
pip install --user -r {self.script_dir}requirements.txt
python multi_run_benchmark.py --testing_mode False --use_cache True --run_mode singularity --num_tasks_to_return 1 --save_every_n_tasks 1 --generate_reports False --generate_sbatch_only True
source deactivate"""
        raise NotImplementedError("This method is not implemented yet.")


ags = argparse.ArgumentParser()
ags.add_argument("--testing_mode", type=bool, default=False)
ags.add_argument("--use_cache", type=bool, default=True)
ags.add_argument("--run_mode", type=str, default="docker")
ags.add_argument("--num_tasks_to_return", type=int, default=1)
ags.add_argument("--save_every_n_tasks", type=int, default=1)
ags.add_argument("--generate_reports", type=bool, default=False)
ags.add_argument("--generate_sbatch_only", type=bool, default=False)
args = ags.parse_args()

tf = AutoMLRunner(
    testing_mode=args.testing_mode,
    use_cache=args.use_cache,
    run_mode=args.run_mode,
    num_tasks_to_return=args.num_tasks_to_return,
    save_every_n_tasks=args.save_every_n_tasks,
    generate_reports=args.regenerate_reports_only,
    sbatch_script_dir="../data/sbatch_scripts",
)
if not args.generate_sbatch_only:
    tf.run_benchmark_on_all_datasets()
else:
    tf.generate_sbatch_scripts()