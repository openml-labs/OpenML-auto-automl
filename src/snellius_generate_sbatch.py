from pathlib import Path
from tqdm.auto import tqdm
import openml
import os
import pandas as pd
from utils import OpenMLTaskHandler, SQLHandler
import argparse
from multiprocessing import Pool, cpu_count
from typing import Union
from datetime import datetime
import concurrent.futures


class AutoMLRunner:
    def __init__(
        self,
        testing_mode=False,
        use_cache=True,
        run_mode="singularity",
        num_tasks_to_return=1,
        save_every_n_tasks=1,
        data_dir="automl_data",
        db_path="runs.db",
        sbatch_script_dir="sbatch_scripts",
        generated_reports_dir="generated_reports",
        template_dir="src/website_assets/templates/",
        cache_file_name="dataset_list.csv",
        results_dir="results",
        username="smukherjee",
        automl_max_time="02:00:00",
        api_key="",
    ):
        self.username = username
        # set all the required directories
        self.main_dir_in_snellius = f"/home/{self.username}/OpenML-auto-automl"
        self.automlb_dir_in_snellius = f"/home/{self.username}/automlbenchmark"
        self.script_dir_in_snellius = f"/home/{self.username}/scripts"
        self.data_dir = Path(f"/home/{self.username}") / data_dir
        self.automl_max_time = automl_max_time

        self.template_dir = Path(self.main_dir_in_snellius) / template_dir

        self.db_path = Path(self.data_dir) / db_path
        self.sbatch_script_dir = Path(self.data_dir) / sbatch_script_dir
        self.generated_reports_dir = Path(self.data_dir) / generated_reports_dir
        self.cache_file_name = Path(self.data_dir) / cache_file_name
        self.results_dir = results_dir

        self.testing_mode = testing_mode
        self.use_cache = use_cache
        self.run_mode = run_mode
        self.num_tasks_to_return = num_tasks_to_return
        self.save_every_n_tasks = save_every_n_tasks
        self.api_key = api_key

        self.benchmarks_to_use = [
            "autosklearn",
            # "autoweka",
            # "decisiontree",
            "flaml",
            "gama",
            # "h2oautoml",
            # "hyperoptsklearn",
            # "lightautoml",
            # "oboe",
            # "randomforest",
            # "tpot",
            # "autogluon",
        ]

        # initialize all the required directories and validate required parameters
        self._initialize()

        # load datasets
        self.datasets = self._load_datasets()
        print(f"Loaded {len(self.datasets)} datasets")

        self.sql_handler = SQLHandler(str(self.db_path))
        self.openml_task_handler = OpenMLTaskHandler()

    def _initialize(self):
        # make required directories
        self._make_dirs(
            [
                self.data_dir,
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
        if self.use_cache == False:
            return datasets

        if self.use_cache == True and os.path.exists(self.cache_file_name):
            cached_datasets = pd.read_csv(self.cache_file_name)

            # append new datasets to the cache
            new_datasets = datasets[~datasets["did"].isin(cached_datasets["did"])]
            cached_datasets = pd.concat([cached_datasets, new_datasets])
            cached_datasets.to_csv(self.cache_file_name, index=False)

            # return new datasets
            return new_datasets

        if self.use_cache == True and not os.path.exists(self.cache_file_name):
            datasets.to_csv(self.cache_file_name, index=False)
            return datasets

    def get_or_create_task_from_dataset(self, dataset_id, timeout=50):
        """Retrieve tasks for a dataset with 10-fold Crossvalidation or try to create a task if not available."""
        dataset_id = int(dataset_id)
        try:
            tasks = openml.tasks.list_tasks(
                data_id=dataset_id, output_format="dataframe"
            )

            # Filter tasks to only include 10-fold Crossvalidation
            # tasks = tasks[tasks["estimation_procedure"] == "10-fold Crossvalidation"]
            tasks["tid"] = tasks["tid"].astype(int)

            return (
                tasks["tid"].head(self.num_tasks_to_return).tolist()
                if not tasks.empty
                else None
            )
        except Exception as e:
            print(f"Trying to create a task for dataset {dataset_id}")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.openml_task_handler.try_create_task, dataset_id
                )
                try:
                    task_id = future.result(timeout=timeout)  # Timeout in seconds
                except concurrent.futures.TimeoutError:
                    print(f"Task creation timed out for dataset {dataset_id}")
                    return None
                except Exception as e:
                    print(f"Error creating task for dataset {dataset_id}: {e}")
                    return None

            try:
                int(task_id)
            except:
                return None

            return [int(task_id)] if task_id else None

    def generate_sbatch_for_dataset(self, dataset_id):
        # Get tasks for the dataset or create a task if not available
        task_ids = self.get_or_create_task_from_dataset(dataset_id)

        # If it was not possible to get tasks or create a task, skip the dataset
        if task_ids:
            for task_id in tqdm(task_ids):
                # commands = []
                for benchmark in self.benchmarks_to_use:
                    script_path = (
                        self.sbatch_script_dir
                        / f"{dataset_id}_{task_id}_{benchmark}.sh"
                    )
                    # Check if the task has already been run
                    if not os.path.exists(script_path):
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
                        # commands.append(" ".join(command))
                        command = " ".join(command)

                        if (
                            command
                        ):  # Only create the script if there are commands to run
                            # combined_commands = "\n".join(commands)

                            # Create the sbatch script
                            sbatch_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --mem=56G
#SBATCH --time=0-{self.automl_max_time}

module load 2022
module spider Anaconda3/2022.05
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh

    yes | conda activate /home/{self.username}/.conda/envs/automl

    cd {self.automlb_dir_in_snellius}
    {command}

# Try to upload the runs
    for result in $(ls {self.results_dir});do
        python {self.automlb_dir_in_snellius}/upload_results.py -m upload -a {self.api_key} -i "{self.results_dir}/$result"
    done

    source deactivate
            """
                            # Save the sbatch script to a file

                            with open(script_path, "w") as f:
                                f.write(sbatch_script)
                            print(sbatch_script)
                            return script_path

    def generate_sbatch_for_dataset_wrapper(self, args):
        self, dataset_id = args
        self.generate_sbatch_for_dataset(dataset_id)

    def generate_sbatch_for_all_datasets(self):
        dataset_ids = self.datasets["did"].tolist()
        print("Generating sbatches")
        # Use a multiprocessing pool for parallel processing
        with Pool(processes=3) as pool:
            # Use tqdm to show progress bar for parallel processing
            list(
                tqdm(
                    pool.imap(
                        self.generate_sbatch_for_dataset_wrapper,
                        [(self, did) for did in dataset_ids],
                    ),
                    total=len(dataset_ids),
                    desc="Processing datasets",
                )
            )


ags = argparse.ArgumentParser()
ags.add_argument("--use_cache", default=True)
ags.add_argument("--run_mode", default="singularity")
ags.add_argument("--generate_reports", "-r", action="store_true")
ags.add_argument("--generate_sbatch", "-s", action="store_true")
ags.add_argument("--username", type=str, default="smukherjee")
ags.add_argument(
    "--cron_mode",
    "-c",
    action="store_true",
    help="Cron mode, checks if there is are new datasets and spawns processes for them.",
)
ags.add_argument("--api-key", "-a", type=str, help="OpenML api key")
args = ags.parse_args()

print("Arguments: ", args)

if args.cron_mode or args.c:
    all_datasets = openml.datasets.list_datasets(output_format="dataframe")
    # reverse dataset
    all_datasets = all_datasets.iloc[::-1]

    new_dids = set(all_datasets["did"].values)

    data_dir = Path(f"/home/{args.username}") / "automl_data"
    os.makedirs(data_dir, exist_ok=True)
    old_datasets_csv_path = data_dir / "dataset_list_for_cronjob.csv"
    if not os.path.exists(old_datasets_csv_path):
        # logging.log(logging.DEBUG, "No cache exists. Running for the first few dataset.")
        dids_to_run = all_datasets["did"].values[:10]
        all_datasets.to_csv(old_datasets_csv_path)
    else:
        # logging.log(logging.DEBUG, "Cache exists, checking for new datasets.")
        old_dids = set(pd.read_csv(old_datasets_csv_path)["did"].values)

        dids_to_run = list(new_dids - old_dids)
        dids_to_run = [int(did) for did in dids_to_run]

        if len(dids_to_run) > 0:
            print(f"Adding {len(dids_to_run)} new datasets.")
        else:
            print("No new datasets")

        all_datasets.to_csv(old_datasets_csv_path)

    results_dir = "/home/smukherjee/automl_data/temp_results/"
    os.makedirs(results_dir, exist_ok=True)

    for did in dids_to_run:
        runner = AutoMLRunner(
            use_cache=args.use_cache,
            run_mode=args.run_mode,
            username=args.username,
            results_dir=results_dir,
            api_key=args.api_key,
        )

        sbatch_path = runner.generate_sbatch_for_dataset(dataset_id=did)
        # run sbatch
        os.system(f"sbatch {sbatch_path}")


if args.generate_sbatch:
    runner = AutoMLRunner(
        use_cache=args.use_cache,
        run_mode=args.run_mode,
        username=args.username,
        results_dir=results_dir,
        api_key=args.api_key,
    )

    runner.generate_sbatch_for_all_datasets()
