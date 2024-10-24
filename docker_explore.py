# %%

from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from glob import glob
from httpx import ConnectTimeout
from pandas import CategoricalDtype
from pathlib import Path
from starlette.middleware.wsgi import WSGIMiddleware
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm.auto import tqdm
from typing import Optional
import dash
import dash_bootstrap_components as dbc
import json
import numpy as np
import openml
import os
import pandas as pd
import plotly.express as px
import sqlite3

# %%


class OpenMLTaskHandler:
    def __init__(self):
        pass

    def get_target_col_type(self, dataset, target_col_name):
        try:
            if dataset.features:
                return next(
                    (
                        feature.data_type
                        for feature in dataset.features.values()
                        if feature.name == target_col_name
                    ),
                    None,
                )
        except Exception as e:
            print(f"Error getting target column type: {e}")
            return None

    def check_if_api_key_is_valid(self):
        if not openml.config.get_config_as_dict()["apikey"]:
            print(
                "API key is not set. Please set the API key using openml.config.apikey = 'your-key'"
            )
            return False
        return True

    def try_create_task(self, dataset_id):
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            target_col_name = dataset.default_target_attribute
            target_col_type = self.get_target_col_type(dataset, target_col_name)

            if target_col_type:
                if target_col_type in ["nominal", "string", "categorical"]:
                    evaluation_measure = "predictive_accuracy"
                    task_type = openml.tasks.TaskType.SUPERVISED_CLASSIFICATION
                elif target_col_type == "numeric":
                    evaluation_measure = "mean_absolute_error"
                    task_type = openml.tasks.TaskType.SUPERVISED_REGRESSION
                else:
                    return None

                task = openml.tasks.create_task(
                    dataset_id=dataset_id,
                    task_type=task_type,
                    target_name=target_col_name,
                    evaluation_measure=evaluation_measure,
                    estimation_procedure_id=1,
                )

                if self.check_if_api_key_is_valid():
                    task.publish()
                    print(f"Task created: {task}, task_id: {task.task_id}")
                    return task.task_id
                else:
                    return None
            else:
                return None

        except Exception as e:
            print(f"Error creating task: {e}")
            return None

    def get_openml_task_id_from_string(self, string):
        try:
            return int(string.split("/")[-1])
        except:
            return None

    def get_dataset_id_from_task_id(self, string):
        task_id = self.get_openml_task_id_from_string(string=string)
        if task_id is not None:
            try:
                return openml.tasks.get_task(
                    task_id=task_id,
                    download_data=False,
                    download_qualities=False,
                    download_splits=False,
                    download_features_meta_data=False,
                ).dataset_id
            except:
                return None
        else:
            return None


class SQLHandler:
    def __init__(self, db_path):
        self.db_path = db_path
        self.initialize_database()

    def initialize_database(self):
        """Set up the SQLite database if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                dataset_id INTEGER,
                task_id INTEGER,
                framework TEXT,
                PRIMARY KEY (dataset_id, task_id, framework)
            )
        """
        )
        conn.commit()
        conn.close()

    def task_already_run(self, dataset_id, task_id, framework):
        """Check if a task has already been run for a given dataset, task, and framework."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM runs WHERE dataset_id = ? AND task_id = ? AND framework = ?
        """,
            (dataset_id, task_id, framework),
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def save_run(self, dataset_id, task_id, framework):
        """Save a run to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO runs (dataset_id, task_id, framework) VALUES (?, ?, ?)
        """,
            (dataset_id, task_id, framework),
        )
        conn.commit()
        conn.close()


class AutoMLRunner:
    def __init__(
        self,
        testing_mode=True,
        use_cache=True,
        run_mode="docker",
        num_tasks_to_return=1,
        save_every_n_tasks=1,
        db_path="data/runs.db",
    ):
        self.testing_mode = testing_mode
        self.cache_file_name = "data/dataset_list.csv"
        self.global_results_store = {}
        self.num_tasks_to_return = num_tasks_to_return
        self.use_cache = use_cache
        self.save_every_n_tasks = save_every_n_tasks
        self.benchmarks_to_use = ["randomforest", "gama"]
        self.run_mode = run_mode
        self.db_path = db_path  # SQLite database path
        self._initialize()
        self.task_handler = OpenMLTaskHandler()
        self.sql_handler = SQLHandler(self.db_path)

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

    def get_or_create_task_from_dataset(self, dataset_id):
        """Retrieve tasks for a dataset with 10-fold Crossvalidation or try to create a task if not available."""
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
            # print(f"Error retrieving tasks for dataset {dataset_id}: {e}")
            # return None
            print(f"Trying to create a task for dataset {dataset_id}")
            task_id = self.task_handler.try_create_task(dataset_id)
            return [task_id] if task_id else None

    def run_all_benchmarks_on_task(self, task_id, dataset_id):
        """Run benchmarks on a task."""
        for benchmark_type in tqdm(self.benchmarks_to_use, desc="Running benchmarks"):
            if not self.sql_handler.task_already_run(
                dataset_id, task_id, benchmark_type
            ):
                # cd to the automlbenchmark directory
                os.chdir("automlbenchmark")
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

                print(f"Running command: {' '.join(command)}")
                os.popen(" ".join(command)).read()

                os.chdir("..")
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
            # Get tasks for the dataset or create a task if not available
            task_ids = self.get_or_create_task_from_dataset(dataset_id)
            # if it was either not possible to get tasks or create a task, skip the dataset
            if task_ids:
                for task_id in tqdm(
                    task_ids, desc=f"Running tasks on dataset {dataset_id}"
                ):
                    # Run benchmarks on the task
                    self.run_all_benchmarks_on_task(task_id, dataset_id)

    def __call__(self):
        self.run_benchmark_on_all_datasets()


class DatasetAutoMLVisualizationGenerator:
    def __init__(self, test_mode_subset=10):
        self.test_mode_subset = test_mode_subset
        self.experiment_directory = Path("./automlbenchmark/results/*")
        # if not self.experiment_directory.exists():
        # raise FileNotFoundError()

        self.all_run_paths = glob(pathname=str(self.experiment_directory))
        if self.test_mode_subset == True:
            # Get a subset of paths for testing
            self.all_run_paths = self.all_run_paths[
                : min(self.test_mode_subset, len(self.all_run_paths))
            ]

        self.all_results = pd.DataFrame()
        self.openml_task_handler = OpenMLTaskHandler()

    def get_all_run_info(self):
        for run_path in tqdm(self.all_run_paths, total=len(self.all_run_paths)):
            run_path = Path(run_path)
            results_file_path = run_path / "results.csv"
            results_file = self.safe_load_file(results_file_path, "pd")
            if results_file is not None:
                self.all_results = pd.concat([self.all_results, results_file])

            self.all_results["dataset_id"] = self.all_results["id"].apply(
                self.openml_task_handler.get_dataset_id_from_task_id
            )

    def safe_load_file(self, file_path, file_type):
        if file_type == "json":
            try:
                with open(str(Path(file_type)), "r") as f:
                    return json.load(f)
            except:
                return None
        elif file_type == "pd":
            try:
                return pd.read_csv(str(file_path))
            except:
                return None
        else:
            raise NotImplementedError

    def create_dash_app_for_dataset(self, dataset_id):
        # all_datasets = self.all_results["dataset_id"].unique()
        # for all datasets, use the dash app layout to create a dashboard and save it to an html file
        # for dataset in all_datasets:
        dataset_results = self.all_results[self.all_results["dataset_id"] == dataset_id]
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            requests_pathname_prefix="/dash/",
        )
        app.layout = self.dash_app_layout(dataset_results)
        return app

    def dash_app_layout(self, df):
        metric_used = df["metric"].unique()[0]
        return html.Div(
            [
                html.H1("Framework Performance Dashboard"),
                # show metric of each framework
                html.H3(f"{metric_used} of each framework"),
                dcc.Graph(
                    id=f"{metric_used}-task",
                    figure=px.bar(
                        df,
                        x="task",
                        y="result",
                        color="framework",
                    ),
                ),
                # Table to display detailed results
                html.H3("Detailed Results"),
                html.Table(
                    [
                        html.Tr([html.Th(col) for col in df.columns]),
                        html.Tbody(
                            [
                                html.Tr(
                                    [html.Td(df.iloc[i][col]) for col in df.columns]
                                )
                                for i in range(len(df))
                            ]
                        ),
                    ]
                ),
            ]
        )


# %%
tf = AutoMLRunner(
    testing_mode=True,
    use_cache=True,
    run_mode="docker",
    num_tasks_to_return=1,
    save_every_n_tasks=1,
)
# tf()

# Initialize FastAPI
app = FastAPI()

# Initialize DatasetAutoMLVisualizationGenerator
visualization_generator = DatasetAutoMLVisualizationGenerator()
visualization_generator.get_all_run_info()

# Dash app instance is created once and mounted globally.
dash_app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix="/dash/",
)
app.mount("/dash", WSGIMiddleware(dash_app.server))


@app.get("/automlbplot", response_class=HTMLResponse)
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ConnectTimeout))
async def automl_plot(q: Optional[int] = Query(None, description="Dataset ID")):
    """Route to serve the Dash app based on the dataset_id passed as a query parameter."""
    if q is None:
        return HTMLResponse(
            content="Error: 'q' (dataset_id) query parameter is required",
            status_code=400,
        )

    # Fetch data for the given dataset_id
    dataset_results = visualization_generator.all_results[
        visualization_generator.all_results["dataset_id"] == q
    ]

    if dataset_results.empty:
        return HTMLResponse(
            content=f"No results found for dataset_id {q}", status_code=404
        )

    dash_app.layout = visualization_generator.dash_app_layout(dataset_results)

    return dash_app.index()
