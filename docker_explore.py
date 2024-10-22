# %%

from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from glob import glob
from httpx import ConnectTimeout
from pathlib import Path
from starlette.middleware.wsgi import WSGIMiddleware
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm.auto import tqdm
from typing import Optional
import dash
import dash_bootstrap_components as dbc
import json
import openml
import os
import pandas as pd
import plotly.express as px
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

    def get_all_run_info(self):
        for run_path in tqdm(self.all_run_paths, total=len(self.all_run_paths)):
            run_path = Path(run_path)
            results_file_path = run_path / "results.csv"
            results_file = self.safe_load_file(results_file_path, "pd")
            if results_file is not None:
                self.all_results = pd.concat([self.all_results, results_file])

            self.all_results["dataset_id"] = self.all_results["id"].apply(
                self.get_dataset_id_from_task_id
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
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], requests_pathname_prefix="/dash/")
        app.layout = self.dash_app_layout(dataset_results)
        return app


    def dash_app_layout(self, df):
        return html.Div([
            html.H1("Framework Performance Dashboard"),

            # show accuracy of each framework
            html.H3("Accuracy of each framework"),
            dcc.Graph(
                id="acc-task",
                figure=px.bar(
                    df,
                    x="task",
                    y="acc",
                    color="framework",
                ),
            ),

            # Table to display detailed results
            html.H3("Detailed Results"),
            html.Table([
                html.Tr([html.Th(col) for col in df.columns]),
                html.Tbody([
                    html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(len(df))
                ])
            ])
        ])

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
dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], requests_pathname_prefix="/dash/")
app.mount("/dash", WSGIMiddleware(dash_app.server))

@app.get("/automlbplot", response_class=HTMLResponse)
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ConnectTimeout))
async def automl_plot(q: Optional[int] = Query(None, description="Dataset ID")):
    """Route to serve the Dash app based on the dataset_id passed as a query parameter."""
    if q is None:
        return HTMLResponse(content="Error: 'q' (dataset_id) query parameter is required", status_code=400)

    # Fetch data for the given dataset_id
    dataset_results = visualization_generator.all_results[visualization_generator.all_results["dataset_id"] == q]

    if dataset_results.empty:
        return HTMLResponse(content=f"No results found for dataset_id {q}", status_code=404)
    
    dash_app.layout = visualization_generator.dash_app_layout(dataset_results)


    return dash_app.index()
