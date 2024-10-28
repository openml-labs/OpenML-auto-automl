from dash import dcc, html
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
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
import sqlite3
from utils import OpenMLTaskHandler, SQLHandler


class DatasetAutoMLVisualizationGenerator:
    def __init__(self, test_mode_subset=10):
        self.test_mode_subset = test_mode_subset
        self.experiment_directory = Path("./data/results/*")
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
                # Grid container for the metrics and graphs
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(f"{metric_used.upper()} of each framework"),
                                dcc.Graph(
                                    id=f"{metric_used.upper()}-task",
                                    figure=px.bar(
                                        df,
                                        x="task",
                                        y="result",
                                        color="framework",
                                        barmode="group",
                                    ),
                                ),
                            ],
                            style={"grid-column": "1"},
                        ),  # Column 1
                        html.Div(
                            [
                                html.H3("Predict Duration of each framework"),
                                dcc.Graph(
                                    id="predict-duration-task",
                                    figure=px.bar(
                                        df,
                                        x="framework",
                                        y="predict_duration",
                                        color="framework",
                                        barmode="group",
                                    ),
                                ),
                            ],
                            style={"grid-column": "2"},
                        ),  # Column 2
                    ],
                    style={
                        "display": "grid",
                        "grid-template-columns": "1fr 1fr",  # Two equal-width columns
                        "gap": "20px",  # Spacing between columns
                    },
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
