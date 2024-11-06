import re
from dash import dcc, html
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from glob import glob
from httpx import ConnectTimeout
from pathlib import Path
from starlette.middleware.wsgi import WSGIMiddleware
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm.auto import tqdm
from typing import Optional, Union
import dash
import dash_bootstrap_components as dbc
import json
import openml
import os
import pandas as pd
import plotly.express as px
import sqlite3
from utils import OpenMLTaskHandler, SQLHandler


class DashboardGenerator:
    def __init__(self, df: pd.DataFrame):
        self.dataframe = df

    def dash_app_layout(self) -> Union[html.Div, str]:
        required_columns = {
            "metric",
            "result",
            "framework",
            "dataset_id",
            "id",
            "task",
            "predict_duration",
            "models",
        }
        # if framework is duplicated, take the one with the best result
        self.dataframe = self.dataframe.drop_duplicates(
            subset=["framework"], keep="first"
        )

        # Add missing columns with "N/A" values
        for column in required_columns:
            if column not in self.dataframe.columns:
                self.dataframe[column] = "N/A"

        # Set the metric used, or 'N/A' if the column doesn't exist
        metric_used = (
            self.dataframe["metric"].unique()[0]
            if "metric" in self.dataframe.columns and not self.dataframe["metric"].empty
            else "N/A"
        )
        try:
            self.dataframe[metric_used]
        except KeyError:
            self.dataframe[metric_used] = "N/A"

        if metric_used == "auc":
            best_result_for_metric = (
                self.dataframe[metric_used].max() if metric_used != "N/A" else "N/A"
            )
            best_result_for_score = self.dataframe["result"].max()
        elif metric_used == "neg_logloss":
            best_result_for_metric = (
                self.dataframe[metric_used].min() if metric_used != "N/A" else "N/A"
            )
            best_result_for_score = self.dataframe["result"].max()
        else:
            print(f"Unknown metric {metric_used}")
            best_result_for_metric = (
                self.dataframe[metric_used].max() if metric_used != "N/A" else "N/A"
            )
            best_result_for_score = self.dataframe["result"].max()

        # row with the best result for the metric
        best_result_row = self.dataframe[
            self.dataframe[metric_used] == best_result_for_metric
        ]
        best_result_framework = (
            best_result_row["framework"].values[0]
            if not best_result_row.empty
            else "N/A"
        )

        # row with the best score
        best_result_row_score = self.dataframe[
            self.dataframe["result"] == best_result_for_score
        ]
        best_score_framework = (
            best_result_row_score["framework"].values[0]
            if not best_result_row_score.empty
            else "N/A"
        )
        # Create table rows only if data exists and is not "N/A"
        details_rows = [
            (
                html.Tr(
                    [
                        html.Th("Frameworks Used"),
                        html.Td(", ".join(self.dataframe["framework"].unique())),
                    ]
                )
                if self.dataframe["framework"].unique()[0] != "N/A"
                else None
            ),
            (
                html.Tr([html.Th("Metric Used"), html.Td(metric_used)])
                if metric_used != "N/A"
                else None
            ),
            html.Tr(
                [
                    html.Th(f"Best Framework by {metric_used}"),
                    html.Td(best_result_framework),
                ]
            ),
            html.Tr(
                [html.Th("Best Framework by Score"), html.Td(best_score_framework)]
            ),
        ]
        color_palette_for_plots = px.colors.qualitative.Safe

        # Filter out None entries (empty rows) from details table
        details_rows = [row for row in details_rows if row]

        # get rows that use autosklearn
        auto_sklearn_rows = self.dataframe[self.dataframe["framework"] == "autosklearn"]
        auto_sklearn_data = pd.DataFrame()
        # for each row, read the json file from the models column and get the model id and cost
        for _, row in auto_sklearn_rows.iterrows():
            models_path = row["models"]
            with open(models_path, "r") as f:
                models_file = json.load(f)
                for model in models_file:
                    model_type = (
                        "sklearn_classifier"
                        if "sklearn_classifier" in models_file[model]
                        else "sklearn_regressor"
                    )

                    auto_sklearn_data = pd.concat(
                        [auto_sklearn_data, pd.DataFrame([models_file[model]])],
                        ignore_index=True,
                    )
                    auto_sklearn_data["cleaned_model_type"] = auto_sklearn_data[
                        model_type
                    ].apply(
                        lambda x: re.sub(
                            r"\s+", "<br>", x
                        )  # Replace multiple spaces with a single space
                    )

                # add this graph to the dashboard
                # details_rows.append(

                # )

        return html.Div(
            style={
                "padding": "20px",
                "max-width": "1200px",
                "margin": "0 auto",
                "font-family": "Arial, sans-serif",
                "line-height": "1.6",
            },
            children=[
                html.Div(
                    [
                        html.H1(
                            "Run Details",
                            style={"margin-bottom": "10px", "text-align": "center"},
                        ),
                        html.Ul(
                            [
                                (
                                    html.Li(
                                        [
                                            html.A(
                                                "Dataset on OpenML",
                                                href=f"https://www.openml.org/search?type=data&sort=runs&id={self.dataframe.iloc[0]['dataset_id']}&status=active",
                                                target="_blank",
                                            ),
                                        ]
                                    )
                                    if self.dataframe.iloc[0]["dataset_id"] != "N/A"
                                    else None
                                ),
                                (
                                    html.Li(
                                        [
                                            html.A(
                                                "Task on OpenML",
                                                href=f"https://{self.dataframe.iloc[0]['id']}",
                                                target="_blank",
                                            ),
                                        ]
                                    )
                                    if self.dataframe.iloc[0]["id"] != "N/A"
                                    else None
                                ),
                            ],
                            style={
                                "list-style-type": "none",
                                "padding": "0",
                                "margin": "0",
                                "text-align": "center",
                                "margin-top": "10px",
                            },
                        ),
                        dbc.Table(
                            details_rows,
                            bordered=True,
                            hover=True,
                            striped=True,
                            style={
                                "width": "60%",
                                "overflow-x": "auto",
                                "margin": "0 auto",
                                "border": "1px solid #ddd",
                            },
                        ),
                    ],
                    style={"margin-top": "20px"},
                ),
                # Dashboard Section
                html.Div(
                    [
                        html.H1(
                            "Framework Performance Dashboard",
                            style={
                                "text-align": "center",
                                "margin-bottom": "20px",
                                "margin-top": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            f"{metric_used.upper()} of each Framework",
                                            style={"text-align": "center"},
                                        ),
                                        dcc.Graph(
                                            id=f"{metric_used.upper()}-task",
                                            figure=px.bar(
                                                self.dataframe,
                                                x="task",
                                                y="result",
                                                color="framework",
                                                barmode="group",
                                                color_discrete_sequence=color_palette_for_plots,
                                                labels={"result": metric_used},
                                            ),
                                        ),
                                    ],
                                    style={"grid-column": "1"},
                                ),
                                html.Div(
                                    [
                                        html.H3(
                                            "Predict Duration of each Framework",
                                            style={"text-align": "center"},
                                        ),
                                        dcc.Graph(
                                            id="predict-duration-task",
                                            figure=px.bar(
                                                self.dataframe,
                                                x="framework",
                                                y="predict_duration",
                                                color="framework",
                                                barmode="group",
                                                color_discrete_sequence=color_palette_for_plots,
                                                labels={
                                                    "predict_duration": "Predict Duration (s)"
                                                },
                                            ),
                                        ),
                                    ],
                                    style={"grid-column": "2"},
                                ),
                                html.Div(
                                    [
                                        html.H3(
                                            "Performance of each Framework",
                                            style={"text-align": "center"},
                                        ),
                                        dcc.Graph(
                                            id="framework-performance",
                                            figure=px.bar(
                                                self.dataframe,
                                                x="framework",
                                                y="result",
                                                color="framework",
                                                barmode="group",
                                                color_discrete_sequence=color_palette_for_plots,
                                            ),
                                        ),
                                    ],
                                    style={"grid-column": "1"},
                                ),
                                # scatter plot of inference time vs predictive performance
                                html.Div(
                                    [
                                        html.H3(
                                            "Predict Duration vs Performance",
                                            style={"text-align": "center"},
                                        ),
                                        dcc.Graph(
                                            id="predict-duration-performance",
                                            figure=px.scatter(
                                                self.dataframe,
                                                x="predict_duration",
                                                y="result",
                                                color="framework",
                                                color_discrete_sequence=color_palette_for_plots,
                                            ),
                                        ),
                                    ],
                                    style={"grid-column": "2"},
                                ),
                            ],
                            style={
                                "display": "grid",
                                "grid-template-columns": "1fr 1fr",
                                "gap": "30px",
                                "margin-bottom": "40px",
                            },
                        ),
                        # graph for task vs result
                    ],
                ),
                html.Tr(
                    [
                        html.Th(f"Model ID vs Cost for {row['framework']}"),
                        html.Td(
                            dcc.Graph(
                                figure=px.bar(
                                    auto_sklearn_data,
                                    x="model_id",
                                    y="cost",
                                    color="model_id",
                                    text="cleaned_model_type",
                                    barmode="group",
                                    color_discrete_sequence=color_palette_for_plots,
                                    hover_data={"cleaned_model_type": True},
                                ).update_traces(
                                    textposition="outside",
                                    hoverlabel=dict(
                                        font_color="black",  # Set hover text color to white
                                        bgcolor="white",
                                    ),
                                )
                            )
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H1(
                            "Detailed Results",
                            style={"margin-bottom": "10px", "text-align": "center"},
                        ),
                        html.Div(
                            dbc.Table.from_dataframe(
                                self.dataframe, striped=True, bordered=True, hover=True
                            ),
                            style={
                                "overflow-x": "auto",
                                "margin": "0 auto",
                                "border": "1px solid #ddd",
                            },
                        ),
                    ],
                    style={"margin-top": "40px"},
                ),
            ],
        )


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
        all_results_list = []  # Temporary list to store individual DataFrames

        for run_path in tqdm(self.all_run_paths, total=len(self.all_run_paths)):
            run_path = Path(run_path)
            results_file_path = run_path / "results.csv"

            # Load results file if it exists
            results_file = self.safe_load_file(results_file_path, "pd")

            # If results file is loaded, proceed to process it
            if results_file is not None:
                # Get the model path specific to this run_path
                models_path_list = list((run_path / "models").rglob("models.json"))
                models_path = str(models_path_list[0]) if models_path_list else None

                # Add the model path as a new column in the current results_file DataFrame
                results_file["models"] = models_path

                # Get the dataset ID for each row in the results file
                results_file["dataset_id"] = results_file["id"].apply(
                    self.openml_task_handler.get_dataset_id_from_task_id
                )

                # Append the processed DataFrame to our list
                all_results_list.append(results_file)

        # Concatenate all individual DataFrames into self.all_results
        if all_results_list:
            self.all_results = pd.concat(all_results_list, ignore_index=True)

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

        # Create the layout for the dashboard
        dashboard_generator = DashboardGenerator(dataset_results)
        app.layout = dashboard_generator.dash_app_layout()
        return app
