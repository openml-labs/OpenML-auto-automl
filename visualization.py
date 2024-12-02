"""
This file is responsible for generating the report. 
Note that it requires the multi_run_benchmark.py to have been run first to generate the results.
The modules in this file are used in main.py
"""

import re
from dash import dcc, html
from glob import glob
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Union
import dash
import dash_bootstrap_components as dbc
import json
from utils import OpenMLTaskHandler
import pandas as pd
import plotly.express as px
import seaborn as sns
import io
import base64
import matplotlib.pyplot as plt


class DatasetAutoMLVisualizationGenerator:
    def __init__(self, test_mode_subset=10):
        self.test_mode_subset = test_mode_subset
        self.experiment_directory = Path("./data/results/*")

        self.all_run_paths = glob(pathname=str(self.experiment_directory))
        if self.test_mode_subset == True:
            # Get a subset of paths for testing
            self.all_run_paths = self.all_run_paths[
                : min(self.test_mode_subset, len(self.all_run_paths))
            ]

        self.all_results = pd.DataFrame()
        self.openml_task_handler = OpenMLTaskHandler()

    def get_all_run_info(self):
        """
        This function is responsible for loading all the results files from the runs and storing them in self.all_results. This is further used to generate the dashboard.
        """
        all_results_list = []  # Temporary list to store individual DataFrames

        for run_path in tqdm(self.all_run_paths, total=len(self.all_run_paths)):
            run_path = Path(run_path)
            results_file_path = run_path / "results.csv"

            # Load results file if it exists
            results_file = self.safe_load_file(results_file_path, "pd")

            # If results file is loaded, proceed to process it
            if results_file is not None:
                # Get the model path specific to this run_path
                models_path_list = list((run_path / "models").rglob("models.*"))
                leaderboard_path_list = list(
                    (run_path / "models").rglob("leaderboard.*")
                )
                # models_path = str(models_path_list[0]) if len(models_path_list) >0 else None

                if len(models_path_list) > 0:
                    models_path = str(models_path_list[0])
                elif len(leaderboard_path_list) > 0:
                    models_path = str(leaderboard_path_list[0])
                else:
                    models_path = None

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

    def safe_load_file(self, file_path, file_type) -> Union[pd.DataFrame, dict, None]:
        """
        This function is responsible for safely loading a file. It returns None if the file is not found or if there is an error loading the file.
        """
        if file_type == "json":
            try:
                with open(str(Path(file_path)), "r") as f:
                    return json.load(f)
            except:
                return None
        elif file_type == "pd":
            try:
                return pd.read_csv(str(file_path))
            except:
                return None
        elif file_type == "textdict":
            try:
                with open(file_path, "r") as f:
                    return json.loads(f.read())
            except:
                return None
        else:
            raise NotImplementedError

    def create_dash_app_for_dataset(self, dataset_id):
        """
        This function is responsible for creating a Dash app for a specific dataset
        """
        dataset_results = self.all_results[self.all_results["dataset_id"] == dataset_id]
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            requests_pathname_prefix="/dash/",
        )

        app.layout = self.dash_app_layout(dataset_results)
        return app

    def graph_and_heading(
        self,
        df,
        graph_id,
        x,
        y,
        color,
        title,
        grid_column="1",
        description=None,
        type="bar",
        hover_data=None,  # Note: hover_data is not used in Seaborn plots
    ):
        """
        This function generates a graph and heading for the dashboard using Seaborn.
        It returns a Div element containing the graph and heading or an empty Div element if an error occurs.
        """
        try:
            if grid_column is not None:
                style = {"grid-column": grid_column}
            else:
                style = {}

            elements = []
            # heading
            elements.append(
                html.H3(
                    title,
                    style={"text-align": "center"},
                ),
            )
            # description
            if description is not None:
                elements.append(html.P(description))

            # Create the Seaborn plot
            plt.figure(figsize=(10, 6))  # Adjust figure size as needed

            if type == "bar":
                sns.barplot(data=df, x=x, y=y, hue=color, palette="muted")
            elif type == "scatter":
                sns.scatterplot(data=df, x=x, y=y, hue=color, palette="muted")

            plt.title(title)
            plt.tight_layout()

            # Save the plot to a buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.read()).decode()
            buffer.close()
            plt.close()

            # Add the image to the Dash layout
            elements.append(
                html.Img(
                    src=f"data:image/png;base64,{encoded_image}",
                    style={"width": "100%"},
                )
            )

            return html.Div(
                elements,
                style=style,
            )
        except Exception as e:
            print(e)
            return html.Div()

    def generate_model_vs_cost_for_frameworks(self, df_process_fns, framework_names):
        assert len(df_process_fns) == len(framework_names)
        div_list = []

        for process_fn, framework_name in zip(df_process_fns, framework_names):
            div_list.append(
                html.Div(
                    [
                        html.H1(
                            f"{framework_name} intermediate model results" if framework_name != "All results" else "All model results",
                            style={
                                "margin-bottom": "10px",
                                "text-align": "center",
                            },
                        ),
                        html.Div(
                            dbc.Table.from_dataframe(
                                process_fn,
                                striped=True,
                                bordered=True,
                                hover=True,
                            ),
                            style={
                                "overflow-x": "auto",
                                "margin": "0 auto",
                                "border": "1px solid #ddd",
                                # size of the table
                                "height": "250px",
                            },
                        ),
                    ],
                    style={"margin-top": "40px"},
                ),
            )

        return html.Div(div_list)

    def dash_app_layout(self, df: pd.DataFrame) -> Union[html.Div, str]:
        try:
            # Validate DataFrame
            if df is None or df.empty:
                return "Error: Provided DataFrame is empty or None."

            # Required columns
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

            # Handle duplicate frameworks by keeping the one with the best result
            df = df.drop_duplicates(subset=["framework"], keep="first")

            # Add missing columns with default values
            for column in required_columns:
                if column not in df.columns:
                    df[column] = "N/A"

            # Determine the metric used or set to "N/A" if unavailable
            metric_used = (
                df["metric"].unique()[0]
                if "metric" in df.columns and not df["metric"].empty
                else "N/A"
            )
            if metric_used not in df.columns:
                df[metric_used] = "N/A"

            # Define how to find the best result for the metric
            metric_used_dict = {
                "auc": lambda x: x.max(),
                "neg_logloss": lambda x: x.min(),
            }
            best_result_for_metric = (
                metric_used_dict[metric_used](df[metric_used])
                if metric_used in metric_used_dict
                else "N/A"
            )
            best_result_for_score = (
                df["result"].max() if "result" in df.columns else "N/A"
            )

            # Identify best frameworks
            try:
                best_result_framework = (
                    df[df[metric_used] == best_result_for_metric]["framework"].values[0]
                    if best_result_for_metric != "N/A"
                    else "N/A"
                )
            except IndexError:
                best_result_framework = "N/A"

            try:
                best_score_framework = (
                    df[df["result"] == best_result_for_score]["framework"].values[0]
                    if best_result_for_score != "N/A"
                    else "N/A"
                )
            except IndexError:
                best_score_framework = "N/A"

            # Generate table rows with error handling
            details_rows = []
            if "framework" in df.columns and df["framework"].unique()[0] != "N/A":
                details_rows.append(
                    html.Tr(
                        [
                            html.Th("Frameworks Used"),
                            html.Td(", ".join(df["framework"].unique())),
                        ]
                    )
                )
            if metric_used != "N/A":
                details_rows.append(
                    html.Tr([html.Th("Metric Used"), html.Td(metric_used)])
                )
            if best_result_framework != "N/A":
                details_rows.append(
                    html.Tr(
                        [
                            html.Th(f"Best Framework by {metric_used}"),
                            html.Td(best_result_framework),
                        ]
                    )
                )
            if best_score_framework != "N/A":
                details_rows.append(
                    html.Tr(
                        [
                            html.Th("Best Framework by Score"),
                            html.Td(best_score_framework),
                        ]
                    )
                )

            # display df in a table

            # Framework names and processing functions
            framework_names = ["Auto-sklearn", "H20AutoML", "AutoGluon", "All results"]
            process_fns = []
            try:
                process_fns = [
                    self.process_auto_sklearn_data(df),
                    self.get_rows_for_framework_from_df(
                        df=df, framework_name="H20AutoML"
                    ),
                    self.get_rows_for_framework_from_df(
                        df=df, framework_name="AutoGluon"
                    ),
                    self.get_rows_for_framework_from_df(
                        df=df, framework_name="All results"
                    ),
                ]
            except Exception as e:
                print(f"Error processing frameworks: {e}")
                process_fns = []

            # Final Div
            final_div = html.Div(
                style={
                    "padding": "20px",
                    "max-width": "1200px",
                    "margin": "0 auto",
                    "font-family": "Arial, sans-serif",
                    "line-height": "1.6",
                },
                children=[
                    self.generate_automl_run_details(df, details_rows),
                    self.generate_dashboard_section(df, metric_used),
                    self.generate_model_vs_cost_for_frameworks(
                        df_process_fns=process_fns, framework_names=framework_names
                    ),
                ],
            )

            return final_div

        except Exception as e:
            print(f"Error in dash_app_layout: {e}")
            return "An error occurred while generating the dashboard layout."
        
    def generate_dashboard_section(self, df, metric_used):
        return html.Div(
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
                        # Main metric vs Result
                        self.graph_and_heading(
                            df=df,
                            graph_id=metric_used.upper() + "-task",
                            x="task",
                            y="result",
                            color="framework",
                            title=f"{metric_used.upper()} of each Framework",
                            grid_column="1",
                            description="This is a plot of the main metric used in the experiment against the result of the experiment for each framework for each task. Use this plot to compare the performance of each framework for each task.",
                        ),
                        # Predict Duration of each Framework
                        self.graph_and_heading(
                            df=df,
                            graph_id="predict-duration-task",
                            x="framework",
                            y="predict_duration",
                            color="framework",
                            title="Predict Duration of each Framework",
                            grid_column="2",
                            description="This is a plot of the prediction duration for each framework for each task. Use this plot to find the framework with the fastest prediction time.",
                        ),
                        # Performance of each Framework
                        self.graph_and_heading(
                            df=df,
                            graph_id="framework-performance",
                            x="framework",
                            y="result",
                            color="framework",
                            title="Performance of each Framework",
                            grid_column="1",
                            description="This is a plot of the performance of each framework for each task. Use this plot find the best framework for the tasks.",
                        ),
                        # Scatter plot of inference time vs predictive performance
                        self.graph_and_heading(
                            df=df,
                            graph_id="predict-duration-performance",
                            x="predict_duration",
                            y="result",
                            color="framework",
                            title="Predict Duration vs Performance",
                            grid_column="2",
                            description="This is a scatter plot of the prediction duration against the performance of each framework for each task. Use this plot to find the best framework for the tasks.",
                            type="scatter",
                        ),
                    ],
                    style={
                        "display": "grid",
                        "grid-template-columns": "1fr 1fr",
                        "gap": "30px",
                        "margin-bottom": "40px",
                    },
                ),
            ],
        )

    def generate_automl_run_details(self, df, details_rows):
        return html.Div(
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
                                        href=f"https://www.openml.org/search?type=data&sort=runs&id={df.iloc[0]['dataset_id']}&status=active",
                                        target="_blank",
                                    ),
                                ]
                            )
                            if df.iloc[0]["dataset_id"] != "N/A"
                            else None
                        ),
                        (
                            html.Li(
                                [
                                    html.A(
                                        "Task on OpenML",
                                        href=f"https://{df.iloc[0]['id']}",
                                        target="_blank",
                                    ),
                                ]
                            )
                            if df.iloc[0]["id"] != "N/A"
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
        )

    def process_auto_sklearn_data(self, df):
        auto_sklearn_rows = df[df["framework"] == "autosklearn"]
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

        return auto_sklearn_data

    def get_rows_for_framework_from_df(self, df, framework_name, top_n=40):
        try:
            if framework_name == "All results":
                return df
            framework_rows = df[df["framework"] == framework_name]["models"].values[0]
            framework_data = self.safe_load_file(framework_rows, "pd")
            if top_n is not None:
                framework_data = framework_data.head(40)

            return framework_data
        except IndexError:
            return pd.DataFrame()
