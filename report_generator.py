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
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, request, send_file
from jinja2 import Environment, FileSystemLoader
from ollama import chat
from ollama import ChatResponse
from itables import to_html_datatable
from typing import Any
import openml
import markdown
import sqlite3
from utils import safe_load_file
import plotly.graph_objs as go
import plotly.express as px
GENERATED_REPORTS_DIR = Path("./generated_reports")
GENERATED_REPORTS_DIR.mkdir(exist_ok=True)


class ResultCollector:
    def __init__(self, path: str = "./data/results/*"):
        self.experiment_directory = Path(path)

        self.all_run_paths = glob(pathname=str(self.experiment_directory))
        self.all_results = pd.DataFrame()
        self.openml_task_handler = OpenMLTaskHandler()
        # Required columns
        self.required_columns = {
            "metric",
            "result",
            "framework",
            "dataset_id",
            "id",
            "task",
            "predict_duration",
            "models",
        }

        # Define how to find the best result for the metric
        self.metric_used_dict = {
            "auc": lambda x: x.max(),
            "neg_logloss": lambda x: x.min(),
        }

    def get_dataset_description_from_id(self, dataset_id: int) -> Optional[str]:
        return openml.datasets.get_dataset(dataset_id).description

    def collect_all_run_info_to_df(self):
        """
        This function is responsible for loading all the results files from the runs and storing them in self.all_results. This is further used to generate the dashboard.
        """
        all_results_list = []  # Temporary list to store individual DataFrames

        for run_path in tqdm(self.all_run_paths, total=len(self.all_run_paths)):
            run_path = Path(run_path)
            results_file_path = run_path / "results.csv"

            # Load results file if it exists
            results_file = safe_load_file(results_file_path, "pd")

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
                # drop all rows wth missing dataset_id

                # Get the dataset ID for each row in the results file
                results_file["dataset_id"] = results_file["id"].apply(
                    self.openml_task_handler.get_dataset_id_from_task_id
                )
                results_file["dataset_description"] = results_file["dataset_id"].apply(
                    self.get_dataset_description_from_id
                )

                # Append the processed DataFrame to our list
                all_results_list.append(results_file)

        # Concatenate all individual DataFrames into self.all_results
        if all_results_list:
            self.all_results = pd.concat(all_results_list, ignore_index=True)

    def validate_dataframe_and_add_extra_info(self):
        # Validate DataFrame
        if self.all_results is None or self.all_results.empty:
            return "Error: Provided DataFrame is empty or None."

        # Handle duplicate frameworks by keeping the one with the best result
        self.all_results = self.all_results.drop_duplicates(
            subset=["framework"], keep="first"
        )

        # Add missing columns with default values
        for column in self.required_columns:
            if column not in self.all_results.columns:
                self.all_results[column] = "N/A"

    def __call__(self):
        self.collect_all_run_info_to_df()
        self.validate_dataframe_and_add_extra_info()


class GetResultForDataset:
    def __init__(self, dataset_id: int, collector: ResultCollector):
        self.dataset_id = dataset_id
        self.collector = collector
        self.current_results = self.get_results_for_dataset_id(self.dataset_id)
        self.jinja_environment = Environment(
            loader=FileSystemLoader("./website_assets/templates/")
        )
        self.template_to_use = {
            "best_result": "best_result_table.html",
            "framework_table": "framework_table.html",
            "metric_vs_result": "metric_vs_result.html",
        }
        binary_metrics = [
            "auc",
            "logloss",
            "acc",
            "balacc",
        ]  # available metrics: auc (AUC), acc (Accuracy), balacc (Balanced Accuracy), pr_auc (Precision Recall AUC), logloss (Log Loss), f1, f2, f05 (F-beta scores with beta=1, 2, or 0.5), max_pce, mean_pce (Max/Mean Per-Class Error).
        multiclass_metrics = [
            "logloss",
            "acc",
            "balacc",
        ]  # available metrics: same as for binary, except auc, replaced by auc_ovo (AUC One-vs-One), auc_ovr (AUC One-vs-Rest). AUC metrics and F-beta metrics are computed with weighted average.
        regression_metrics = [
            "rmse",
            "r2",
            "mae",
        ]  # available metrics: mae (Mean Absolute Error), mse (Mean Squared Error), msle (Mean Squared Logarithmic Error), rmse (Root Mean Square Error), rmsle (Root Mean Square Logarithmic Error), r2 (R^2).
        timeseries_metrics = [
            "mase",
            "mape",
            "smape",
            "wape",
            "rmse",
            "mse",
            "mql",
            "wql",
            "sql",
        ]  # available metrics: mase (Mean Absolute Scaled Error), mape (Mean Absolute Percentage Error),
        self.all_metrics = (
            binary_metrics
            + multiclass_metrics
            + regression_metrics
            + timeseries_metrics
        )

        # run the function to get the best result
        self.get_best_result()
        self.framework_names = ["Auto-sklearn", "H20AutoML", "AutoGluon", "All results"]
        self.process_fns = [
            self.process_auto_sklearn_data(self.current_results),
            self.get_rows_for_framework_from_df(
                df=self.current_results, framework_name="H20AutoML", top_n=10
            ),
            self.get_rows_for_framework_from_df(
                df=self.current_results, framework_name="AutoGluon", top_n=10
            ),
            self.get_rows_for_framework_from_df(
                df=self.current_results, framework_name="All results"
            ),
        ]

    def get_results_for_dataset_id(self, dataset_id: int) -> Optional[pd.DataFrame]:
        """
        This function returns the results for a given dataset_id. If no results are found, it returns None.
        """
        results_for_dataset = self.collector.all_results[
            self.collector.all_results["dataset_id"] == dataset_id
        ]
        if results_for_dataset.empty:
            return None
        return results_for_dataset

    def get_best_result(self):
        """
        This function returns the best result from the current_results DataFrame. It first sorts the DataFrame based on the metric used and then returns the best result.
        """
        if self.current_results is None:
            return None
        metric_used = self.current_results["metric"].iloc[0]
        if metric_used in ["auc", "acc", "balacc"]:
            # Since higher value is better we sort in descending order
            sort_in_ascending_order = False
        elif metric_used in ["logloss", "neg_logloss"]:
            # Since lower value is better we sort in ascending order
            sort_in_ascending_order = True
        else:
            sort_in_ascending_order = False

        sorted_results = self.current_results.sort_values(
            by="result", ascending=sort_in_ascending_order
        ).head()

        best_result = sorted_results.iloc[0]
        self.best_framework = best_result["framework"]
        self.best_metric = best_result["metric"]
        self.type_of_task = best_result["type"]
        self.dataset_id = best_result["dataset_id"]
        self.task_id = "https://" + best_result["id"]
        self.task_name = best_result["task"]
        self.best_result_for_metric = best_result["result"]
        self.description = best_result["dataset_description"]

        # all metric columns that are in the dataframe and in the list of all metrics
        metric_columns = [
            col for col in self.current_results.columns if col in self.all_metrics
        ]
        all_metrics_present = []
        for metric in metric_columns:
            try:
                all_metrics_present.append(self.current_results[metric].values[0])
            except:
                pass

        self.metric_and_result = " ".join(
            [
                f"The {metric} is {result} "
                for metric, result in zip(metric_columns, all_metrics_present)
            ]
        )

    def generate_best_result_table(self):
        """
        This function generates the best result table using the best result information.
        """
        template = self.jinja_environment.get_template(
            self.template_to_use["best_result"]
        )
        return template.render(
            best_framework=self.best_framework,
            best_metric=self.best_metric,
            type_of_task=self.type_of_task,
            dataset_id=self.dataset_id,
            task_id=self.task_id,
            task_name=self.task_name,
        )

    def process_auto_sklearn_data(self, df, top_n=10):
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
        try:
            auto_sklearn_data = auto_sklearn_data.sort_values(
                by="cost", ascending=True
            ).head(top_n)
        except KeyError:
            auto_sklearn_data = auto_sklearn_data.head(top_n)

            # return auto_sklearn_data.to_html()
            return to_html_datatable(auto_sklearn_data, caption="Auto Sklearn Models")
        

    def get_rows_for_framework_from_df(
        self, df: pd.DataFrame, framework_name, top_n=40
    ):
        try:
            if framework_name == "All results":
                # drop the description column if it exists
                try:
                    df.drop("dataset_description", axis=1, inplace=True)
                except:
                    pass
                return to_html_datatable(df, caption="All Results")
            framework_rows: pd.DataFrame = df[df["framework"] == framework_name][
                "models"
            ].values[0]
            framework_data = safe_load_file(framework_rows, "pd")
            if top_n is not None:
                framework_data = framework_data.head(40)

            return to_html_datatable(framework_data, caption=f"{framework_name} Models")
        except:
            return ""

    def generate_framework_table(self):
        """
        This function generates the framework table using the framework_name information.
        """

        complete_html = ""
        for framework_name, process_fn in zip(self.framework_names, self.process_fns):
            if framework_name == "All results":
                complete_html += process_fn
            else:
                complete_html += process_fn

        return f"""<div class="container">
                <h2>{framework_name}</h2>

                    {complete_html}
                </div>
                </div>
                """

    def generate_dashboard_section(self):
        dashboard_html = f"""
        <div style="text-align: center; margin-bottom: 20px; margin-top: 20px;">
            <h1>Framework Performance Dashboard</h1>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;">
        {self.graph_and_heading(self.current_results, self.best_metric.upper() + "-task", "task", "result", "framework", f"{self.best_metric.upper()} of each Framework", "1", "This is a plot of the main metric used in the experiment against the result of the experiment for each framework for each task. Use this plot to compare the performance of each framework for each task.", "bar")}
        {self.graph_and_heading(self.current_results, "predict-duration-task", "framework", "predict_duration", "framework", "Predict Duration of each Framework", "2", "This is a plot of the prediction duration for each framework for each task. Use this plot to find the framework with the fastest prediction time.", "bar")}
        {self.graph_and_heading(self.current_results, "framework-performance", "framework", "result", "framework", "Performance of each Framework", "1", "This is a plot of the performance of each framework for each task. Use this plot find the best framework for the tasks.", "bar")}
        {self.graph_and_heading(self.current_results, "predict-duration-performance", "predict_duration", "result", "framework", "Predict Duration vs Performance", "2", "This is a scatter plot of the prediction duration against the performance of each framework for each task. Use this plot to find the best framework for the tasks.", "scatter")}
        </div>
        """
        return dashboard_html

    def graph_and_heading(
        self,
        df,
        graph_id,
        x,
        y,
        color,
        title,
        grid_column,
        description,
        plot_type="bar",
    ):
        try:
            # Create the plot
            # plt.figure(figsize=(10, 6))
            # if plot_type == "bar":
            #     sns.barplot(data=df, x=x, y=y, hue=color, palette="muted")

            # elif plot_type == "scatter":
            #     sns.scatterplot(data=df, x=x, y=y, hue=color, palette="muted")
            # plt.title(title)
            # # display values on top of the bars

            # plt.tight_layout()

            # # Save the plot to a buffer
            # buffer = io.BytesIO()
            # plt.savefig(buffer, format="png")
            # buffer.seek(0)
            # encoded_image = base64.b64encode(buffer.read()).decode()
            # buffer.close()
            # plt.close()


            # use plotly to create the plot
            if plot_type == "bar":
                fig = px.bar(df, x=x, y=y, color=color, title=title)
            elif plot_type == "scatter":
                fig = px.scatter(df, x=x, y=y, color=color, title=title)
            fig.update_layout(
                title=title,
                xaxis_title=x,
                yaxis_title=y,
            )
            encoded_image = fig.to_html(full_html=False, include_plotlyjs="cdn")

            # Embed the plot in HTML
            graph_html = f"""
            <div style="grid-column: {grid_column};">
                <h3 style="text-align: center;">{title}</h3>
                <p>{description}</p>
                <img src="data:image/png;base64,{encoded_image}" style="width: 100%;">
            </div>
            """
            return graph_html
        except Exception as e:
            print(e)
            return f"<div style='grid-column: {grid_column};'><p>Error generating graph: {str(e)}</p></div>"

    def get_explanation_from_llm(self):
        prompt_format = f"""For a dataset called {self.task_name} , the best framework is {self.best_framework} with a {self.best_metric} of {self.best_result_for_metric}. This is a {self.type_of_task} task. The results are as follows {self.metric_and_result}. For each metric, tell me if this is a good score (and why), and if it is not, how can I improve it? Keep your answer to the point.
        The dataset description is: {self.description}
    """
        response: ChatResponse = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt_format,
                },
            ],
            options={
                "temperature": 0.3,
            },
        )
        response = response["message"]["content"]
        markdown_response = markdown.markdown(response)
        return markdown_response


def run_for_single_dataset(dataset_id, collector, GENERATED_REPORTS_DIR):
    result_for_dataset = GetResultForDataset(dataset_id, collector=collector)
    best_result_table = result_for_dataset.generate_best_result_table()

    framework_table = result_for_dataset.generate_framework_table()
    dashboard_section = result_for_dataset.generate_dashboard_section()
    explanation = result_for_dataset.get_explanation_from_llm()

    combined_html = f"""
    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">

    <!-- jQuery library -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>

    <!-- Latest compiled JavaScript -->
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
    <div class="container">
        {best_result_table}
        {dashboard_section}
        <h2>Explanation and What's next?</h2>
        <p>!!! This is an AI-generated (llama3.2) explanation of the results. Please take the response with a grain of salt and use your own judgement.</p>
        <p>{explanation}</p>
        {framework_table}

    </div>
    """

    with open(GENERATED_REPORTS_DIR / f"report_{dataset_id}.html", "w") as f:
        f.write(combined_html)


def run_report_script_for_all_datasets(
    GENERATED_REPORTS_DIR, max_existing_dataset_id, collector
):
    for dataset_id in tqdm(range(1, max_existing_dataset_id + 1)):
        try:
            run_for_single_dataset(dataset_id, collector, GENERATED_REPORTS_DIR  )
        except Exception as e:
            print(f"Error generating report for dataset {dataset_id}: {str(e)}")
