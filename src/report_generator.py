"""
This file is responsible for generating the report. 
Note that it requires the multi_run_benchmark.py to have been run first to generate the results.
The modules in this file are used in main.py
"""

from dash import dcc, html
from flask import Flask, render_template, jsonify, request, send_file
from glob import glob
from interpret import set_visualize_provider
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.provider import InlineProvider
from itables import to_html_datatable
from jinja2 import Environment, FileSystemLoader
from ollama import ChatResponse
from ollama import chat
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.auto import tqdm
from typing import Optional, Union
from utils import OpenMLTaskHandler
import base64
import dash
import dash_bootstrap_components as dbc
import io
import json
import markdown
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
import re
import seaborn as sns
import sqlite3
from typing import Any

matplotlib.use("agg")
set_visualize_provider(InlineProvider())


def safe_load_file(file_path, file_type) -> Union[pd.DataFrame, dict, None]:
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


class DataReportGenerator:
    def __init__(self, generated_ebm_report_dir):
        self.generated_ebm_report_dir = generated_ebm_report_dir

    def generate_ebm_report(self, names, scores):
        fig = px.bar(
            x=names,
            y=scores,
            orientation="v",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(
            title="Feature Importance", xaxis_title="Feature", yaxis_title="Score"
        )
        return fig.to_html(
            full_html=False, include_plotlyjs="cdn", div_id="feature-importance"
        )

    def run_ebm_on_dataset(self, dataset_id, X_train, y_train):
        try:
            ebm = ExplainableBoostingClassifier(random_state=42)
            ebm.fit(X_train, y_train)
            ebm_global = ebm.explain_global().data()
            names, scores = ebm_global["names"], ebm_global["scores"]
            return self.generate_ebm_report(names, scores)
        except Exception as e:
            return "<div>Unable to generate feature importance report</div>"

    def get_data_and_split(self, dataset_id):
        dataset = openml.datasets.get_dataset(dataset_id=dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = pd.get_dummies(X, prefix_sep=".").astype(float)
        y, y_categories = y.factorize()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        return X, y, X_train, y_train

    def get_feature_distribution(self, X):
        try:
            return to_html_datatable(X.describe().T)
        except Exception as e:
            return "<div>Unable to generate feature distribution</div>"

    def class_imbalance(self, y):
        try:
            return to_html_datatable(pd.DataFrame(y).value_counts(), index=True)
        except Exception as e:
            return "<div>Unable to generate class imbalance report</div>"

    def get_missing_value_count(self, X):
        try:
            return to_html_datatable(
                pd.DataFrame(X.isnull().sum(), columns=["Missing Value Count"])
            )
        except Exception as e:
            return "<div>Unable to generate missing value count</div>"

    def generate_data_report_for_dataset(self, dataset_id):
        report_path = f"{self.generated_ebm_report_dir}/{dataset_id}_report.html"
        if os.path.exists(report_path):
            pass
        else:
            try:
                X, y, X_train, y_train = self.get_data_and_split(dataset_id)

                ebm_report = self.run_ebm_on_dataset(dataset_id, X_train, y_train)
                missing_value_count = self.get_missing_value_count(X)
                feature_distribution = self.get_feature_distribution(X)
                class_imbalance_report = self.class_imbalance(y)

                report_html = f"""
                    <h1>Extra Dataset Information</h1>
                    <h2>Feature Importance</h2>
                    {ebm_report}
                    <h2>Feature Distribution</h2>
                    {feature_distribution}
                    <h2>Class Imbalance</h2>
                    {class_imbalance_report}
                    <h2>Missing Value Count</h2>
                    {missing_value_count}
                    </div>
                    """

                report_path = (
                    f"{self.generated_ebm_report_dir}/{dataset_id}_report.html"
                )
                with open(report_path, "w") as f:
                    f.write(report_html)
            except Exception as e:
                print(f"Error processing dataset {dataset_id}: {e}")


class ResultCollector:
    def __init__(self, path: str = "../data/results/*"):
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
        dataset_id = int(dataset_id)
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

                # Get the dataset ID for each row in the results file
                try:
                    results_file["dataset_id"] = results_file["id"].apply(
                        self.openml_task_handler.get_dataset_id_from_task_id
                    )
                except Exception as e:
                    results_file["dataset_id"] = None

                results_file["dataset_description"] = results_file["dataset_id"].apply(
                    self.get_dataset_description_from_id
                )
                results_file["dataset_description"] = None

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
        return self.all_results
        # self.validate_dataframe_and_add_extra_info()


class GenerateCompleteReportForDataset:
    def __init__(
        self,
        dataset_id: int,
        collector_results,
        GENERATED_REPORTS_DIR: str = "../data/generated_reports",
        GENERATED_DATA_REPORT_DIR: str = "../data/generated_data_reports",
    ):
        self.dataset_id = dataset_id
        self.collector_results = collector_results
        self.current_results = self.get_results_for_dataset_id(self.dataset_id)
        self.jinja_environment = Environment(
            loader=FileSystemLoader("./website_assets/templates/")
        )
        self.generated_final_reports_dir = GENERATED_REPORTS_DIR
        self.generated_data_reports_dir = GENERATED_DATA_REPORT_DIR
        self.template_to_use = {
            "dataset_info": "data_information.html",
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

        self.best_framework = ""
        self.best_metric = ""
        self.type_of_task = ""
        self.dataset_id = ""
        self.task_id = ""
        self.task_name = ""
        self.best_result_for_metric = ""
        self.description = ""
        self.metric_and_result = ""

        self.get_best_result()

    def get_results_for_dataset_id(self, dataset_id: int) -> Optional[pd.DataFrame]:
        """
        This function returns the results for a given dataset_id. If no results are found, it returns None.
        """
        results_for_dataset = self.collector_results[
            self.collector_results["dataset_id"] == dataset_id
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
        self.best_framework = best_result.get("framework", "")
        self.best_metric = best_result.get("metric", "")
        self.type_of_task = best_result.get("type", "")
        self.dataset_id = best_result.get("dataset_id", "")
        self.task_id = "https://" + best_result.get("id", "")
        self.task_name = best_result.get("task", "")
        self.best_result_for_metric = best_result.get("result", "")
        self.description = best_result.get("dataset_description", "")

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

    def generate_dataset_info(self):
        """
        This function generates the dataset information table using the dataset information.
        """
        template = self.jinja_environment.get_template(
            self.template_to_use["dataset_info"]
        )
        return template.render(
            dataset_id=self.dataset_id,
            task_name=self.task_name,
        )

    def process_auto_sklearn_data(self, df, top_n=10):
        auto_sklearn_data = pd.DataFrame()
        try:
            auto_sklearn_rows = df[df["framework"] == "autosklearn"]
            # for each row, read the json file from the models column and get the model id and cost
            for _, row in auto_sklearn_rows.iterrows():
                models_path = row["models"]
                try:
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
                except:
                    pass
                auto_sklearn_data = auto_sklearn_data.sort_values(
                    by="cost", ascending=True
                ).head(top_n)
                return to_html_datatable(
                    auto_sklearn_data, caption="Auto Sklearn Models"
                )
        except Exception as e:
            print(e)
            return "<div></div>"

        # return auto_sklearn_data.to_html()

    def get_rows_for_framework_from_df(
        self, df: pd.DataFrame, framework_name, top_n=40
    ):
        try:
            if framework_name == "All results":
                # drop the description column if it exists
                try:
                    df.drop("dataset_description", axis=1)
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
            try:
                complete_html += process_fn
            except:
                pass

        return f"""<div class="container" style="margin-top: 20px; text-align: left;">
                <h2>{framework_name}</h2>
                    {complete_html}
                </div>
                """

    def generate_dashboard_section(self):
        dashboard_html = f"""
        <div style="text-align: left; margin: 20px 0;">
            <h1>Framework Performance Dashboard</h1>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin-bottom: 40px;">
            {self.graph_and_heading(
                self.current_results,
                self.best_metric.upper() + "-task",
                "task",
                "result",
                "framework",
                f"{self.best_metric.upper()} of each Framework",
                "1",
                "This is a plot of the main metric used in the experiment against the result of the experiment for each framework for each task. Use this plot to compare the performance of each framework for each task.",
                "bar"
            )}
            {self.graph_and_heading(
                self.current_results,
                "predict-duration-task",
                "framework",
                "predict_duration",
                "framework",
                "Predict Duration of each Framework",
                "2",
                "This is a plot of the prediction duration for each framework for each task. Use this plot to find the framework with the fastest prediction time.",
                "bar"
            )}
            {self.graph_and_heading(
                self.current_results,
                "framework-performance",
                "framework",
                "result",
                "framework",
                "Performance of each Framework",
                "1",
                "This is a plot of the performance of each framework for each task. Use this plot find the best framework for the tasks.",
                "bar"
            )}
            {self.graph_and_heading(
                self.current_results,
                "predict-duration-performance",
                "predict_duration",
                "result",
                "framework",
                "Predict Duration vs Performance",
                "2",
                "This is a scatter plot of the prediction duration against the performance of each framework for each task. Use this plot to find the best framework for the tasks.",
                "scatter"
            )}
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
            colors = px.colors.qualitative.Safe
            if len(x) == 0:
                return "<div></div>"

            # use plotly to create the plot
            if plot_type == "bar":
                fig = px.bar(
                    df,
                    x=x,
                    y=y,
                    color=color,
                    title=title,
                    color_discrete_sequence=colors,
                )
            elif plot_type == "scatter":
                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    color=color,
                    title=title,
                    color_discrete_sequence=colors,
                )

            fig.update_layout(
                title=title,
                xaxis_title=x,
                yaxis_title=y,
            )
            encoded_image = fig.to_html(full_html=False, include_plotlyjs="cdn")

            return f"<div style='grid-column: {grid_column};'>{encoded_image}</div>"
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
        return f'<div id="explanation-and-whats-next style="text-align: left; margin-bottom: 20px;">{markdown_response}</div>'

    def get_data_report(self):
        try:
            with open(
                f"{self.generated_data_reports_dir}/{self.dataset_id}_report.html", "r"
            ) as f:
                return f.read()
        except FileNotFoundError:
            return "<div><p>Feature importance not available for this dataset</p></div>"

    def generate_table_of_contents(self):
        return f"""
        <div class="container" style="margin-top: 20px; text-align: left;">
            <h1>Table of Contents</h1>
            <ul>
                <li><a href="#best-result">Best Result</a></li>
                <li><a href="#feature-importance">Feature Importance</a></li>
                <li><a href="#dashboard-section">Dashboard Section</a></li>
                <li><a href="#explanation-and-whats-next">Explanation and What's next?</a></li>
                <li><a href="#framework-table">Framework Table</a></li>
            </ul>
        </div>
        """
    def generate_download_current_page_button(self):
        return f"""
        <a href="report_{self.dataset_id}.html" download="report_{self.dataset_id}.html" class="btn btn-primary">Download Report</a>
        """

    def __call__(self):
        toc = self.generate_table_of_contents()
        dataset_info = self.generate_dataset_info()
        best_result_table = self.generate_best_result_table()
        framework_table = self.generate_framework_table()
        dashboard_section = self.generate_dashboard_section()
        explanation = self.get_explanation_from_llm()
        feature_importance = self.get_data_report()
        # download_button = self.generate_download_current_page_button()
        combined_html = f"""
    <!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">

<!-- jQuery library -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>

<!-- Latest compiled JavaScript -->
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
<body>
    <div class="container">
        {dataset_info}
        {toc}
        <div id="best-result">
        {best_result_table}
        </div>
        <div id="feature-importance">
        {feature_importance}
        </div>
        <div id="dashboard-section">
        {dashboard_section}
        </div>
        <div id="explanation-and-whats-next">
        <h1>Explanation and What's next?</h1>
        <p>!!! This is an AI-generated (llama3.2) explanation of the results. Please take the response with a grain of salt and use your own judgement.</p>
        {explanation}
        </div>
        <div id="framework-table">
        {framework_table}
        </div>

        </div>
    </body>
        """

        with open(
            Path(self.generated_final_reports_dir) / f"report_{self.dataset_id}.html",
            "w",
        ) as f:
            f.write(combined_html)


def run_report_script_for_dataset(
    GENERATED_DATA_REPORT_DIR, GENERATED_REPORTS_DIR, dataset_id
):
    # collect all the results from the runs
    collector = ResultCollector()
    all_results = collector()
    drg = DataReportGenerator(GENERATED_DATA_REPORT_DIR)
    try:
        # generate the data report for all datasets
        drg.generate_data_report_for_dataset(dataset_id=dataset_id)
        # write complete report to a file
        GenerateCompleteReportForDataset(
            dataset_id=dataset_id,
            collector_results=all_results,
            GENERATED_DATA_REPORT_DIR=GENERATED_DATA_REPORT_DIR,
            GENERATED_REPORTS_DIR=GENERATED_REPORTS_DIR,
        )()
    except Exception as e:
        print(f"Error generating report for dataset {dataset_id}: {str(e)}")
