# Report Generator
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
import json
import markdown
import matplotlib
import openml
import pandas as pd
import plotly.express as px
from typing import Any
from dataclasses import dataclass
from utils import OpenMLTaskHandler, safe_load_file
from metric_info import MetricsFromAMLB

matplotlib.use("agg")
set_visualize_provider(InlineProvider())

## Result collector


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


## Dashboard
### Data summary table
class FeatureImportance:
    """
    Feature Importance using Explainable Boosting Classifier from InterpretML
    """

    def generate_ebm_report(self, names, scores):
        df = pd.DataFrame({"Feature": names, "Score": scores}).sort_values(
            by="Score", ascending=False
        )
        fig = px.bar(df, x="Score", y="Feature", orientation="h")
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Score",
            yaxis_title="Feature",
            width=800,
            height=800,
        )
        fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return f"""<div id="feature-importance" class="container" style="margin-top: 20px; text-align: left">{fig_html}</div>"""

    def run_ebm_on_dataset(self, dataset_id, X_train, y_train):
        try:
            ebm = ExplainableBoostingClassifier(random_state=42)
            ebm.fit(X_train, y_train)
            ebm_global = ebm.explain_global().data()
            names, scores = ebm_global["names"], ebm_global["scores"]
            return self.generate_ebm_report(names, scores)
        except Exception as e:
            print(f"Error running EBM on dataset {dataset_id}: {e}")
            return "<div>Unable to generate feature importance report</div>"


class DataOverviewGenerator:
    """
    This class does the following
    - Generates a data summary table for the dataset
    - Generates the Feature Importance report using Explainable Boosting Classifier
    """

    def __init__(self, template_dir="./website_assets/templates/"):
        self.template_dir = template_dir
        self.jinja_environment = Environment(loader=FileSystemLoader(template_dir))
        self.template_to_use = {"data_summary_table": "data_summary_table.html"}
        self.explainable_boosting = FeatureImportance()

    def get_data_and_split(self, dataset_id):
        dataset = openml.datasets.get_dataset(dataset_id=dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = pd.get_dummies(X, prefix_sep=".").astype(float)
        y, _ = y.factorize()
        return train_test_split(X, y, random_state=42)

    def generate_data_summary_table(self, X, max_cols=100):
        col_visualizations = {}
        # Get the first 100 columns if max_cols is not specified. This is to avoid generating too many visualizations.
        if max_cols is not None:
            cols = X.columns[:max_cols]

        # Generate visualizations for each column. Pie or histogram based on the number of unique values.
        self.generate_visualizations_for_each_column(X, col_visualizations, cols)

        try:
            visualization_row = pd.DataFrame([col_visualizations])
            missing_values = X.isnull().sum().to_frame().T
            missing_values.index = ["Missing Values"]
            X_preview = X.head(10)

            table_data = pd.concat(
                [visualization_row, missing_values, X_preview], ignore_index=True
            )
            row_names = ["Visualization", "Missing Values"]
            # Assign the 'Headers' column directly
            table_data["Headers"] = row_names + [""] * (
                len(table_data) - len(row_names)
            )

            # Rearrange columns to move 'Headers' to the front
            table_data = table_data[
                ["Headers"] + [col for col in table_data.columns if col != "Headers"]
            ]

            headers_html = "".join(f"<th>{col}</th>" for col in table_data.columns)
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                for row in table_data.values
            )

            return (
                f"""
        <div class="container" style="overflow-x:auto;">
            <h2>Data Summary Table</h2>
            <input type="text" id="searchBar" placeholder="Search columns..." style="margin-bottom:10px; width:100%;">
            <table id="dataTable" class="table table-striped table-bordered" style="display:none;">
                <thead><tr>{headers_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            <button id="prevBtn">Previous</button>
            <button id="nextBtn">Next</button>
        </div>"""
                + """
        <script>
            const table = document.getElementById('dataTable');
            const searchBar = document.getElementById('searchBar');
            const cols = table.querySelectorAll('thead th');
            const rows = table.querySelectorAll('tbody tr');
            const totalCols = cols.length;
            const colsPerPage = 10;
            let startCol = 0;

            function updateTable() {
                cols.forEach((col, i) => col.style.display = i >= startCol && i < startCol + colsPerPage ? '' : 'none');
                rows.forEach(row => {
                    Array.from(row.children).forEach((cell, i) => cell.style.display = i >= startCol && i < startCol + colsPerPage ? '' : 'none');
                });
                document.getElementById('prevBtn').disabled = startCol === 0;
                document.getElementById('nextBtn').disabled = startCol + colsPerPage >= totalCols;
            }

            searchBar.addEventListener('input', () => {
                const query = searchBar.value.toLowerCase();
                cols.forEach((col, i) => {
                    if (col.textContent.toLowerCase().includes(query)) {
                        col.style.display = '';
                        rows.forEach(row => {
                            row.children[i].style.display = '';
                        });
                    } else {
                        col.style.display = 'none';
                        rows.forEach(row => {
                            row.children[i].style.display = 'none';
                        });
                    }
                });
            });

            document.getElementById('prevBtn').addEventListener('click', () => {
                if (startCol > 0) startCol -= colsPerPage;
                updateTable();
            });

            document.getElementById('nextBtn').addEventListener('click', () => {
                if (startCol + colsPerPage < totalCols) startCol += colsPerPage;
                updateTable();
            });

            table.style.display = '';
            updateTable();
        </script>
        """
            )
        except Exception as e:
            print(f"Error generating data summary table: {e}")
            return "<div>Unable to generate data summary table</div>"

    def generate_visualizations_for_each_column(self, X, col_visualizations, cols):
        for col in cols:
            try:
                X[col] = X[col].astype(str)
                unique_values = len(X[col].unique())

                if unique_values < 10:
                    fig = px.histogram(X, x=col)
                else:
                    value_counts = X[col].value_counts()
                    if len(value_counts) > 2:
                        value_counts = pd.concat(
                            [
                                value_counts[:2],
                                pd.Series([value_counts[2:].sum()], index=["Other"]),
                            ]
                        )
                    fig = px.pie(names=value_counts.index, values=value_counts.values)

                fig.update_layout(
                    title=None,
                    width=300,
                    height=300,
                )
                col_visualizations[col] = fig.to_html(
                    full_html=False, include_plotlyjs="cdn"
                )
            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                col_visualizations[col] = ""

    def generate_complete_report(self, dataset_id):
        try:
            X_train, _, y_train, _ = self.get_data_and_split(dataset_id)
            ebm_report = self.explainable_boosting.run_ebm_on_dataset(
                dataset_id, X_train, y_train
            )
            data_summary_table = self.generate_data_summary_table(X_train)
            return data_summary_table, ebm_report
        except Exception as e:
            print(f"Error generating report for dataset {dataset_id}: {e}")
            return None, None

### Best result table
class BestResult:
    """This generates the Best result table for the dashboard"""

    def __init__(
        self, current_results, metrics_info, jinja_environment, template_to_use
    ):
        self.current_results = current_results
        self.metrics_info = metrics_info
        self.jinja_environment = jinja_environment
        self.template_to_use = template_to_use
        self.best_framework = ""
        self.best_metric = ""
        self.type_of_task = ""
        self.dataset_id = ""
        self.task_id = ""
        self.task_name = ""
        self.best_result_for_metric = ""
        self.description = ""
        self.metric_and_result = ""

    def get_best_result(self):
        """
        This function returns the best result from the current_results DataFrame.
        It first sorts the DataFrame based on the metric used and then returns the best result.
        """
        if self.current_results is None:
            return None

        metric_used = self.current_results["metric"].iloc[0]
        sort_in_ascending_order = True  # Default to ascending order

        # Determine sorting order based on metrics_info
        for category, metrics in self.metrics_info.items():
            if metric_used in metrics:
                sort_in_ascending_order = (
                    metrics[metric_used]["better_value"] == "lower"
                )
                self.metric_description = metrics[metric_used]["description"]
                break

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

        # all metric columns that are in the dataframe and in the metrics_info
        metric_columns = [
            col
            for col in self.current_results.columns
            if any(col in metrics for metrics in self.metrics_info.values())
        ]

        self.all_metrics_present = []
        for metric in metric_columns:
            try:
                self.all_metrics_present.append(self.current_results[metric].values[0])
            except:
                pass

        self.metric_and_result = " ".join(
            [
                f"The {metric} is {result} "
                for metric, result in zip(metric_columns, self.all_metrics_present)
            ]
        )

    def generate_best_result_table(self):
        """
        This function generates the best result table using the best result information.
        """
        template = self.jinja_environment.get_template(
            self.template_to_use["best_result"]
        )
        try:
            return template.render(
                best_framework=self.best_framework,
                best_metric=self.best_metric,
                type_of_task=self.type_of_task,
                dataset_id=self.dataset_id,
                task_id=self.task_id,
                task_name=self.task_name,
                best_metric_explanation = self.metric_description,
            )
        except Exception as e:
            print(f"Error generating best result table: {e}")
            return "<div>Unable to generate best result table</div>"


### LLM Explanation
class LLMExplanation:
    def __init__(self, best_result: BestResult, model="llama3.2", temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.best_result = best_result

    def get_explanation_from_llm(self):
        """
        Based on information obtained from the AutoML systems and OpenML generate an explanation using LLM.
        """

        prompt_format = f"""For a dataset called {self.best_result.task_name} , the best framework is {self.best_result.best_framework} with a {self.best_result.best_metric} of {self.best_result.best_result_for_metric}. This is a {self.best_result.type_of_task} task. The results are as follows {self.best_result.metric_and_result}. For each metric, tell me if this is a good score (and why), and if it is not, how can I improve it? Keep your answer to the point.
        The dataset description is: {self.best_result.description}
        """
        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_format,
                },
            ],
            options={
                "temperature": self.temperature,
            },
        )
        response = response["message"]["content"]
        response = markdown.markdown(response)

        return f"""<div
            style="text-align: left; margin-bottom: 20px"
            >
            <h1>Explanation and What's next?</h1>
            <p>
                !!! This is an AI-generated (llama3.2) explanation of the results.
                Please take the response with a grain of salt and use your own
                judgement.
            </p>
            {response}
            </div>"""


## Generate complete report for ds


@dataclass
class FrameworkProcessor:
    # def __init__(self, framework_name, process_fn):
    framework_name: str
    process_fn: Any


class GenerateCompleteReportForDataset:
    def __init__(
        self,
        dataset_id: int,
        collector_results,
        GENERATED_REPORTS_DIR: str = "../data/generated_reports",
        template_dir="./website_assets/templates/",
    ):
        self.dataset_id = int(dataset_id)
        self.collector_results = collector_results
        self.current_results = self.get_results_for_dataset_id(self.dataset_id)
        self.jinja_environment = Environment(loader=FileSystemLoader(template_dir))
        self.generated_final_reports_dir = GENERATED_REPORTS_DIR
        self.template_dir = template_dir
        self.template_to_use = {
            "dataset_info": "data_information.html",
            "best_result": "best_result_table.html",
            "framework_table": "framework_table.html",
            "metric_vs_result": "metric_vs_result.html",
        }
        self.framework_processor = FrameworkProcessor
        # all metrics that are in the dataframe
        self.metrics_info = MetricsFromAMLB().metrics_info

        # run the function to get the best result
        self.frameworks_that_support_extra_information = [
            "Auto-sklearn",
            "H20AutoML",
            "AutoGluon",
            "All results",
        ]

        self.framework_processors = [
            FrameworkProcessor("Auto-sklearn", self.process_auto_sklearn_data),
            FrameworkProcessor("H20AutoML", self.process_h2oautoml_data),
            FrameworkProcessor("AutoGluon", self.process_auto_gluon_data),
            FrameworkProcessor("All results", self.process_all_results_data),
        ]

        self.best_result = BestResult(
            self.current_results,
            self.metrics_info,
            self.jinja_environment,
            self.template_to_use,
        )
        # for all init in best result, add them to the current object
        self.best_result.get_best_result()

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

    def generate_dataset_info(self):
        """
        This function generates the dataset information table using the dataset information.
        """
        template = self.jinja_environment.get_template(
            self.template_to_use["dataset_info"]
        )
        return template.render(
            dataset_id=self.dataset_id,
            task_name=self.best_result.task_name,
        )

    def process_h2oautoml_data(self, current_results, top_n=10):
        # TODO
        return self.get_rows_for_framework_from_df(
            df=current_results, framework_name="H20AutoML", top_n=10
        )

    def process_auto_gluon_data(self, current_results, top_n=10):
        # TODO
        return self.get_rows_for_framework_from_df(
            df=current_results, framework_name="AutoGluon", top_n=10
        )

    def process_all_results_data(self, df, top_n=40):
        try:
            df = df.drop(columns="dataset_description", errors="ignore")
            return to_html_datatable(
                df,
                caption="Results by AutoML Framework",
                table_id="all-framework-results",
            )
        except Exception as e:
            print(e)
            return ""

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

    def get_rows_for_framework_from_df(
        self, df: pd.DataFrame, framework_name, top_n=40
    ):
        try:
            framework_rows = df[df["framework"] == framework_name]["models"].values[0]
            framework_data = safe_load_file(framework_rows, "pd")
            return to_html_datatable(
                framework_data.head(top_n), caption=f"{framework_name} Models"
            )
        except Exception as e:
            print(e)
            return ""

    def generate_framework_table(self):
        complete_html = ""
        for processor in self.framework_processors:
            try:
                complete_html += processor.process_fn(self.current_results)
            except Exception as e:
                print(e)
                continue

        return f"""
        <div class="container" style="margin-top: 20px; text-align: left;">
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
                self.best_result.best_metric.upper() + "-task",
                "task",
                "result",
                "framework",
                f"{self.best_result.best_metric.upper()} of each Framework",
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

            return f"<div style='margin-top: 20px; text-align: left grid-column: {grid_column};'>{encoded_image}</div>"
        except Exception as e:
            print(e)
            return f"<div style='margin-top: 20px; text-align: left grid-column: {grid_column};'><p>Error generating graph: {str(e)}</p></div>"

    def __call__(self):
        report_path = (
            Path(self.generated_final_reports_dir) / f"report_{self.dataset_id}.html"
        )
        if report_path.exists():
            return
        data_overview = DataOverviewGenerator(
            self.template_dir
        )
        data_summary_table, ebm_report = data_overview.generate_complete_report(
            self.dataset_id
        )
        if data_summary_table is None or ebm_report is None:
            return

        dataset_info = self.generate_dataset_info()
        best_result_table = self.best_result.generate_best_result_table()
        framework_table = self.generate_framework_table()
        dashboard_section = self.generate_dashboard_section()
        # explanation = LLMExplanation(
        #     best_result=self.best_result
        # ).get_explanation_from_llm()

        combined_html = self.jinja_environment.get_template(
            "complete_page.html"
        ).render(
            dataset_info=dataset_info,
            best_result_table=best_result_table,
            framework_table=framework_table,
            dashboard_section=dashboard_section,
            # explanation=explanation,
            ebm_report=ebm_report,
            data_summary_table=data_summary_table,
        )

        with open(report_path, "w") as f:
            f.write(combined_html)


## run report


def run_report_script_for_dataset(
    GENERATED_REPORTS_DIR, dataset_id, result_path, template_dir
):
    # collect all the results from the runs
    collector = ResultCollector(result_path)
    all_results = collector()
    # drg = DataOverviewGenerator(template_dir)
    try:
        # generate the data report for all datasets
        # drg.generate_complete_report(dataset_id=dataset_id)
        # write complete report to a file
        report_gen = GenerateCompleteReportForDataset(
            dataset_id=dataset_id,
            collector_results=all_results,
            GENERATED_REPORTS_DIR=GENERATED_REPORTS_DIR,
            template_dir=template_dir,
        )
        report_gen()
    except Exception as e:
        print(f"Error generating report for dataset {dataset_id}: {str(e)}")
