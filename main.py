# %%

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
from visualization import DatasetAutoMLVisualizationGenerator
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# os.makedirs("generated_reports",exist_ok=True)

# Directory to save generated HTML pages
GENERATED_REPORTS_DIR = Path("./generated_reports")
GENERATED_REPORTS_DIR.mkdir(exist_ok=True)
# Initialize FastAPI
app = FastAPI()

# Initialize DatasetAutoMLVisualizationGenerator
visualization_generator = DatasetAutoMLVisualizationGenerator()


# Function to run every hour
def scheduled_run_info():
    visualization_generator.get_all_run_info()


scheduled_run_info()

# Schedule the job
scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(scheduled_run_info, IntervalTrigger(hours=1))

# # Dash app instance is created once and mounted globally.
dash_app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix="/dash/",
    eager_loading=True,
)
app.mount("/dash", WSGIMiddleware(dash_app.server))


def generate_all_reports():
    """Generate reports for dataset IDs from 1 to 100 and save them as HTML files."""
    for q in tqdm(range(1, 101)):  # Dataset IDs from 1 to 100
        # Path for the report file
        report_file_path = GENERATED_REPORTS_DIR / f"dataset_{q}.html"

        # Skip if the report already exists
        if report_file_path.exists():
            continue

        # Fetch data for the given dataset_id
        dataset_results = visualization_generator.all_results[
            visualization_generator.all_results["dataset_id"] == q
        ]

        if dataset_results.empty:
            # Log or handle missing dataset results
            print(f"No results found for dataset_id {q}")
            continue

        # Generate Dash layout
        try:
            dash_app.layout = visualization_generator.dash_app_layout(dataset_results)

            # Save the generated layout's HTML to the file
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(dash_app.index())
            print(f"Generated report for dataset_id {q}")
        except Exception as e:
            print(f"Error generating report for dataset_id {q}: {e}")


@app.on_event("startup")
async def on_startup():
    """Pre-generate all reports at startup."""
    generate_all_reports()


@app.get("/automlbplot", response_class=HTMLResponse)
async def automl_plot(q: Optional[int] = Query(None, description="Dataset ID")):
    """Serve the pre-generated HTML file for the requested dataset_id."""
    if q is None:
        return HTMLResponse(
            content="Error: 'q' (dataset_id) query parameter is required",
            status_code=400,
        )

    # Path for the report file
    report_file_path = GENERATED_REPORTS_DIR / f"dataset_{q}.html"

    # Serve the pre-generated file if it exists
    if report_file_path.exists():
        with open(report_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)

    # If the file is missing
    return HTMLResponse(
        content=f"No pre-generated report found for dataset_id {q}",
        status_code=404,
    )
