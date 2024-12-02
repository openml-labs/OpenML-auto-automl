# %%

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
from visualization import DatasetAutoMLVisualizationGenerator
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import sqlite3
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template_string, request, jsonify
from flask_apscheduler import APScheduler
matplotlib.use('agg')


# Directory for generated reports
GENERATED_REPORTS_DIR = Path("./generated_reports")
GENERATED_REPORTS_DIR.mkdir(exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Initialize DatasetAutoMLVisualizationGenerator
visualization_generator = DatasetAutoMLVisualizationGenerator()
visualization_generator.get_all_run_info()

# Helper: Find max existing dataset ID
def find_max_existing_dataset_id():
    conn = sqlite3.connect("./data/runs.db")
    c = conn.cursor()
    c.execute("SELECT DISTINCT dataset_id FROM runs")
    rows = c.fetchall()
    conn.close()
    return max([x[0] for x in rows]) if rows else None

max_existing_dataset_id = find_max_existing_dataset_id()

# Scheduler Configuration
class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Scheduled job to fetch run info
def scheduled_run_info():
    visualization_generator.get_all_run_info()

scheduler.add_job(
    id="Fetch_Run_Info",
    func=scheduled_run_info,
    trigger="interval",
    hours=1,
)

# Helper: Render Seaborn/Matplotlib Plot
def render_plot(dataset_results):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=dataset_results,
        x="framework",
        y="result",
        hue="metric",
        palette="muted",
    )
    plt.title("Results by Framework and Metric")
    plt.tight_layout()

    # Save plot to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode()
    buffer.close()
    plt.close()

    return f"data:image/png;base64,{encoded_image}"

# Route for AutoML Plot
@app.route("/automlbplot")
def automl_plot():
    dataset_id = request.args.get("q", type=int)
    if dataset_id is None:
        return jsonify({"error": "'q' (dataset_id) query parameter is required"}), 400

    # Fetch data for the given dataset_id
    dataset_results = visualization_generator.all_results[
        visualization_generator.all_results["dataset_id"] == dataset_id
    ]

    if dataset_results.empty:
        return jsonify({"error": f"No results found for dataset_id {dataset_id}"}), 404

    # Generate plot and HTML
    image_src = render_plot(dataset_results)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AutoML Plot</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 20px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Results for Dataset ID: {dataset_id}</h1>
        <img src="{image_src}" alt="AutoML Results">
    </body>
    </html>
    """
    return render_template_string(html_content)

# Start Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)