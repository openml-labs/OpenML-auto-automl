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
