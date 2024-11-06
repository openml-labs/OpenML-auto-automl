# Automatic running of AutoML benchmark on OpenML datasets

- This project aims to run [AutoML benchmark](https://openml.github.io/automlbenchmark/) on OpenML datasets automatically. 
- Still a work in progress.

## Instructions
- Install the requirements using `pip install -r requirements.txt`
- Run the AutoMLRunner class with your chosen arguments : `python multi_run_benchmark.py --testing_mode=False`
  - Possible arguments:
    - testing_mode: If True, only the first 10 datasets will be used.
    - use_cache: If True, the dataset list will be cached and not redownloaded. Disable this if you want to fetch the latest dataset list.
    - run_mode: The mode in which the benchmarks will be run. Valid modes are: local, aws, docker, singularity. Default is docker.
    - num_tasks_to_return: The number of tasks to return for each dataset. Default is 1.
    - save_every_n_tasks: Save the run to the database after every n tasks. Default is 1.
- The results will be saved in the `data/results/` folder in the directory 
- Then to start the dashboard server, run `uvicorn main:app --host 0.0.0.0 --port 8000` and go to `http://0.0.0.0:8000/automlbplot?q=dataset_id` to view the results for a particular dataset.
