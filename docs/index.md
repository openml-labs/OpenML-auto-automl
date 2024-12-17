# Automatic running of AutoML benchmark on OpenML datasets

- This project aims to run [AutoML benchmark](https://openml.github.io/automlbenchmark/) on OpenML datasets automatically. 
- Still a work in progress.

## Instructions
- One time steps
  - Clone the repository
  - Install the requirements
    - Install docker
    - Install pyenv
    - Make a virtual env with python 3.9.19
      - `pyenv install 3.9.19 && pyenv virtualenv 3.9.19 automl-benchmark && pyenv activate automl-benchmark`
  - `pip install -r requirements.txt`
- Update the automlbenchmark submodule if required using `./update_automlb.sh'
- Run the AutoMLRunner class with your chosen arguments
  - `python multi_run_benchmark.py --testing_mode=False`
  - Possible arguments:
    - testing_mode: If True, only the first 10 datasets will be used.
    - use_cache: If True, the dataset list will be cached and not redownloaded. Disable this if you want to fetch the latest dataset list.
    - run_mode: The mode in which the benchmarks will be run. Valid modes are: local, aws, docker, singularity. Default is docker.
    - num_tasks_to_return: The number of tasks to return for each dataset. Default is 1.
    - save_every_n_tasks: Save the run to the database after every n tasks. Default is 1.
- The results will be saved in the `data/results/` folder in the directory
  - `generated_data_reports` contains the data reports by dataset. eg: feature importance, class imbalance etc. These are stored as html files.
  - `generated_reports` contains the benchmark reports + data reports by dataset. These are stored as html files.
  - `results` contains the benchmark results with models/results and logs (as defined by automlbenchmark) in the respective folders.
  - `runs.db` is a sqlite database that stores a record of the runs and frameworks that were used.