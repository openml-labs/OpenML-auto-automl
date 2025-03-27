# OpenML Auto-AutoML

- This repository is under a project called AutoML4All, an initiative to provide AutoML tools to anyone without programming experience.
- At the moment, this is focused on only Tabular data.
- If anyone uploads a dataset to [OpenML](https://openml.org), we perform these steps every n (here 5) hours
    - Check if there are new datasets based on what was previously stored
    - Identify if there is a [Task](https://openml.github.io/openml-python/main/usage.html#key-concepts), if not, then try to create one based on the target variable and data type of the target column.
    - Once a task is created, summon [amlb](https://github.com/openml/automlbenchmark) and based on the chosen automl frameworks, send a request to the GPU servers
    - For now, we are using [Snellius](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius)
    - Once the frameworks are done running, upload the run results back to OpenML.

## How to use?
- If this project has not shut down, you should just be able to access it by uploading a dataset to OpenML and checking back in a few hours.

## Developer configuration
- Hello future OpenML developer, so you want to re-run/make a new version of this? Just look at the rest of this documentation.

## Contribution
- A rather important note is that unless you are a core developer, it might be a little
hard to contribute to this particular project BUT since a large part of this project depends more on
[amlb](https://github.com/openml/automlbenchmark), in all
  honesty, it makes sense to focus contributions there more than this particular library.
