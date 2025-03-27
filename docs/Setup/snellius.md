# Snellius Setup
- Now for Snellius
- As always, first ssh into Snellius `ssh user@snellius.surf.nl`
- This one is pretty involved, so make sure you have the time for it.

## Steps
- We need to first log into Github of course, if you haven't done so already run `git
config --global credential.helper cache` and then enter your details. Note that the
password is your access token. You can always use another authentication if you like.
    - I prefer to use the github command tool `gh` but we cannot install it here.
- Once that is done, go to your home directory and clone these repositories
    - `git clone https://github.com/openml-labs/OpenML-auto-automl`
    - `git clone https://github.com/SubhadityaMukherjee/automlbenchmark`
- In your home directory, do a `mkdir automl_data` . This should run automatically when
- Now just load Python and it's required modules so we can do the rest of the setup. You
can do this using `cd /home/user/OpenML-auto-automl && ./snellius_env_loader.sh` . If you
already built a conda environment, feel free to skip recreating it.
you execute it but just in case :P
- Now we need to build the Singularity (It is a docker alternative. Snellius only supports
  open source software so we use it) images for all the frameworks we want to run.
    - Since all of this is on the server, I prefer using `nvim` as my editor of choice.
    But if you are using VSCode or some such, feel free to just open it however you want.
    - cd to `automlbenchmark/scripts/`
    - `nohup python build_images.py -m singularity -f autosklearn,flaml,gama >
    build_log.txt 2>&1 &`
    - Of course, add whatever other frameworks you want to add.
    - If this does not work by any chance, you can also use `cd
    /home/user/OpenML-auto-automl && nohup ./build_images.sh user singularity >
    build_log.txt 2>&1 &`
    - This will take A LONG TIME to run and so the logs are stored in `build_log.txt`
    - Note that NOTHING else will work until these are built.
- Alright! Now that all of this is done (hopefully) you should be good to go.
- Do you want to set up the cron job? Move to [test server docs](./openml_test_server.md)
- Do you want to manually check if it is working? `cd
/home/user/OpenML-auto-automl/src && python snellius_generate_sbatch.py -c -a apikey`
- Results are stored in `/home/automl_data`

## FAQs
- How do I check if new datasets are being downloaded?
    - Go to `/home/user/automl_data` and check the file `dataset_list_for_cronjob.csv`. If the
      first few rows do not correspond to new data, something is wrong
- How do I know if the pipeline is working?
    - New dataset on OpenML -> New task + run on OpenML
    - There should be new slurm files in `/home/user/automl_data`
