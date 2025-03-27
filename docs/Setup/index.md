# Setup Overview

- The setup here is a little complicated due to the limitations of not having our own GPU
server (for now).
- Since we don't have our own GPUs (yet), we are using Snellius to run the
automl pipelines. Now, Snellius does not really want us to do that, hence not
allowing for cron jobs. To overcome this, we use the OpenML test server.

## Components
- Therefore to deploy the solution, we need to configure multiple things in
multiple places. This will be explained in this section.
- Look at [snellius](./snellius.md) first, then [test server](./openml_test_server.md)
