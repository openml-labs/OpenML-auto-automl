# Limitations -> Future Work

- So obviously, there are a few limitations of this library. These are listed here for
future reference.

## Snellius
- Using Snellius complicates this a lot since they dont' allow for cron jobs. But until we
  get our own GPU server, this has to suffice.

## Dataset Limitations
- Sometimes new data sets that are uploaded do not specify the correct target variable or
  have some metadata missing that does not allow us to automatically decide what to do
with it. Since there is currently no way to deal with that, we just ignore running the
framework for these datasets.

## MultiModal
- OpenML Currently does not have a tag that mentions what kind of data the current data
set is. For instance, if there is an image data set open ML currently does not upload all
the images directly and so what we get as a header file. As you can imagine, though,
running an auto ML pipeline on such a file is pointless. But at the moment there is no way
for us to detect it so we ignore that.
- Due to this, we also do not support specific types of tasks like time series prediction,
for example since there is currently no way for us to know if the dataset being
processed belongs to one of these categories.

## Framework Limitations
- Since most of the auto ML frameworks that we use in this library are externally
managed,We do not control how frequently they are updated. This means that sometimes
frameworks just straight out do not work maybe because they are out of date or they are no
longer maintained, etc.. While we will do our best to make sure something like this does
not happen, sometimes it is a bit unpredictable.
- For example at the time of writing this documentation, AutoGluon no longer supports
OpenML. The developers are of course working to fix this, but we do not know how long it
will take.
