This reposistory consists of code to generate plots from our continuous-flow MIMS system and publish these plots as html files.  The html files are used in the real-time plots on ecoobs.ucsd.edu.

The basic procedure is:
* The MIMS system is controlled by (LabView)[https://github.com/bowmanlab/MIMS], which produces a text file located in a Dropbox folder.  New data is written out approximately once per minute, and a new data file is generated every day.
* The plotting computer accesses the text file using Dropbox.
* A python script (read_lvm.py) is executed by a cron job.  This script uses plotly to generate HTML files containing the plots.
* The plots are then pushed to this repository by Git and a cron job.
* ecoobs.ucsd.edu uses these plots as iframes.

test
