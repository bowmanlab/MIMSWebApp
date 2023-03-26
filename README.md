This reposistory consists of code to generate plots from our continuous-flow MIMS system and publish these plots as html files.  The html files are used in the real-time plots on ecoobs.ucsd.edu.

The basic procedure is:
* The MIMS system is controlled by MasSoft 10, which produces a text file located in a Dropbox folder.  A new data file is created approximately every 5 minutes.
* The plotting computer accesses the text file using Dropbox.
* A python script (read_lvm.py) is executed by a cron job on the plotting computer.  This script uses plotly to generate HTML files containing the plots.  The data files are moved outside of the Dropbox folder after being added to a continually growing dataframe.
* The plots are then copied using scp and a crontab to a public-facing server.
* ecoobs.ucsd.edu uses these plots as iframes.

