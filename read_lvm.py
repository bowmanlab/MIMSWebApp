import numpy as np
import pandas as pd
import glob
import plotly.graph_objs as go
import plotly.io as pio
from scipy import stats
import os
from datetime import datetime, timedelta
import shutil
import codecs
import re

#!!! Don't hard code header length.

#%%% Switch for transitioning between dev machine (windows) and production
## machine (Linux)

development = False
use_sccoos = True

if development == True:
    path_mims = 'C://Users//jeff//Documents//bowman_lab//MIMS//MIMS_Data_v3//'
    path_ctd = 'C://Users//jeff//Documents//bowman_lab//MIMS//CTD_Data_v1//'
    path_suna = 'C://Users//jeff//Documents//bowman_lab//MIMS//SUNAV2_Data_v1//'
    
    data_store = 'C://Users//jeff//Documents//bowman_lab//MIMS//data_store//'
    data_store_ctd = 'C://Users//jeff//Documents//bowman_lab//CTD//data_store//'
    data_store_suna = 'C://Users//jeff//Documents//bowman_lab//SUNA//data_store//'
    
else:  
    path_mims = '/home/jeff/Dropbox/MIMS_Data_v3/'  # use your path
    path_ctd = '/home/jeff/Dropbox/CTD_Data_v1/'
    path_suna = '/home/jeff/Dropbox/SUNAV2_Data_v1/'
    
    data_store = '/home/jeff/data_store/'
    data_store_ctd = '/home/jeff/data_store_CTD/'
    data_store_suna = '/home/jeff/data_store_SUNA/'

#%%% Functions

## Define function to calculate Ar at saturation based on Hamme and Emerson, 2004

def Arsat(S, T):
    
    TS = np.log(np.subtract(298.15,T) / np.add(273.15,T))
    
    A0 = 2.7915
    A1 = 3.17609
    A2 = 4.13116
    A3 = 4.90379
    B0 = -6.96233 * 10 ** -3
    B1 = -7.66670 * 10 ** -3
    B2 = -1.16888 * 10 ** -2
    
    Ar = np.exp(A0 + A1*TS + A2*np.power(TS,2) + A3*np.power(TS,3) + S*(B0 + B1*TS + B2*np.power(TS,2)))
    
    ## final units are umol kg-1
    
    return(Ar)

## Define function to calculate O2 at saturation based on Garcia and Gordon, 1992

def O2sat(S, T):
    
    TS = np.log(np.subtract(298.15,T) / np.add(273.15,T))
    
    A0 = 5.80818
    A1 = 3.20684
    A2 = 4.11890
    A3 = 4.93845
    A4 = 1.01567
    A5 = 1.41575
    B0 = -7.01211 * 10 ** -3
    B1 = -7.25958 * 10 ** -3
    B2 = -7.93334 * 10 ** -3
    B3 = -5.54491 * 10 ** -3
    C0 = -1.32412 * 10 ** -7
    
    O2 = np.exp(A0 + A1*TS + A2*np.power(TS,2) + A3*np.power(TS,3) + A4*np.power(TS,4) + A5*np.power(TS,5) + S*(B0 + B1*TS + B2*np.power(TS,2) + B3*np.power(TS,3) + C0*np.power(S,2)))
    
    ## final units are umol kg-1
    
    return(O2)

## Function for general layout for plots.

def plot_layout(name, ylab, xlab='Date'):
    layout = go.Layout(
        plot_bgcolor='#f6f7f8',
        paper_bgcolor='#f6f7f8',
        title=go.layout.Title(
            text=name,
            xref='paper',
            font=dict(
                family='Open Sans, sans-serif',
                size=22, 
                color='#000000'
            )
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text=xlab,
                font=dict(
                    family='Open Sans, sans-serif',
                    size=18,
                    color='#000000'
                )
            )
        ),
        yaxis=go.layout.YAxis(
            showexponent='all',
            exponentformat='e',
            title=go.layout.yaxis.Title(
                text=ylab,
                font=dict(
                    family='Open Sans, sans-serif',
                    size=18,
                    color='#000000'
                )
            )
        )
    )
    
    return(layout)

## Function for line trace for plots.

def plot_trace(data, paramx, paramy, name, data_filter = None, length_limit = 1000):
    
    ## Allow the index to be the x variable.
    
    if paramx == 'index':
        paramx = data.index
    else:
        paramx = data[paramx]
        
    try:
        if data_filter == None:
            data_filter = [False] * len(paramx)
            data_filter[-length_limit:] = [True] * len(data_filter[-length_limit:])
    except ValueError:
        inv_length_limit = len(data_filter) - length_limit
        data_filter[0:inv_length_limit] = [False] * inv_length_limit
    
    trace = go.Scatter(
    x = paramx[data_filter],
    y = data.loc[:, paramy][data_filter],
    xaxis='x1',
    yaxis='y1',
    marker=go.scatter.Marker(
        color='rgb(26, 118, 255)'
    ),
    line_shape='spline',
    line_smoothing=0,
    name = name
    )
    
    return(trace)

#%% Set things up

## Older data files will be moved to data_store. Create the directory if
## not present.
    
if not os.path.isdir(data_store):
    os.makedirs(data_store)
    
#%% SUNA data
    
## Note that SUNA data are UTC

suna_files = glob.glob(path_suna + "*.sbslog")
suna_files.sort(key = lambda x: os.path.getmtime(x))

suna_col_str = ['nitrate_uM', 'nitrate_mg', 'source_file']

try:
    suna_old_frame = pd.read_csv('SUNAV2_data_vol1.csv.gz', index_col = 0)
    suna_old_frame.index = pd.to_datetime(suna_old_frame.index, format = '%Y-%m-%d %H:%M:%S', utc = True)
except FileNotFoundError:
    suna_old_frame = pd.DataFrame(columns = suna_col_str)
    
li = [suna_old_frame]

## Iterate across the sbslog files.

old_files = set(suna_old_frame.source_file)
suna_new_data = pd.DataFrame(columns = suna_col_str)

for filename in suna_files:
    
    if development == True:
        base_name = filename.split('\\')[-1]
    else:
        base_name = filename.split('/')[-1]
        
    nitrate_um = []
    nitrate_mg = []
    
    if base_name not in old_files:
        try:
            with open(filename, 'r') as file_in:
                for line in file_in:
                    if line.startswith('<?xml'):
                        names = re.findall('<Name>[^<]*</Name>', line)
                    elif line.startswith('SATSLF1921'):
                        line = line.strip()
                        line = line.rstrip()
                        line = line.split(',')
                        
                        year = pd.to_datetime(line[1][0:4], format = '%Y', utc = True)
                        date = year + timedelta(float(line[1][-3:]) - 1)
                        date_time = date + timedelta(float(line[2])/24)
                        
                        ## It's a little silly to parse dates for all lines and only use the last date, but it works.
                        
                        nitrate_um.append(float(line[3]))
                        nitrate_mg.append(float(line[4]))

            print('adding', base_name)                     
            suna_new_data.loc[date_time] = pd.Series([np.average(nitrate_um), np.average(nitrate_mg), base_name], index = suna_col_str)  
            
        except NameError:
            
            ## NameError happens if there isn't a valid data line in file.  Currently, the SUNA is creating
            ## two types of log files but it isn't clear why this is happening.
            
            continue
                    
li.append(suna_new_data)                              
suna_frame = pd.concat(li, axis = 0, ignore_index=False)
suna_frame.sort_index(ascending = True, inplace = True)
        
#%% CTD data

ctd_files = glob.glob(path_ctd + "*.cnv")
ctd_files.sort(key = lambda x: os.path.getmtime(x))

## Iterate across all ctd files, parse, add to list. Start by reading the old
## combined data file, and create a new one if not present.

ctd_col_str = ['Conductivity [mS/cm]',
               'Density [sigma-theta, kg/m^3]',
               'Depth [salt water, m]',
               'Fluorescence [mg/m^3]',
               'Oxygen [umol/l]',
               'Oxygen [% saturation]',
               'Potential Temperature [ITS-90 deg C]',
               'Salinity [PSU]',
               'Temperature [ITS-90 deg C]',
               'Instrument Time [seconds]',
               'Instrument Time [julian days]',
               'flag']

try:
    old_frame = pd.read_csv('CTD_data_vol1.csv.gz', index_col = 0)
    old_frame.index = pd.to_datetime(old_frame.index, format = '%Y-%m-%d %H:%M:%S', utc = True)
except FileNotFoundError:
    old_frame = pd.DataFrame(columns = ctd_col_str)
    old_frame['source_file'] = []

li = [old_frame]

## Iterate across the csv files.

old_files = set(old_frame.source_file)

for filename in ctd_files:
    
    if development == True:
        base_name = filename.split('\\')[-1]
    else:
        base_name = filename.split('/')[-1]
    
    if base_name not in old_files:
        df = pd.DataFrame(columns = ctd_col_str)
        with codecs.open(filename, 'r', 'latin-1') as file_in:
            for line in file_in:
                
                ## Need correct year.
                
                if line.startswith('# start_time'):
                    line = line.split()
                    year = line[5]                    
                    year_start = pd.to_datetime(year, format = '%Y', utc = True)
    
                elif line.startswith('*') == False and line.startswith('#') == False:
                    line = line.rstrip()
                    line = line.strip()
                    line = line.split()
                    line = pd.Series(line, index = ctd_col_str)
                    
                    ## Straight julian decimal will put you one day ahead, must substract 1
                    
                    line_time = year_start + timedelta(float(line['Instrument Time [julian days]']) - 1)
                    df.loc[line_time] = line
        
        df['source_file'] = base_name
          
        li.append(df)
        print('adding', base_name)
    
## Concatenate individual dataframes to master frame and sort by date.

ctd_frame = pd.concat(li, axis=0, ignore_index=False)
ctd_frame.sort_index(ascending = True, inplace = True)
ctd_frame.replace([np.inf, -np.inf], np.nan, inplace=True) 

#%% MIMS data

csv_files = glob.glob(path_mims + "*.csv")
csv_files.sort(key = lambda x: os.path.getmtime(x))

col_str = ["time", "ms", "Water", "N2", "O2", "Ar", "Inlet Temperature", "Vacuum Pressure"]

## Iterate across all csv files parse, adding to list. Start by reading the old
## combined data file, and create a new one if not present.

try:
    old_frame = pd.read_csv('MIMS_data_vol2.csv.gz', index_col = 0)
except FileNotFoundError:
    old_frame = pd.DataFrame(columns = col_str)
    old_frame['source_file'] = []

old_frame['time'] = pd.to_datetime(old_frame['time'], format = '%Y-%m-%d %H:%M:%S')
li = [old_frame]

## Iterate across the csv files, but don't include the most recent because
## that creates problems for the instrument software. 

old_files = set(old_frame.source_file)

for filename in csv_files[0:-1]:
    
    if development == True:
        base_name = filename.split('\\')[-1]
    else:
        base_name = filename.split('/')[-1]
    
    if base_name not in old_files:
    
    ## Try clause added because Massoft occasionally starts exporting wrong number of columns
    ## and this needs to be fixed manually.  The first error this will raise is ValueError
    ## when pandas fails to parse date string.
    
        try:
        
            with open(filename, 'r') as file_in:
                for line in file_in.readlines():
                    if line.startswith('\"Date\"'):
                        line = line.strip()
                        line = line.split(',')
                        date = line[1]
                        time = line[3]
                        date_time_0 = ' '.join([date, time])
                        date_time_0 = pd.to_datetime(date_time_0, format = '%m/%d/%Y %I:%M:%S %p')
                        break
            
            df = pd.read_csv(filename, skiprows=30, header=0, names = col_str, index_col = False)
            df['elapsed_time'] = df['time']
            
            for index, row in df.iterrows():
                time_delta = list(map(int, row['time'].split(':')))
                df.loc[index, 'time'] = date_time_0 + timedelta(hours = time_delta[0], minutes = time_delta[1], seconds = time_delta[2])
        
            #df['time'] = pd.to_datetime(df['time'], format = '%m/%d/%Y %I:%M:%S %p', exact = True)
            df['source_file'] = base_name
            df['start_time'] = date_time_0
              
            li.append(df)
            print('adding', base_name)
            
        except ValueError:
            continue
    
## Concatenate individual dataframes to master frame and sort by date.

frame = pd.concat(li, axis=0, ignore_index=True)
sort = frame.sort_values(by = 'time', ascending = True)
sort.replace([np.inf, -np.inf], np.nan, inplace=True) 
sort['O2:Ar'] = sort['O2']/sort['Ar']
sort['N2:Ar'] = sort['N2']/sort['Ar']

## Round CTD to 5 minute intervals and calculate %O2/%Ar at sat.

ctd_temp_round = ctd_frame[['Temperature [ITS-90 deg C]', 'Salinity [PSU]', 'Oxygen [umol/l]', 'Oxygen [% saturation]', 'Fluorescence [mg/m^3]']]
ctd_temp_round.index = ctd_temp_round.index.tz_convert('US/Pacific')
ctd_temp_round.index = ctd_temp_round.index.tz_localize(None) # Must remove TZ info to match with MIMS
ctd_temp_round['date_time'] = ctd_temp_round.index.round('5T')
ctd_temp_round.drop_duplicates(subset = 'date_time', inplace = True)
ctd_temp_round.index = ctd_temp_round.date_time
ctd_temp_round.drop(columns = 'date_time', inplace = True)

## Change to numeric types here, should probably just change the
## dtype of the whole frame much earlier

ctd_temp_round['O2_sat'] = O2sat([33.5] * ctd_temp_round.shape[0], pd.to_numeric(ctd_temp_round['Temperature [ITS-90 deg C]']))
ctd_temp_round['Ar_sat'] = Arsat([33.5] * ctd_temp_round.shape[0], pd.to_numeric(ctd_temp_round['Temperature [ITS-90 deg C]']))
ctd_temp_round['O2:Ar_sat'] = ctd_temp_round['O2_sat'] / ctd_temp_round['Ar_sat']

## combine the CTD and MIMS datasets

sort_round = sort[['O2', 'O2:Ar', 'N2:Ar']]
sort_round.index = sort.time
sort_round['date_time'] = sort_round.index.round('5T')
sort_round = sort_round.groupby(sort_round.date_time).mean()

ctd_mims_round = pd.concat([ctd_temp_round, sort_round], axis = 1, join = 'inner')

## Derive a column filter based on N2:Ar values which should only vary
## during calibration or if something is very wrong.  Note that these do
## actually vary over time, so probably you'll have to adjust this at some
## point.

ctd_mims_round_col_filter = ctd_mims_round['N2:Ar'] > 0
#edna_mims_round_col_filter = (edna_mims_round['N2:Ar'] > 9) & (edna_mims_round['N2:Ar'] < 20) & (edna_mims_round.index > pd.to_datetime('2021-01-01', format = '%Y-%m-%d', exact = True))

## O2 correction - correction factor derived from calibrations with aged water.
## This value is calculated as O2_cf = (O2*/Ar*)/(O2/Ar), where * are the theoretical
## values at saturation, and the other values are the measured values at saturation.
## The O2_cf value given here is the mean of all values derived during calibrations.

#O2_cf = 2.24 # prior to 20 May 2021, after this date 1.5

ctd_mims_round.loc[ctd_mims_round.index.tz_localize(None) < pd.to_datetime('2021-03-26 12:00:00'), 'O2_CF'] = 1.44 # New inlet after 26 March
ctd_mims_round.loc[(ctd_mims_round.index.tz_localize(None) >= pd.to_datetime('2021-03-26 12:00:00')) & (ctd_mims_round.index.tz_localize(None) < pd.to_datetime('2021-05-20 12:00:00')), 'O2_CF'] = 2.24
ctd_mims_round.loc[(ctd_mims_round.index.tz_localize(None) >= pd.to_datetime('2021-05-20 12:00:00')) & (ctd_mims_round.index.tz_localize(None) < pd.to_datetime('2021-08-6 12:00:00')), 'O2_CF'] = 1.5
ctd_mims_round.loc[ctd_mims_round.index.tz_localize(None) >= pd.to_datetime('2021-08-6 12:00:00'), 'O2_CF'] = 2.0
ctd_mims_round.loc[ctd_mims_round.index.tz_localize(None) >= pd.to_datetime('2022-02-12 12:00:00'), 'O2_CF'] = 1.54
ctd_mims_round.loc[ctd_mims_round.index.tz_localize(None)>= pd.to_datetime('2022-11-17 12:00:00'), 'O2_CF'] = 1.76
ctd_mims_round.loc[ctd_mims_round.index.tz_localize(None) >= pd.to_datetime('2023-01-23 12:00:00'), 'O2_CF'] = 2.0

## calculate [O2]bio.  Units are umol L-1

ctd_mims_round['o2_bio'] = ((ctd_mims_round['O2:Ar'] * ctd_mims_round['O2_CF']) / ctd_mims_round['O2:Ar_sat'] - 1) * ctd_mims_round['O2_sat']

#!!! add AOU calculation here

#%% Create plots.

## Use ctd_temp_round for CTD data so that it plots when MIMS is down.

## Plot NO3

trace1 = plot_trace(suna_frame, 'index', 'nitrate_uM', 'Nitrate')
data = [trace1]
layout = plot_layout('Nitrate - TESTING', '<span>&#181;</span>M', 'Date (UTC)') ## Testing in plot title.
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'Nitrate' + ".html", auto_open=False)

## Plot [O2]bio

trace1 = plot_trace(ctd_mims_round, 'index', 'o2_bio', '[O2]bio', ctd_mims_round_col_filter)
data = [trace1]
layout = plot_layout('[O<sub>2</sub>]<sub>bio</sub> - TESTING', '[O<sub>2</sub>]<sub>bio</sub> <span>&#181;</span>M') ## Testing in plot title.
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'O2_bio' + ".html", auto_open=False)

## Plot temp

trace1 = plot_trace(ctd_temp_round, 'index', 'Temperature [ITS-90 deg C]', 'T deg C')
data = [trace1]
layout = plot_layout('Temperature - TESTING', 'T deg C') ## Testing in plot title.
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'Temperature' + ".html", auto_open=False)

## Plot salinity

trace1 = plot_trace(ctd_temp_round, 'index', 'Salinity [PSU]', 'PSU')
data = [trace1]
layout = plot_layout('Salinity - TESTING', 'PSU') ## Testing in plot title.
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'Salinity' + ".html", auto_open=False)

## Plot fluorescence

trace1 = plot_trace(ctd_temp_round, 'index', 'Fluorescence [mg/m^3]', 'mg/m^3')
data = [trace1]
layout = plot_layout('Chlorophyll - TESTING', 'mg m<sup>3</sup>') ## Testing in plot title.
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'Chlorophyll' + ".html", auto_open=False)

## Plot O2

trace1 = plot_trace(ctd_temp_round, 'index', 'Oxygen [umol/l]', 'umol/l')
data = [trace1]
layout = plot_layout('Dissolved Oxygen - TESTING', '<span>&#181;</span>M') ## Testing in plot title.
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'Oxygen' + ".html", auto_open=False)

## Plot O2 % Sat

trace1 = plot_trace(ctd_temp_round, 'index', 'Oxygen [% saturation]', '%')
data = [trace1]
layout = plot_layout('Dissolved Oxygen % Sat - TESTING', '%') ## Testing in plot title.
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'Oxygen_percent_sat' + ".html", auto_open=False)

## SCCOOS results in fewer datapoints because the temperature time steps are coarser.

mims_col_filter = sort['N2:Ar'] > 0
#mims_col_filter = (sort['N2:Ar'] > 9) & (sort['N2:Ar'] < 20)
mims_col_filter[0:-20000] = False
            
for col in ['O2', 'Ar', 'Inlet Temperature', 'Vacuum Pressure', 'N2','O2:Ar', 'N2:Ar', 'Water']:
    
    trace1 = plot_trace(sort, 'time', col, '', mims_col_filter, length_limit = 20000)
    data = [trace1]
    layout = plot_layout(col + ' - TESTING', col) ## Testing in plot title.
    fig = go.Figure(data=data, layout=layout)
    pio.write_html(fig, file= 'ecoobs/' + col.replace(':', '_') + ".html", auto_open=False)
    
#%% export data
   
frame.to_csv('MIMS_data_vol2.csv.gz')
ctd_mims_round.to_csv('o2bio_vol2.1.csv') ## vol 2.1 uses CTD for temp instead of SCCOOS, starts on May 18, 2023
ctd_frame.to_csv('CTD_data_vol1.csv.gz')
suna_frame.to_csv('SUNAV2_data_vol1.csv.gz')

#%% clean dropbox folder

## Clean the dropbox folder by moving all MIMS files that base name match to
## a csv file to data_store. This works because the csv files are created last,
## after the experimental files are no longer needed.

processed_files = set(frame.source_file)

for f in os.listdir(path_mims):
    f_base = f.split('.')[0]
    if f_base + '.csv' in processed_files:
        shutil.move(path_mims + f, data_store + f)
        
## Clean the dropbox folder by moving all CTD hex, cnv, and xmlcon files. The
## cnv file are only created by the Seabird Data Conversion utility after
## the run is complete so this should be safe.

processed_files = set(ctd_frame.source_file.str.split('.', expand = True)[0])
for f in os.listdir(path_ctd):
    f_base = f.split('.')[0]
    if f_base in processed_files:
        shutil.move(path_ctd + f, data_store_ctd + f)
        
## Clean the dropbox folder by moving all SUNA sbslog files. Files currently
## being written have "lock" in the extension.

processed_files = set(suna_frame.source_file.str.split('.', expand = True)[0])
for f in os.listdir(path_suna):
    f_base = f.split('.')[0]
    if f_base in processed_files:
        shutil.move(path_suna + f, data_store_suna + f)
