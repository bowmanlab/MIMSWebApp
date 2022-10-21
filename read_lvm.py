
import numpy as np
import pandas as pd
import glob
import plotly.graph_objs as go
import plotly.io as pio
from scipy import stats
import os
from datetime import datetime, timedelta

#!!! To do: mv the files out of Dropbox to harddrive at some frequency.  Skip calibration files.  Don't
## hard code header length.

#%%% Switch for transitioning between dev machine (windows) and production
## machine (Linux)

development = False
use_sccoos = True

if development == True:
    path_mims = 'C://Users//jeff//Documents//bowman_lab//MIMS//MIMS_Data//'
    path_edna = 'C://Users//jeff//Documents//bowman_lab//MIMS//Apps//Pier-Sampler-Data//'
    
else:  
    path_mims = '/home/jeff/Dropbox/MIMS_Data_v3/'  # use your path
    path_edna = '/home/jeff/Dropbox/Apps/Pier-Sampler-Data/'  # use your path

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

def plot_layout(name, ylab):
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
                text='Date',
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

def plot_trace(data, paramx, paramy, name, data_filter = None):
    
    ## Allow the index to be the x variable.
    
    if paramx == 'index':
        paramx = data.index
    else:
        paramx = data[paramx]
        
    try:
        if data_filter == None:
            data_filter = [True] * len(paramx)
    except ValueError:
        pass
    
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

csv_files = glob.glob(path_mims + "*.csv")
edna_log_files = glob.glob(path_edna + 'PierSamplerData-*.log')
edna_event_log_files = glob.glob(path_edna + 'PierSamplerEventLog-*.log')

#%% SCCOOS temperature data

## Note that SCCOOS temperature data is recorded in UTC 

base = 'https://erddap.sccoos.org/erddap/tabledap/autoss.csv?station%2Ctime%2Ctemperature&station=%22scripps_pier%22&time%3E=2018-01-01&temperature_flagPrimary=1&orderBy(%22time%22)'
sccoos_temp = pd.read_csv(base, skiprows = [1], index_col = 'time')

## Unused code for just getting a single day
    
#current_time = pd.to_datetime('today').tz_localize('America/Los_Angeles').tz_convert('UTC')
#current_day = current_time.strftime('%Y-%m-%d')

#base = 'https://erddap.sccoos.org/erddap/tabledap/autoss.csv?station%2Ctime%2Ctemperature&station=%22scripps_pier%22&time%3E=' + current_day + '&temperature_flagPrimary=1&orderBy(%22time%22)'

#sccoos_temp = pd.read_csv(base, skiprows = [1], index_col = 'time')

sccoos_temp.index = pd.to_datetime(sccoos_temp.index, format = '%Y-%m-%dT%H:%M:%SZ')

#%% MIMS data

col_str = ["time", "ms", "Water", "N2", "O2", "Ar", "Inlet Temperature", "Vacuum Pressure"]

## Iterate across all csv files parse, adding to list.

li = []

for filename in csv_files:
    
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
                                
    if development == True:
        base_name = filename.split('\\')[-1]
    else:
        base_name = filename.split('/')[-1]
    
    df = pd.read_csv(filename, skiprows=30, header=0, names = col_str, index_col = False)
    df['elapsed_time'] = df['time']
    
    for index, row in df.iterrows():
        time_delta = list(map(int, row['time'].split(':')))
        df.loc[index, 'time'] = date_time_0 + timedelta(hours = time_delta[0], minutes = time_delta[1], seconds = time_delta[2])

    #df['time'] = pd.to_datetime(df['time'], format = '%m/%d/%Y %I:%M:%S %p', exact = True)
    df['source_file'] = base_name
    df['start_time'] = date_time_0
      
    li.append(df)
    
## Concatenate individual dataframes to master frame and sort by date.

frame = pd.concat(li, axis=0, ignore_index=True)
sort = frame.sort_values(by = 'time', ascending = True)
sort.replace([np.inf, -np.inf], np.nan, inplace=True) 
sort['O2:Ar'] = sort['O2']/sort['Ar']
sort['N2:Ar'] = sort['N2']/sort['Ar']

## Round SCCOOS to 5 minute intervals and calculate %O2/%Ar at sat

sccoos_temp_round = sccoos_temp[['temperature']]
sccoos_temp_round['date_time'] = sccoos_temp.index.round('5T')
sccoos_temp_round.drop_duplicates(subset = 'date_time', inplace = True)
sccoos_temp_round.index = sccoos_temp_round.date_time
sccoos_temp_round.drop(columns = 'date_time', inplace = True)

sccoos_temp_round['O2_sat'] = O2sat([33.5] * sccoos_temp_round.shape[0], sccoos_temp_round['temperature'])
sccoos_temp_round['Ar_sat'] = Arsat([33.5] * sccoos_temp_round.shape[0], sccoos_temp_round['temperature'])
sccoos_temp_round['O2:Ar_sat'] = sccoos_temp_round['O2_sat'] / sccoos_temp_round['Ar_sat']

if use_sccoos == True:
    sort_round = sort[['time', 'O2', 'O2:Ar', 'N2:Ar']]
    sort_round.loc['time'] = sort.time.round('5T')
    sort_round = sort_round.groupby(sort_round.time).mean()
    
    edna_mims_round = pd.concat([sccoos_temp_round, sort_round], axis = 1, join = 'inner')
    
else:

## It looks like the easiest way to join the MIMS and eDNA datasets is to round
## both to nearest minute, eliminate duplicates, and glue together.
    
#!!! Currently this block is not in use, as only applies if SCCOOS is not being used for T data.

    edna_log_df_round = edna_log_df[['date_time', 'O2_sat', 'O2:Ar_sat']]
    edna_log_df_round['date_time'] = edna_log_df_round.date_time.round('min')
    edna_log_df_round.drop_duplicates(subset = 'date_time', inplace = True)
    edna_log_df_round.index = edna_log_df_round.date_time
    
    sort_round = sort[['date_time', 'O2:Ar', 'N2:Ar']]
    sort_round['date_time'] = sort.date_time.round('min')
    sort_round.drop_duplicates(subset = 'date_time', inplace = True)
    sort_round.index = sort_round.date_time
    
    edna_mims_round = pd.concat([edna_log_df_round, sort_round], axis = 1, join = 'inner')
    edna_mims_round.drop(columns = 'date_time', inplace = True)

## Derive a column filter based on N2:Ar values which should only vary
## during calibration or if something is very wrong.  Note that these do
## actually vary over time, so probably you'll have to adjust this at some
## point.

edna_mims_round_col_filter = edna_mims_round['N2:Ar'] > 0
#edna_mims_round_col_filter = (edna_mims_round['N2:Ar'] > 9) & (edna_mims_round['N2:Ar'] < 20) & (edna_mims_round.index > pd.to_datetime('2021-01-01', format = '%Y-%m-%d', exact = True))

## O2 correction - correction factor derived from calibrations with aged water.
## This value is calculated as O2_cf = (O2*/Ar*)/(O2/Ar), where * are the theoretical
## values at saturation, and the other values are the measured values at saturation.
## The O2_cf value given here is the mean of all values derived during calibrations.

#O2_cf = 2.24 # prior to 20 May 2021, after this date 1.5

edna_mims_round.loc[edna_mims_round.index < pd.to_datetime('2021-03-26 12:00:00'), 'O2_CF'] = 1.44 # New inlet after 26 March
edna_mims_round.loc[(edna_mims_round.index >= pd.to_datetime('2021-03-26 12:00:00')) & (edna_mims_round.index < pd.to_datetime('2021-05-20 12:00:00')), 'O2_CF'] = 2.24
edna_mims_round.loc[(edna_mims_round.index >= pd.to_datetime('2021-05-20 12:00:00')) & (edna_mims_round.index < pd.to_datetime('2021-08-6 12:00:00')), 'O2_CF'] = 1.5
edna_mims_round.loc[edna_mims_round.index >= pd.to_datetime('2021-08-6 12:00:00'), 'O2_CF'] = 2.0
edna_mims_round.loc[edna_mims_round.index >= pd.to_datetime('2022-02-12 12:00:00'), 'O2_CF'] = 1.54

## calculate [O2]bio.  Units are umol L-1

edna_mims_round['o2_bio'] = ((edna_mims_round['O2:Ar'] * edna_mims_round['O2_CF']) / edna_mims_round['O2:Ar_sat'] - 1) * edna_mims_round['O2_sat']

## Plot [O2]bio

trace1 = plot_trace(edna_mims_round, 'index', 'o2_bio', '[O2]bio', edna_mims_round_col_filter)
data = [trace1]
layout = plot_layout('[O<sub>2</sub>]<sub>bio</sub> - TESTING', '[O<sub>2</sub>]<sub>bio</sub> (micromolar)') ## Testing in plot title.
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'O2_bio' + ".html", auto_open=False)

#%% Create plots.

## SCCOOS results in fewer datapoints because the temperature time steps are coarser.

mims_col_filter = sort['N2:Ar'] > 0
#mims_col_filter = (sort['N2:Ar'] > 9) & (sort['N2:Ar'] < 20)
#mims_col_filter[0:-20000] = False
            
for col in ['O2', 'Ar', 'Inlet Temperature', 'Vacuum Pressure', 'N2','O2:Ar', 'N2:Ar', 'Water']:
    
    ## filter outliers based on z-score
    
    #col_filter = np.abs(stats.zscore(sort.loc[:,col], nan_policy = 'omit')) < 3
    
    ## Limit to about a months worth of data.  If you load the full dataset
    ## the website loads pretty slow and the plots are difficult to work with.
    
    #col_filter[0:-10000] = False
    
    trace1 = plot_trace(sort, 'time', col, '', mims_col_filter)
    data = [trace1]
    layout = plot_layout(col + ' - TESTING', col) ## Testing in plot title.
    fig = go.Figure(data=data, layout=layout)
    pio.write_html(fig, file= 'ecoobs/' + col.replace(':', '_') + ".html", auto_open=False)
    
frame.to_csv('MIMS_data_vol2_test.csv.gz')
edna_mims_round.to_csv('o2bio_test.csv')
