
import numpy as np
import pandas as pd
import glob
import plotly.graph_objs as go
import plotly.io as pio
from scipy import stats

#%%% Switch for transitioning between dev machine (windows) and production
## machine (Linux)

development = False

if development == True:
    path_mims = 'C://Users//jeff//Documents//bowman_lab//MIMS//MIMS_Data//'
    path_edna = 'C://Users//jeff//Documents//bowman_lab//MIMS//Apps//Pier-Sampler-Data//'
    
else:  
    path_mims = '/home/jeff/Dropbox/MIMS_Data/'  # use your path
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

## Define function to calculate O2 at saturation based on Hamme and Emerson, 2004

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

lvm_files = glob.glob(path_mims + "*.lvm")
edna_log_files = glob.glob(path_edna + 'PierSamplerData-*.log')
edna_event_log_files = glob.glob(path_edna + 'PierSamplerEventLog-*.log')

#%% eDNA sampler data 

## Iterate across all log files, parse, adding to list.

edna_logs = []

for log in edna_log_files:
    edna_logs_cols = ['date', 'time', 'epoch', 'temp_1_min_mean', 'temp_1_day_mean', 'flow_meter_count', 'flow_meter_hz', 'flow_L_min']
    df = pd.read_csv(log, names = edna_logs_cols, delim_whitespace = True, index_col = False)
    df['date_time'] = df['date'] + ' ' + df['time']
    df['date_time'] = pd.to_datetime(df['date_time'], format = '%Y-%m-%d %H:%M:%S', exact = True)
    
    edna_logs.append(df)
    
edna_log_df = pd.concat(edna_logs, axis=0, ignore_index = True)
edna_log_df.sort_values(by = 'date_time', ascending = True, inplace = True)

df = ''

edna_event_logs = []

for log in edna_event_log_files:
    edna_event_logs_cols = ['date', 'time', 'epoch', 'temp_1_sec_mean', 'temp_1_min_mean', 'temp_1_day_mean','flow_meter_count', 'flow_meter_hz', 'flow_L_min', 'filter_number']
    df = pd.read_csv(log, names = edna_event_logs_cols, delim_whitespace = True, index_col = False)
    df['date_time'] = df['date'] + ' ' + df['time']
    df['date_time'] = pd.to_datetime(df['date_time'], format = '%Y-%m-%d %H:%M:%S', exact = True)
    
    edna_event_logs.append(df)
    
edna_event_log_df = pd.concat(edna_event_logs, axis=0, ignore_index = True)
edna_event_log_df.sort_values(by = 'date_time', ascending = True, inplace = True)

## plot ##

trace1 = plot_trace(edna_log_df, 'date_time', 'temp_1_min_mean', 'Temperature')

edna_colors = ['#fbdad8', '#f6b6b0', '#f29189', '#ed6d61', '#e94b3c', '#e6301f', '#d12717', '#9d1d11', '#e6e93c']

trace2 = go.Scatter(
    x = edna_event_log_df['date_time'],
    y = edna_event_log_df.loc[:, 'temp_1_min_mean'],
    xaxis='x1',
    yaxis='y1',
    mode='markers',
    marker=dict(size=20, color = [edna_colors[x - 1] for x in edna_event_log_df.filter_number.tolist()]),
    name = 'eDNA sample'
)

data = [trace1, trace2]
layout = plot_layout('In situ temperature', 'In situ temperature')
fig = go.Figure(data=data, layout=layout)

fig.update_layout(legend_font = dict(
                family='Open Sans, sans-serif',
                size=18,
                color='#000000'),
    legend=dict(
        yanchor = 'top',
        xanchor = 'left',
        y = 0.99,
        x = 0.01)
)

pio.write_html(fig, file= 'ecoobs/' + 'In situ temperature' + ".html", auto_open=False)

#%% MIMS data

col_str = ["Empty", "Julian-Date", "Inlet Temperature", "Water", "N2", "O2", "Ar", "O2:Ar", "N2:Ar", "Total", "DMS-62",
           "DMS-47", "Bromoform-173", "Bromoform-171", "Bromoform-175", "Isoprene-67", "Isoprene-68", "Isoprene-53", "notes"]

## Iterate across all lvm files parse, adding to list.

li = []

for filename in lvm_files:
    
    if development == True:
        base_name = filename.split('\\')[-1]
    else:
        base_name = filename.split('/')[-1]

    year = base_name.split('_')[0][0:4]
    
    df = pd.read_csv(filename, sep='\t', skiprows=21, header=0, names = col_str, index_col = False)
    df['year'] = year
    df['source_file'] = base_name
    
    ## need to define hours, minutes, seconds independently here
    
    temp_time = pd.DataFrame(columns = ['day', 'day_decimal', 'hour', 'hour_decimal', 'minute', 'minute_decimal', 'seconds'])
    
    temp_time['day_decimal'] = df['Julian-Date']
    temp_time['day'] = temp_time['day_decimal'].apply(np.floor)
    temp_time['day_decimal'] = temp_time['day_decimal'] - temp_time['day']
    
    temp_time['hour_decimal'] = 24 * temp_time['day_decimal']
    temp_time['hour'] = temp_time['hour_decimal'].apply(np.floor)
    temp_time['hour_decimal'] = temp_time['hour_decimal'] - temp_time['hour']
    
    temp_time['minute_decimal'] = 60 * temp_time['hour_decimal']
    temp_time['minute'] = temp_time['minute_decimal'].apply(np.floor)
    temp_time['minute_decimal'] = temp_time['minute_decimal'] - temp_time['minute']
    
    temp_time['seconds'] = 60 * temp_time['minute_decimal'].astype(float)
    
    temp_time['date_time'] = year + '-' + temp_time['day'].astype(int).astype(str) + '-' + temp_time['hour'].astype(int).astype(str) + '-' + temp_time['minute'].astype(int).astype(str) + '-' + temp_time['seconds'].round(0).astype(int).astype(str)
        
    df['day'] = temp_time['day']
    df['date_time'] = pd.to_datetime(temp_time['date_time'], format = '%Y-%j-%H-%M-%S', exact = True)
    
    ## For reasons that aren't clear the last measurement of 2020 ends up in
    ## 2021 file.  This then shows up as day 366 for 2021.  Easiest way to deal
    ## with it is to just delete this measurement.
    
    if year == '2021':
        df.drop(df[df.day == 366].index, inplace = True)
    
    li.append(df)
    
## Concatenate individual dataframes to master frame and sort by date.

frame = pd.concat(li, axis=0, ignore_index=True)
sort = frame.sort_values(by = 'date_time', ascending = True)
sort.replace([np.inf, -np.inf], np.nan, inplace=True) 

## Calculate %O2/%Ar at sat

edna_log_df['O2_sat'] = O2sat([33.5] * edna_log_df.shape[0], edna_log_df['temp_1_min_mean'])
edna_log_df['Ar_sat'] = Arsat([33.5] * edna_log_df.shape[0], edna_log_df['temp_1_min_mean'])
edna_log_df['O2:Ar_sat'] = edna_log_df['O2_sat'] / edna_log_df['Ar_sat']

## It looks like the easiest way to join the MIMS and eDNA datasets is to round
## both to nearest minute, eliminate duplicates, and glue together.

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

edna_mims_round_col_filter = (edna_mims_round['N2:Ar'] > 11) & (edna_mims_round['N2:Ar'] < 20)

## O2 correction - correction factor derived from calibrations with aged water.
## This value is calculated as O2_cf = (O2*/Ar*)/(O2/Ar), where * are the theoretical
## values at saturation, and the other values are the measured values at saturation.
## The O2_cf value given here is the mean of all values derived during calibrations.

#O2_cf = 2.24 # prior to 20 May 2021, after this date 1.5

edna_mims_round.loc[edna_mims_round.index < pd.to_datetime('2020-04-15 12:00:00'), 'O2_CF'] = 2.24
edna_mims_round.loc[edna_mims_round.index >= pd.to_datetime('2020-04-15 12:00:00'), 'O2_CF'] = 1.5

## calculate [O2]bio.  Units are umol L-1

edna_mims_round['o2_bio'] = ((edna_mims_round['O2:Ar'] * edna_mims_round['O2_CF']) / edna_mims_round['O2:Ar_sat'] - 1) * edna_mims_round['O2_sat']

## Plot [O2]bio

trace1 = plot_trace(edna_mims_round, 'index', 'o2_bio', '[O2]bio', edna_mims_round_col_filter)
data = [trace1]
layout = plot_layout('[O<sub>2</sub>]<sub>bio</sub>', '[O<sub>2</sub>]<sub>bio</sub> (micromolar)')
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file= 'ecoobs/' + 'O2_bio' + ".html", auto_open=False)

## Create plots.

mims_col_filter = (sort['N2:Ar'] > 11) & (sort['N2:Ar'] < 20)
mims_col_filter[0:-20000] = False
            
for col in sort.columns[2:18]:
    
    ## filter outliers based on z-score
    
    #col_filter = np.abs(stats.zscore(sort.loc[:,col], nan_policy = 'omit')) < 3
    
    ## Limit to about a months worth of data.  If you load the full dataset
    ## the website loads pretty slow and the plots are difficult to work with.
    
    #col_filter[0:-10000] = False
    
    trace1 = plot_trace(sort, 'date_time', col, '', mims_col_filter)
    data = [trace1]
    layout = plot_layout(col, col)
    fig = go.Figure(data=data, layout=layout)
    pio.write_html(fig, file= 'ecoobs/' + col.replace(':', '_') + ".html", auto_open=False)
    
frame.to_csv('MIMS_data_vol1.csv.gz')
