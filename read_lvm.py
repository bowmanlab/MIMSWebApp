
import numpy as np
import pandas as pd
import glob
import plotly.graph_objs as go
import plotly.io as pio
from scipy import stats

## Switch for transitioning between dev machine (windows) and production
## machine (Linux)

development = False

if development == True:
    path_mims = 'C://Users//jeff//Documents//bowman_lab//MIMS//MIMS_Data//'
    path_edna = 'C://Users//jeff//Documents//bowman_lab//MIMS//Apps//Pier-Sampler-Data//'
    
else:  
    path_mims = '/home/jeff/Dropbox/MIMS_Data/'  # use your path
    path_edna = '/home/jeff/Dropbox/Apps/Pier-Sampler-Data/'  # use your path

lvm_files = glob.glob(path_mims + "*.lvm")
edna_log_files = glob.glob(path_edna + 'PierSamplerData-*.log')
edna_event_log_files = glob.glob(path_edna + 'PierSamplerEventLog-*.log')

#### eDNA sampler data ####

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

trace1 = go.Scatter(
    x = edna_log_df['date_time'],
    y = edna_log_df.loc[:, 'temp_1_min_mean'],
    xaxis='x1',
    yaxis='y1',
    marker=go.scatter.Marker(
        color='rgb(26, 118, 255)'
    ),
    line_shape='spline',
    line_smoothing=0,
    name = 'Temperature'
)

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

layout = go.Layout(
    plot_bgcolor='#f6f7f8',
    paper_bgcolor='#f6f7f8',
    title=go.layout.Title(
        text='In situ Temperature',
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
            text='In situ temperature',
            font=dict(
                family='Open Sans, sans-serif',
                size=18,
                color='#000000'
            )
        )
    )
)

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

#### MIMS data ####

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

## Create plots.
            
for col in range(2, 18):
    
    ## filter outliers based on z-score
    
    col_filter = np.abs(stats.zscore(sort.iloc[:,col], nan_policy = 'omit')) < 3
    
    ## Limit to about a months worth of data.  If you load the full dataset
    ## the website loads pretty slow and the plots are difficult to work with.
    
    col_filter[0:-10000] = False
    
    trace1 = go.Scatter(
        x = sort['date_time'][col_filter],
        y = sort.iloc[:, col][col_filter],
        xaxis='x1',
        yaxis='y1',
        marker=go.scatter.Marker(
            color='rgb(26, 118, 255)'
        ),
        line_shape='spline',
        line_smoothing=0,
    )

    data = [trace1]

    layout = go.Layout(
        plot_bgcolor='#f6f7f8',
        paper_bgcolor='#f6f7f8',
        title=go.layout.Title(
            text=col_str[col],
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
                text=col_str[col],
                font=dict(
                    family='Open Sans, sans-serif',
                    size=18,
                    color='#000000'
                )
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    pio.write_html(fig, file= 'ecoobs/' + col_str[col] + ".html", auto_open=False)
    
frame.to_csv('MIMS_data_vol1.csv.gz')
