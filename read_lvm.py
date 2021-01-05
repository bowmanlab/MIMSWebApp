
import numpy as np
import pandas as pd
import glob
import plotly.graph_objs as go
import plotly.io as pio
from scipy import stats

path = '/home/jeff/Dropbox/MIMS_Data'  # use your path
#path = 'C://Users//jeff//Documents//bowman_lab//MIMS//MIMS_Data'
all_files = glob.glob(path + "/*.lvm")

col_str = ["Empty", "Julian-Date", "Temperature", "Water", "N2", "O2", "Ar", "O2-Ar", "N2-Ar", "Total", "DMS-62",
           "DMS-47", "Bromoform-173", "Bromoform-171", "Bromoform-175", "Isoprene-67", "Isoprene-68", "Isoprene-53"]

## Iterate across all lvm files parse, adding to list.

li = []

for filename in all_files:
    
    base_name = filename.split('/')[-1]
#    base_name = filename.split('\\')[-1]
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

## Create plots.
            
for col in range(2, 18):
    
    ## filter outliers based on z-score
    
    col_filter = np.abs(stats.zscore(sort.iloc[:,col])) < 3
    
    trace1 = go.Scatter(
        x = sort['date_time'][col_filter],
        y = sort.iloc[:, col][col_filter],
        xaxis='x1',
        yaxis='y1',
        marker=go.scatter.Marker(
            color='rgb(26, 118, 255)'
        ),
        line_shape='spline',
        line_smoothing=1.3,
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
    pio.write_html(fig, file= col_str[col] + ".html", auto_open=False)
    
## placing this outside of the github repository to save space
    
frame.to_csv('../MIMS_data.csv.gz')
