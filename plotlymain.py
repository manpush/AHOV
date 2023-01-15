import plotly.graph_objs as go
import pandas as pd
import plotly.figure_factory as ff
df = pd.read_csv('hydro2.csv')
# https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv

import plotly.express as px

# df = px.data.gapminder().query("year == 2007")

fig = ff.create_hexbin_mapbox(
    data_frame=df,  lat='x', lon='y',
    nx_hexagon=30, opacity=0.5
    ,labels={"color": "val"}
    ,color='val'
    ,mapbox_style="stamen-terrain"
    ,color_continuous_scale=['white','white', 'yellow']
    ,range_color=[0,10]
)
# fig = px.density_mapbox(data_frame=df, lat='x', lon='y', z="lat", radius=35,
#                         center=dict(lat=0, lon=180), zoom=0,
#                         mapbox_style="stamen-terrain")
#fig.add_trace(go.Scattermapbox(mode="lines", hoverinfo='skip', lat=df['x'], lon=df['y']))
fig.show()
