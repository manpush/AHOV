import plotly.graph_objects as go
fig = go.Figure(go.Scattermapbox(
    mode = "lines",
    lon = [10, 20],
    lat = [10,20],


    marker = {'size': 10}))
fig.update_layout(
    margin ={'l':0,'t':0,'b':0,'r':0},
    mapbox = {
        'center': {'lon': 10, 'lat': 10},
        'style': "stamen-terrain",
        'center': {'lon': 10, 'lat': 10},
        'zoom': 3})

fig.show()