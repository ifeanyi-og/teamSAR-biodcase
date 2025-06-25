## Verify that datasets look as expected

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path
np.random.seed(1)
import plotly.express as px

from data_preprocessing import run_preprocessing
from config import load_config

if __name__ == '__main__':
    config = load_config()
    run_preprocessing(config)

    a = pd.read_parquet('data/02_intermediate/preproc.parquet/part.1.parquet', engine='pyarrow')
    a = a[["path", "split", "label"]]
    print(a.head(20))


'''

print(a.columns.get_loc("features"))
b = a.iloc[2, 4]

print(type(b))
print(b.shape)

##

N = 100
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sz = np.random.rand(N) * 30

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    marker=go.scatter.Marker(
        size=sz,
        color=colors,
        opacity=0.6,
        colorscale="Viridis"
    )
))

#fig.show()


if not os.path.exists("images"):
    os.mkdir("images")

print("Saving image...")
# fig.write_image("images/fig1.png")
print("Saved image")


fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])

try:
    fig.write_image("images/test_plot.png")
    print("✅ Plotly image saved successfully.")
except Exception as e:
    print(f"❌ Plotly image export failed: {e}")
    
'''