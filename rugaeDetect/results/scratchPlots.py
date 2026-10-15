import pandas as pd
import pyvista as pv
import numpy as np

path01 = "K:/iowaExpTest/rugaePipeOptim/labeledDescriptors/pre/remesh8500/norm0.016_desc1.3/pat001Pre.csv"
dat01 = pd.read_csv(path01)

points01 = dat01[["x", "y", "z"]].to_numpy()
labels01 = dat01["label"].to_numpy()
plot01 = pv.Plotter()
plot01.add_points(
    points = points01,
    scalars = labels01,
    render_points_as_spheres = True,
    point_size = 20,
    categories = True,
    cmap = ["blue", "grey"]
)
plot01.show()