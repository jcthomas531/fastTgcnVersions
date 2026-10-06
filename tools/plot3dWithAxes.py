import pyvista as pv



def plot3dWithAxes(filePath):
    mesh = pv.read(filePath)

    plotter = pv.Plotter()

    plotter.add_mesh(mesh)
    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        xtitle="X",
        ytitle="Y",
        ztitle="Z"
    )

    plotter.show()


# file1 = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMast/pre/pat001Pre_formCSOriMast.ply"
# plot3dWithAxes(file1)


