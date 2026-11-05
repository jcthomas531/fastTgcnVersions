#original position
surf_prei = giveSurf.giveSurf(face = preDati["face"], vertex = preDati["vert"])
surf_posti = giveSurf.giveSurf(face = postDati["face"], vertex = postDati["vert"])
surf_rigidi = giveSurf.giveSurf(face = rigidi["face"], vertex = rigidi["vert"])

p = pv.Plotter()
p.add_mesh(surf_prei, color = "white")
p.add_mesh(surf_rigidi, scalars = "rgba", rgb = True,  opacity = .6)
p.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
    xtitle="X",
    ytitle="Y",
    ztitle="Z"
)
p.view_xy()
p.show()


#apply rigid transformation to pre dat then move all into quadrant 1 for measuring
#move far into quadrant 1
quadMove = np.array([
    [1,0,0,500],
    [0,1,0,500],
    [0,0,1,500],
    [0,0,0,1]
])
preDati_moved = applySpatialTrans(dat = preDati, trans = quadMove)
rigidi_moved = applySpatialTrans(dat = rigidi, trans = quadMove)

surf_prei_moved = giveSurf.giveSurf(face = preDati_moved["face"], vertex = preDati_moved["vert"])
surf_rigidi_moved = giveSurf.giveSurf(face = rigidi_moved["face"], vertex = rigidi_moved["vert"])


p = pv.Plotter()
p.add_mesh(surf_prei_moved, color = "white")
p.add_mesh(surf_rigidi_moved, scalars = "rgba", rgb = True,  opacity = .6)
p.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
    xtitle="X",
    ytitle="Y",
    ztitle="Z"
)
p.view_xy()
p.show()



