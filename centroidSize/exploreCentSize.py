import sys
import pyvista as pv
import numpy as np
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import readAndFormat as raf
import toothCentroids as toCe
import giveSurf


prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/pat001Post_formCSOriMastRemesh_seg.ply"

datPre = raf.readAndFormat(file = prePath, arch = "U")

tcPre = toCe.toothCentroids(face = datPre["face"], vertex = datPre["vert"])


#visualize
sPre = giveSurf.giveSurf(face = datPre["face"], vertex = datPre["vert"])

#all points in black
plotAll = pv.Plotter()
plotAll.add_mesh(sPre, scalars = "rgba", rgb = True)
plotAll.add_points(np.array(tcPre.iloc[:,range(1,4)]),
                    color = "black", point_size=10,
                    render_points_as_spheres=True)
#plotAll.show()


#separate color for each central centroid options
gumCent = tcPre.loc[tcPre["toothNum"] == "gum"].iloc[:,range(1,4)].copy()
allCent = tcPre.loc[tcPre["toothNum"] == "allScan"].iloc[:,range(1,4)].copy()
noGumCent = tcPre.loc[tcPre["toothNum"] == "noGum"].iloc[:,range(1,4)].copy()
justTeethCent = tcPre.loc[~tcPre["toothNum"].isin(["gum", "allScan", "noGum"])].iloc[:,range(1,4)].copy()

plotSep = pv.Plotter()
plotSep.add_mesh(sPre, scalars = "rgba", rgb = True)
plotSep.add_points(np.array(justTeethCent),
                    color = "black", point_size=10,
                    render_points_as_spheres=True)
plotSep.add_points(np.array(gumCent),
                    color = "red", point_size=10,
                    render_points_as_spheres=True)
plotSep.add_points(np.array(allCent),
                    color = "green", point_size=10,
                    render_points_as_spheres=True)
plotSep.add_points(np.array(noGumCent),
                    color = "blue", point_size=10,
                    render_points_as_spheres=True)
#plotSep.show()

#using noGumCent, plot lines from each tooth centroid
plotCentSize = pv.Plotter()
plotCentSize.add_mesh(sPre, scalars = "rgba", rgb = True, opacity = .75)
plotCentSize.add_points(np.array(justTeethCent),
                    color = "black", point_size=10,
                    render_points_as_spheres=True)
plotCentSize.add_points(np.array(noGumCent),
                    color = "blue", point_size=10,
                    render_points_as_spheres=True)
for pt in np.array(justTeethCent):
    plotCentSize.add_mesh(
        pv.Line(np.array(noGumCent), pt),
        color="black",
        line_width=2,
    )
plotCentSize.show()
#plotCentSize.export_html("K:/iowaExpTest/testDir/test1.html")




#distance between each tooth centriod and the central centroid
teethDat = tcPre.loc[~tcPre["toothNum"].isin(["gum", "allScan", "noGum"])].copy()
noGumDat = tcPre.loc[tcPre["toothNum"] == "noGum"].copy()

teethDat["xDistCent"] = teethDat["x"] - noGumDat["x"].iloc[0]
teethDat["yDistCent"] = teethDat["y"] - noGumDat["y"].iloc[0]
teethDat["zDistCent"] = teethDat["z"] - noGumDat["z"].iloc[0]

teethDat["l2Norm"] = np.linalg.norm(
    teethDat[["xDistCent", "yDistCent", "zDistCent"]],
    axis = 1,
    ord = 2
    ) 

centSize = np.linalg.norm(teethDat["l2Norm"], ord = 2)

aaa = 1.32965890035
bbb = 1.28828426
aaa - bbb
