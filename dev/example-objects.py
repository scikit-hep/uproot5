"""
Use this to make an example ROOT file for src/uproot/models/TH.py and src/uproot/models/TGraph.py.
"""

import numpy as np
import ROOT

file = ROOT.TFile("example-objects.root", "recreate")

th1c = ROOT.TH1C("th1c", "title", 7, -3.14, 12.3)
th1c.Fill(0)
th1c.Fill(0)
th1c.Fill(-1000)
th1c.Fill(1000)
th1c.Write()
th1d = ROOT.TH1D("th1d", "title", 7, -3.14, 12.3)
th1d.Fill(0)
th1d.Fill(0)
th1d.Fill(-1000)
th1d.Fill(1000)
th1d.Write()
th1f = ROOT.TH1F("th1f", "title", 7, -3.14, 12.3)
th1f.Fill(0)
th1f.Fill(0)
th1f.Fill(-1000)
th1f.Fill(1000)
th1f.Write()
th1i = ROOT.TH1I("th1i", "title", 7, -3.14, 12.3)
th1i.Fill(0)
th1i.Fill(0)
th1i.Fill(-1000)
th1i.Fill(1000)
th1i.Write()
th1s = ROOT.TH1S("th1s", "title", 7, -3.14, 12.3)
th1s.Fill(0)
th1s.Fill(0)
th1s.Fill(-1000)
th1s.Fill(1000)
th1s.Write()

th2c = ROOT.TH2C("th2c", "title", 7, -3.14, 12.3, 3, -123.0, 999.0)
th2c.Fill(0, 0)
th2c.Fill(0, 0)
th2c.Fill(-1000, -1000)
th2c.Fill(1000, 1000)
th2c.Fill(1000, -1000)
th2c.Write()
th2d = ROOT.TH2D("th2d", "title", 7, -3.14, 12.3, 3, -123.0, 999.0)
th2d.Fill(0, 0)
th2d.Fill(0, 0)
th2d.Fill(-1000, -1000)
th2d.Fill(1000, 1000)
th2d.Fill(1000, -1000)
th2d.Write()
th2f = ROOT.TH2F("th2f", "title", 7, -3.14, 12.3, 3, -123.0, 999.0)
th2f.Fill(0, 0)
th2f.Fill(0, 0)
th2f.Fill(-1000, -1000)
th2f.Fill(1000, 1000)
th2f.Fill(1000, -1000)
th2f.Write()
th2i = ROOT.TH2I("th2i", "title", 7, -3.14, 12.3, 3, -123.0, 999.0)
th2i.Fill(0, 0)
th2i.Fill(0, 0)
th2i.Fill(-1000, -1000)
th2i.Fill(1000, 1000)
th2i.Fill(1000, -1000)
th2i.Write()
th2s = ROOT.TH2S("th2s", "title", 7, -3.14, 12.3, 3, -123.0, 999.0)
th2s.Fill(0, 0)
th2s.Fill(0, 0)
th2s.Fill(-1000, -1000)
th2s.Fill(1000, 1000)
th2s.Fill(1000, -1000)
th2s.Write()

th3c = ROOT.TH3C("th3c", "title", 7, -3.14, 12.3, 3, -123.0, 999.0, 2, -1.23, 9.99)
th3c.Fill(0, 0, 0)
th3c.Fill(0, 0, 0)
th3c.Fill(-1000, -1000, -1000)
th3c.Fill(1000, 1000, 1000)
th3c.Fill(1000, 1000, -1000)
th3c.Fill(1000, -1000, -1000)
th3c.Write()
th3d = ROOT.TH3D("th3d", "title", 7, -3.14, 12.3, 3, -123.0, 999.0, 2, -1.23, 9.99)
th3d.Fill(0, 0, 0)
th3d.Fill(0, 0, 0)
th3d.Fill(-1000, -1000, -1000)
th3d.Fill(1000, 1000, 1000)
th3d.Fill(1000, 1000, -1000)
th3d.Fill(1000, -1000, -1000)
th3d.Write()
th3f = ROOT.TH3F("th3f", "title", 7, -3.14, 12.3, 3, -123.0, 999.0, 2, -1.23, 9.99)
th3f.Fill(0, 0, 0)
th3f.Fill(0, 0, 0)
th3f.Fill(-1000, -1000, -1000)
th3f.Fill(1000, 1000, 1000)
th3f.Fill(1000, 1000, -1000)
th3f.Fill(1000, -1000, -1000)
th3f.Write()
th3i = ROOT.TH3I("th3i", "title", 7, -3.14, 12.3, 3, -123.0, 999.0, 2, -1.23, 9.99)
th3i.Fill(0, 0, 0)
th3i.Fill(0, 0, 0)
th3i.Fill(-1000, -1000, -1000)
th3i.Fill(1000, 1000, 1000)
th3i.Fill(1000, 1000, -1000)
th3i.Fill(1000, -1000, -1000)
th3i.Write()
th3s = ROOT.TH3S("th3s", "title", 7, -3.14, 12.3, 3, -123.0, 999.0, 2, -1.23, 9.99)
th3s.Fill(0, 0, 0)
th3s.Fill(0, 0, 0)
th3s.Fill(-1000, -1000, -1000)
th3s.Fill(1000, 1000, 1000)
th3s.Fill(1000, 1000, -1000)
th3s.Fill(1000, -1000, -1000)
th3s.Write()

tprofile = ROOT.TProfile("tprofile", "title", 7, -3.14, 12.3, 2.2, 22.2)
tprofile.Fill(0, 0, 5.5)
tprofile.Fill(0, 0, 5.5)
tprofile.Fill(0, 0, 5.5)
tprofile.Write()
tprofile2d = ROOT.TProfile2D(
    "tprofile2d", "title", 7, -3.14, 12.3, 3, -123.0, 999.0, 2.2, 22.2
)
tprofile2d.Fill(0, 0, 5.5)
tprofile2d.Fill(0, 0, 5.5)
tprofile2d.Fill(0, 0, 5.5)
tprofile2d.Write()
tprofile3d = ROOT.TProfile3D(
    "tprofile3d", "title", 7, -3.14, 12.3, 3, -123.0, 999.0, 2, -1.23, 9.99
)
tprofile3d.Fill(0, 0, 0, 5.5)
tprofile3d.Fill(0, 0, 0, 5.5)
tprofile3d.Fill(0, 0, 0, 5.5)
tprofile3d.Write()

x = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
y = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
xup = np.array([0.2, 0.1, 0.1, 0.1, 0.1])
xdown = np.array([0.1, 0.2, 0.1, 0.1, 0.1])
yup = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
ydown = np.array([0.1, 0.1, 0.1, 0.2, 0.1])

tgraphasymmerrors = ROOT.TGraphAsymmErrors(5, x, y, xup, xdown, yup, ydown)
tgraphasymmerrors.SetName("tgraphasymmerrors")
tgraphasymmerrors.SetTitle("title")
tgraphasymmerrors.Write()

tgrapherrors = ROOT.TGraphErrors(5, x, y, xup, yup)
tgrapherrors.SetName("tgrapherrors")
tgrapherrors.SetTitle("title")
tgrapherrors.Write()

tgraph = ROOT.TGraph(5, x, y)
tgraph.SetName("tgraph")
tgraph.SetTitle("title")
tgraph.Write()

file.Close()
