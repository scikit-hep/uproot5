import uproot
import os
import pandas as pd
import pytest
import ROOT
import numpy as np

EPS = 1e-6


def test_saving_TGraph_to_file(tmp_path):
    newfile = os.path.join(tmp_path, "test_file.root")
    x = [i for i in range(10)]
    y = [i * 10 for i in range(10)]
    df = pd.DataFrame({"x": x, "y": y})

    with uproot.recreate(newfile) as f:
        f["myTGraph"] = uproot.to_TGraph(df)

    with uproot.open(newfile) as f:
        tgraph = f["myTGraph"]

        x_new = tgraph.values("x")
        y_new = tgraph.values("y")

        for i in range(len(x)):
            assert abs(x_new[i] - x[i]) < EPS
            assert abs(y_new[i] - y[i]) < EPS


def test_opening_TGraph_with_root(tmp_path):
    newfile = os.path.join(tmp_path, "test_file.root")
    x = [i for i in range(10)]
    y = [i * 10 for i in range(10)]
    df = pd.DataFrame({"x": x, "y": y})
    tGraphName = "myTGraph"
    title = "My TGraph"
    xLabel = "xLabel"
    yLabel = "yLabel"

    with uproot.recreate(newfile) as f:
        f[tGraphName] = uproot.to_TGraph(
            df, title=title, xAxisLabel=xLabel, yAxisLabel=yLabel
        )

    with ROOT.TFile.Open(newfile) as file:
        tgraph = file.Get(tGraphName)

        assert tgraph.GetTitle() == f"{title};{xLabel};{yLabel}"
        assert tgraph.GetXaxis().GetTitle() == xLabel
        assert tgraph.GetYaxis().GetTitle() == yLabel

        for i in range(len(x)):
            xAxis = tgraph.GetX()
            yAxis = tgraph.GetY()
            assert abs(xAxis[i] - x[i]) < EPS
            assert abs(yAxis[i] - y[i]) < EPS
