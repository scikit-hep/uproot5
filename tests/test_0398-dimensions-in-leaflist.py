# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE


import numpy
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-398.root")) as file:
        assert [x.tolist() for x in file["orange/orange"].array(library="np")[0]] == [
            1,
            0.05928909033536911,
            0.0687987208366394,
            412.7960510253906,
            0.05928909033536911,
            412.7960510253906,
            0,
            1,
            0.9996464848518372,
            [65.1502685546875, -122.76300811767578, -153.02999877929688],
            21.341833114624023,
            0.9430077075958252,
            28.992263793945312,
            27.94194221496582,
            0.10401010513305664,
            1.1628128290176392,
            2.410203456878662,
            [27.68297004699707, 31.953433990478516, 31.953433990478516],
            [-9.4708890914917, 17.89554214477539, 92.4335708618164, 97.2898941040039],
            [
                -9.345060348510742,
                19.61220932006836,
                93.16610717773438,
                97.70188903808594,
            ],
            4.53577995300293,
            999.9000244140625,
            450.4504699707031,
            -0.011151830665767193,
            0.05031048506498337,
            449.5561218261719,
            0.08810456097126007,
            0.047824934124946594,
            408.0093994140625,
            0.08411796391010284,
        ]
