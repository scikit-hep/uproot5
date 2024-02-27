# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import uproot


def test_split_ranges_if_large_file_in_http():
    fname = (
        "https://xrootd-local.unl.edu:1094//store/user/AGC/nanoAOD/TT_TuneCUETP8M1_13TeV"
        "-powheg-pythia8/cmsopendata2015_ttbar_19980_PU25nsData2015v1_76X_mcRun2_asymptotic"
        "_v12_ext3-v1_00000_0000.root"
    )

    arrays_to_read = [
        "Jet_mass",
        "nJet",
        "Muon_pt",
        "Jet_phi",
        "Jet_btagCSVV2",
        "Jet_pt",
        "Jet_eta",
    ]

    f = uproot.open(
        fname, handler=uproot.source.http.HTTPSource, http_max_header_bytes=21786
    )
    assert f.file.options["http_max_header_bytes"] == 21786

    f["Events"].arrays(arrays_to_read)
