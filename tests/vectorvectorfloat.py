import uproot
import skhep_testdata

def test():
    with uproot.open(
        skhep_testdata.data_path("uproot-vectorVectorDouble.root")
    ) as file:
        branch = file["t/x"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = True
        py = branch.array(interp,library="ak")
        py.show()
    return 0

if __name__ == "__main__":
    test()