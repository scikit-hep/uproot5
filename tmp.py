import sys
import glob

import uproot4.reading

numer = 0
denom = 0

# for filename in glob.glob("../uproot/tests/samples/*.root") + glob.glob("/home/pivarski/storage/data/**/*.root", recursive=True):
for filename in glob.glob(sys.argv[1]):
    file = uproot4.reading.ReadOnlyFile(filename)
    file.streamers

    if file.source.num_requests == 1:
        numer += 1
    denom += 1

    print(float(numer) / float(denom))
