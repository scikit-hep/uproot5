import glob

import uproot4.reading

for filename in glob.glob("../uproot/tests/samples/*.root") + glob.glob("/home/pivarski/storage/data/**/*.root", recursive=True):
    file = uproot4.reading.ReadOnlyFile(filename)
    file.streamers

    print(file.source.num_requests, file.source.num_bytes - file.root_directory.fSeekKeys, file.source.num_bytes - file.fSeekInfo)


    # distance1 = len(file.source) - file.root_directory.fSeekKeys
    # distance2 = len(file.source) - file.fSeekInfo
    # distance3 = None
    # distance4 = None
    # for x in file.root_directory.keys:
    #     if x.fClassName == "TTree":
    #         if distance3 is None or x.fName.lower() == "events":
    #             distance3 = x.fSeekKey + x.fNbytes
    #             distance4 = len(file.source) - x.fSeekKey

    # print("TTreeBegin", distance3, "TTreeEnd", distance4, "TDirectoryEnd", distance1, "TStreamerInfoEnd", distance2)
