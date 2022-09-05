# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines integer constants used by serialization and deserialization routines.
"""


import numpy

# determines when a file is "big"
kStartBigFile = 2000000000

# used in unmarshaling
kByteCountMask = numpy.int64(0x40000000)
kByteCountVMask = numpy.int64(0x4000)
kClassMask = numpy.int64(0x80000000)
kNewClassTag = numpy.int64(0xFFFFFFFF)

kIsOnHeap = numpy.uint32(0x01000000)
kNotDeleted = numpy.uint32(0x02000000)
kMustCleanup = numpy.uint32(1 << 3)
kIsReferenced = numpy.uint32(1 << 4)

kMapOffset = 2

# not used?
kNullTag = 0
kZombie = numpy.uint32(0x04000000)
kBitMask = numpy.uint32(0x00FFFFFF)
kDisplacementMask = numpy.uint32(0xFF000000)

############# core/zip/inc/Compression.h

kZLIB = 1
kLZMA = 2
kOldCompressionAlgo = 3
kLZ4 = 4
kZSTD = 5
kUndefinedCompressionAlgorithm = 6

############# constants for streamers

kBase = 0
kChar = 1
kShort = 2
kInt = 3
kLong = 4
kFloat = 5
kCounter = 6
kCharStar = 7
kDouble = 8
kDouble32 = 9
kLegacyChar = 10
kUChar = 11
kUShort = 12
kUInt = 13
kULong = 14
kBits = 15
kLong64 = 16
kULong64 = 17
kBool = 18
kFloat16 = 19
kOffsetL = 20
kOffsetP = 40
kObject = 61
kAny = 62
kObjectp = 63
kObjectP = 64
kTString = 65
kTObject = 66
kTNamed = 67
kAnyp = 68
kAnyP = 69
kAnyPnoVT = 70
kSTLp = 71

kSkip = 100
kSkipL = 120
kSkipP = 140

kConv = 200
kConvL = 220
kConvP = 240

kSTL = 300
kSTLstring = 365

kStreamer = 500
kStreamLoop = 501

############# constants from core/foundation/inc/ESTLType.h

kNotSTL = 0
kSTLvector = 1
kSTLlist = 2
kSTLdeque = 3
kSTLmap = 4
kSTLmultimap = 5
kSTLset = 6
kSTLmultiset = 7
kSTLbitset = 8
kSTLforwardlist = 9
kSTLunorderedset = 10
kSTLunorderedmultiset = 11
kSTLunorderedmap = 12
kSTLunorderedmultimap = 13
kSTLend = 14
kSTLany = 300

############# IOFeatures

kGenerateOffsetMap = numpy.uint8(1)

############# other

kStreamedMemberWise = numpy.uint16(1 << 14)

############ RNTuple https://github.com/root-project/root/blob/master/tree/ntuple/v7/doc/specifications.md
rntuple_col_num_to_dtype_dict = {
    1: "uint64",
    2: "uint32",
    3: "uint64",  # Switch
    4: "uint8",
    5: "uint8",  # char
    6: "bit",
    7: "float64",
    8: "float32",
    9: "float16",
    10: "int64",
    11: "int32",
    12: "int16",
    13: "int8",
    14: "uint32",  # SplitIndex64 delta encoding
    15: "uint64",  # SplitIndex32 delta encoding
    16: "float64",  # split
    17: "float32",  # split
    18: "float16",  # split
    19: "int64",  # split
    20: "int32",  # split
    21: "int16",  # split
}

rntuple_col_type_to_num_dict = {
    "index64": 1,
    "index32": 2,
    "switch": 3,
    "byte": 4,
    "char": 5,
    "bit": 6,
    "real64": 7,
    "real32": 8,
    "real16": 9,
    "int64": 10,
    "int32": 11,
    "int16": 12,
    "int8": 13,
    "splitindex64": 14,
    "splitindex32": 15,
    "splitreal64": 16,
    "splitreal32": 17,
    "splitreal16": 18,
    "splitin64": 19,
    "splitint32": 20,
    "splitint16": 21,
}

rntuple_role_leaf = 0
rntuple_role_vector = 1
rntuple_role_struct = 2
rntuple_role_union = 3
