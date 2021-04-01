# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os
import shutil
import struct

import numpy as np
import pytest
import skhep_testdata

import uproot

_file_header_fields_small = struct.Struct(">4siiiiiiiBiiiH16s")
_file_header_fields_big = struct.Struct(">4siiqqiiiBiqiH16s")

_directory_format_small = struct.Struct(">hIIiiiii")
_directory_format_big = struct.Struct(">hIIiiqqq")
_directory_format_num_keys = struct.Struct(">i")

_key_format_small = struct.Struct(">ihiIhhii")
_key_format_big = struct.Struct(">ihiIhhqq")

_string_size_format_1 = struct.Struct(">B")
_string_size_format_4 = struct.Struct(">I")

_free_format_1 = struct.Struct(">HII")
_free_format_2 = struct.Struct(">HQQ")


@pytest.mark.parametrize(
    "source", ["uproot-Zmumu.root", "uproot-issue243-new.root", "uproot-issue64.root"]
)
def test(tmp_path, source):
    print(source)

    # tmp_path = "/home/jpivarski/irishep/uproot4"

    filename = os.path.join(tmp_path, "update-me.root")
    shutil.copyfile(skhep_testdata.data_path(source), filename)

    with open(filename, "r+b") as rawfile:
        rawfile.seek(0)
        (
            magic,
            fVersion,
            fBEGIN,
            fEND,
            fSeekFree,
            fNbytesFree,
            nfree,
            fNbytesName,
            fUnits,
            fCompress,
            fSeekInfo,
            fNbytesInfo,
            fUUID_version,
            fUUID,
        ) = _file_header_fields_small.unpack(
            rawfile.read(_file_header_fields_small.size)
        )

        if fVersion > 1000000:
            rawfile.seek(0)
            (
                magic,
                fVersion,
                fBEGIN,
                fEND,
                fSeekFree,
                fNbytesFree,
                nfree,
                fNbytesName,
                fUnits,
                fCompress,
                fSeekInfo,
                fNbytesInfo,
                fUUID_version,
                fUUID,
            ) = _file_header_fields_big.unpack(
                rawfile.read(_file_header_fields_big.size)
            )

        rawfile.seek(fSeekFree)
        freesegments = read_key(rawfile)

        rawfile.seek(freesegments["fSeekKey"] + freesegments["fKeylen"])

        for _ in range(nfree):
            version, fFirst, fLast = _free_format_1.unpack(
                rawfile.read(_free_format_1.size)
            )
            if version > 1000:
                rawfile.seek(rawfile.tell() - _free_format_1.size)
                version, fFirst, fLast = _free_format_2.unpack(
                    rawfile.read(_free_format_2.size)
                )
            # print(f"{version = } {fFirst = } {fLast = } {fLast - fFirst = }")

        directory_start = fBEGIN + fNbytesName

        rawfile.seek(directory_start)
        (
            fVersion,
            fDatimeC,
            fDatimeM,
            fNbytesKeys,
            fNbytesName,
            fSeekDir,
            fSeekParent,
            fSeekKeys,
        ) = _directory_format_small.unpack(rawfile.read(_directory_format_small.size))

        if fVersion > 1000:
            rawfile.seek(directory_start)
            (
                fVersion,
                fDatimeC,
                fDatimeM,
                fNbytesKeys,
                fNbytesName,
                fSeekDir,
                fSeekParent,
                fSeekKeys,
            ) = _directory_format_big.unpack(rawfile.read(_directory_format_big.size))

        rawfile.seek(fSeekKeys)

        header_key = read_key(rawfile)
        print(header_key)

        (num_keys,) = _directory_format_num_keys.unpack(
            rawfile.read(_directory_format_num_keys.size)
        )
        print(num_keys)

        for _ in range(num_keys):
            key = read_key(rawfile)
            print(key)


def string(rawfile):
    (string_size,) = _string_size_format_1.unpack(rawfile.read(1))
    if string_size == 255:
        (string_size,) = _string_size_format_4.unpack(rawfile.read(4))
    return rawfile.read(string_size).decode(errors="surrogateescape")


def read_key(rawfile):
    position = rawfile.tell()

    (
        fNbytes,
        fVersion,
        fObjlen,
        fDatime,
        fKeylen,
        fCycle,
        fSeekKey,
        fSeekPdir,
    ) = _key_format_small.unpack(rawfile.read(_key_format_small.size))

    if fVersion > 1000:
        rawfile.seek(position)
        (
            fNbytes,
            fVersion,
            fObjlen,
            fDatime,
            fKeylen,
            fCycle,
            fSeekKey,
            fSeekPdir,
        ) = _key_format_big.unpack(rawfile.read(_key_format_big.size))

    fClassName = string(rawfile)
    fName = string(rawfile)
    fTitle = string(rawfile)

    return {
        "fNbytes": fNbytes,
        "fVersion": fVersion,
        "fObjlen": fObjlen,
        "fDatime": fDatime,
        "fKeylen": fKeylen,
        "fCycle": fCycle,
        "fSeekKey": fSeekKey,
        "fSeekPdir": fSeekPdir,
        "fClassName": fClassName,
        "fName": fName,
        "fTitle": fTitle,
    }
