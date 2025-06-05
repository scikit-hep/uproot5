from __future__ import annotations

from array import array

import numpy as np
from AsCustom import (
    ObjectHeaderReader,
)
from AsCustom.should_be_cpp import Cpp_BaseReader

bes3_branch2types = {
    "/Event:TMcEvent/m_mdcMcHitCol": "TMdcMc",
    "/Event:TMcEvent/m_cgemMcHitCol": "TCgemMc",
    "/Event:TMcEvent/m_emcMcHitCol": "TEmcMc",
    "/Event:TMcEvent/m_tofMcHitCol": "TTofMc",
    "/Event:TMcEvent/m_mucMcHitCol": "TMucMc",
    "/Event:TMcEvent/m_mcParticleCol": "TMcParticle",
    "/Event:TDigiEvent/m_mdcDigiCol": "TMdcDigi",
    "/Event:TDigiEvent/m_cgemDigiCol": "TCgemDigi",
    "/Event:TDigiEvent/m_emcDigiCol": "TEmcDigi",
    "/Event:TDigiEvent/m_tofDigiCol": "TTofDigi",
    "/Event:TDigiEvent/m_mucDigiCol": "TMucDigi",
    "/Event:TDigiEvent/m_lumiDigiCol": "TLumiDigi",
    "/Event:TDstEvent/m_mdcTrackCol": "TMdcTrack",
    "/Event:TDstEvent/m_emcTrackCol": "TEmcTrack",
    "/Event:TDstEvent/m_tofTrackCol": "TTofTrack",
    "/Event:TDstEvent/m_mucTrackCol": "TMucTrack",
    "/Event:TDstEvent/m_mdcDedxCol": "TMdcDedx",
    "/Event:TDstEvent/m_extTrackCol": "TExtTrack",
    "/Event:TDstEvent/m_mdcKalTrackCol": "TMdcKalTrack",
    "/Event:TRecEvent/m_recMdcTrackCol": "TRecMdcTrack",
    "/Event:TRecEvent/m_recMdcHitCol": "TRecMdcHit",
    "/Event:TRecEvent/m_recEmcHitCol": "TRecEmcHit",
    "/Event:TRecEvent/m_recEmcClusterCol": "TRecEmcCluster",
    "/Event:TRecEvent/m_recEmcShowerCol": "TRecEmcShower",
    "/Event:TRecEvent/m_recTofTrackCol": "TRecTofTrack",
    "/Event:TRecEvent/m_recMucTrackCol": "TRecMucTrack",
    "/Event:TRecEvent/m_recMdcDedxCol": "TRecMdcDedx",
    "/Event:TRecEvent/m_recMdcDedxHitCol": "TRecMdcDedxHit",
    "/Event:TRecEvent/m_recExtTrackCol": "TRecExtTrack",
    "/Event:TRecEvent/m_recMdcKalTrackCol": "TRecMdcKalTrack",
    "/Event:TRecEvent/m_recMdcKalHelixSegCol": "TRecMdcKalHelixSeg",
    "/Event:TRecEvent/m_recEvTimeCol": "TRecEvTime",
    "/Event:TRecEvent/m_recZddChannelCol": "TRecZddChannel",
    "/Event:TEvtRecObject/m_evtRecTrackCol": "TEvtRecTrack",
    "/Event:TEvtRecObject/m_evtRecVeeVertexCol": "TEvtRecVeeVertex",
    "/Event:TEvtRecObject/m_evtRecPi0Col": "TEvtRecPi0",
    "/Event:TEvtRecObject/m_evtRecEtaToGGCol": "TEvtRecEtaToGG",
    "/Event:TEvtRecObject/m_evtRecDTagCol": "TEvtRecDTag",
    "/Event:THltEvent/m_hltRawCol": "THltRaw",
}


class Cpp_Bes3TObjArrayReader(Cpp_BaseReader):
    """
    This class reads a TObjArray from a binary parser.

    I know that there is only 1 kind of class in the TObjArray I will read,
    so I can use only 1 reader to read all elements in TObjArray.
    """

    def __init__(self, name: str, element_reader: ObjectHeaderReader):
        """
        Args:
            element_reader (BaseReader): The reader for the elements in the array.
        """
        self.name = name
        self.element_reader = element_reader
        self.counts = array("Q")

    def read(self, parser):
        _ = parser.read_fNBytes()
        _ = parser.read_fVersion()
        _ = parser.read_fVersion()
        _ = parser.read_number("u4")  # fUniqueID
        _ = parser.read_number("u4")  # fBits

        # Just directly read data
        _ = parser.read_number("u1")  # fName
        fSize = parser.read_number("u4")
        _ = parser.read_number("u4")  # fLowerBound

        for _ in range(fSize):
            self.element_reader.read(parser)

        # Update offsets
        self.counts.append(fSize)

    def get_data(self):
        return (
            np.asarray(self.counts, dtype="Q"),
            self.element_reader.get_data(),
        )
