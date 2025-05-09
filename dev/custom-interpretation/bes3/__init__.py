from __future__ import annotations

import awkward as ak
from AsCustom import (
    AsCustom,
    BaseReader,
    ReaderType,
    gen_reader_config,
    get_reader_instance,
    readers,
    reconstruct_array,
)

from .should_be_cpp import Cpp_Bes3TObjArrayReader

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


class Bes3TObjArrayReader(BaseReader):
    @classmethod
    def gen_reader_config(
        cls,
        top_type_name: str,
        cls_streamer_info: dict,
        all_streamer_info: dict,
        item_path: str = "",
    ):
        if top_type_name != "TObjArray":
            return None

        obj_typename = bes3_branch2types.get(item_path.replace(".TObjArray*", ""))
        if obj_typename is None:
            return None

        if obj_typename not in all_streamer_info:
            return {
                "reader": "MyTObjArrayReader",
                "name": cls_streamer_info["fName"],
                "element_reader": {
                    "reader": ReaderType.Empty,
                    "name": obj_typename,
                },
            }

        sub_reader_config = []
        for s in all_streamer_info[obj_typename]:
            sub_reader_config.append(
                gen_reader_config(s, all_streamer_info, item_path + f".{obj_typename}")
            )

        return {
            "reader": "MyTObjArrayReader",
            "name": cls_streamer_info["fName"],
            "element_reader": {
                "reader": ReaderType.ObjectHeader,
                "name": obj_typename,
                "sub_readers": sub_reader_config,
            },
        }

    @staticmethod
    def get_reader_instance(reader_config: dict):
        if reader_config["reader"] != "MyTObjArrayReader":
            return None

        element_reader_config = reader_config["element_reader"]
        element_reader = get_reader_instance(element_reader_config)

        return Cpp_Bes3TObjArrayReader(reader_config["name"], element_reader)

    @staticmethod
    def reconstruct_array(raw_data, reader_config: dict):
        if reader_config["reader"] != "MyTObjArrayReader":
            return None

        counts, element_raw_data = raw_data
        element_reader_config = reader_config["element_reader"]
        element_data = reconstruct_array(
            element_raw_data,
            element_reader_config,
        )

        return ak.unflatten(element_data, counts)


def register():
    readers.add(Bes3TObjArrayReader)
    AsCustom.target_branches |= set(bes3_branch2types.keys()) | {
        "/Event:EventNavigator/m_mcMdcMcHits",
        "/Event:EventNavigator/m_mcMdcTracks",
        "/Event:EventNavigator/m_mcEmcMcHits",
        "/Event:EventNavigator/m_mcEmcRecShowers",
    }
