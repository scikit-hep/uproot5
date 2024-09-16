# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines known forth code for types it is known a priori.
"""
from __future__ import annotations

from uproot.interpretation.known_forth.atlas.element_link import VectorVectorElementLink

KNOWN_FORTH_DICT = {
    "std::vector<std::vector<ElementLink<DataVector<xAOD::CaloCluster_v1>>>>": VectorVectorElementLink,
    "std::vector<std::vector<ElementLink<DataVector<xAOD::Vertex_v1>>>>": VectorVectorElementLink,
    "std::vector<std::vector<ElementLink<DataVector<xAOD::TruthParticle_v1>>>>": VectorVectorElementLink,
    "std::vector<std::vector<ElementLink<DataVector<xAOD::TrackParticle_v1>>>>": VectorVectorElementLink,
    "std::vector<std::vector<ElementLink<DataVector<xAOD::TauTrack_v1>>>>": VectorVectorElementLink,
    "std::vector<std::vector<ElementLink<DataVector<xAOD::IParticle>>>>": VectorVectorElementLink,
    "std::vector<std::vector<ElementLink<DataVector<xAOD::NeutralParticle_v1>>>>": VectorVectorElementLink,
}


def known_forth_of(model):
    if not hasattr(model, "typename"):
        return

    typename = model.typename

    if typename not in KNOWN_FORTH_DICT:
        return

    return KNOWN_FORTH_DICT[typename](typename)
