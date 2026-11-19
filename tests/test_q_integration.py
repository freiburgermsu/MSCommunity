"""Integration of the composition, MICOM, chemostat and dynamic additions into MSCommunity.

Covers what the merge itself added: the MSCommunity wrapper methods, the kinetic_coeff="calibrate"
constructor option, the package exports, and set_abundance inside `with model:` when a member's
stored biomass metabolite is a different object from the one in bio1 (as on the SynCom build) or
an abundance is zero.
"""
import os
import sys

import pytest
from cobra import Metabolite

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_l_members as T  # noqa: E402  (toy two-member community builder)

import mscommunity  # noqa: E402
from mscommunity.mscommsim import MSCommunity  # noqa: E402


def _abund(a_a):
    return {"memA": {"abundance": a_a}, "memB": {"abundance": 1 - a_a}}


@pytest.mark.parametrize("swap_objects", [False, True])
@pytest.mark.parametrize("zero_first", [False, True])
def test_set_abundance_inside_a_model_context_is_undone_exactly(swap_objects, zero_first):
    comm = T._community()
    if swap_objects:  # the SynCom build: same ids, different objects
        for mem in comm.members:
            mem.biomass_cpd = Metabolite(mem.biomass_cpd.id, compartment=mem.biomass_cpd.compartment)
    before = comm.primary_biomass.reaction
    abundances_before = {m.id: m.abundance for m in comm.members}
    with comm.util.model:
        if zero_first:
            comm.set_abundance(_abund(0.0))
        comm.set_abundance(_abund(0.3))
        coefs = {m.id: c for m, c in comm.primary_biomass.metabolites.items()}
        assert coefs["cpd11416_c1"] == pytest.approx(-0.3)
        assert coefs["cpd11416_c2"] == pytest.approx(-0.7)
        assert comm.util.model.slim_optimize() > 0
    assert comm.primary_biomass.reaction == before
    assert {m.id: m.abundance for m in comm.members} == abundances_before


def test_set_abundance_outside_a_context_still_writes_bio1():
    comm = T._community()
    comm.set_abundance(_abund(0.25))
    coefs = {m.id: c for m, c in comm.primary_biomass.metabolites.items()}
    assert coefs["cpd11416_c1"] == pytest.approx(-0.25)
    assert comm.abundances == {"memA": pytest.approx(0.25), "memB": pytest.approx(0.75)}


def _merged():
    from mscommunity.commhelper import build_from_species_models
    return build_from_species_models([T._member("memA"), T._member("memB", periplasm=True)],
                                     abundances={"memA": 0.5, "memB": 0.5})


def test_kinetic_coeff_calibrate_sets_the_shared_kappa_sat():
    kappa = max(T._community().calibrate_kappa(per_member=False).values())
    comm = MSCommunity(model=_merged(), ids=["memA", "memB"], kinetic_coeff="calibrate",
                       close_member_drains=True)
    assert comm.kinCoef == pytest.approx(kappa)
    for member in comm.members:
        row = comm.util.model.constraints[f"{member.id}_commKin"]
        coef = row.get_linear_coefficients([member.primary_biomass.forward_variable])
        assert coef[member.primary_biomass.forward_variable] == pytest.approx(-kappa)


def test_kinetic_coeff_rejects_an_unknown_string():
    with pytest.raises(ValueError):
        MSCommunity(model=_merged(), ids=["memA", "memB"], kinetic_coeff="auto")


def test_wrappers_reach_their_modules():
    comm = T._community()
    assert type(comm.member_table()).__name__ == "MemberTable"
    result = comm.steadycom()
    assert result.status == "optimal" and result.mu_max > 0
    assert {w.code for w in result.warnings} >= {"not_identifiable"}
    assert set(comm.calibrate_kappa()) == {"memA", "memB"}
    assert type(comm.composition_profile()).__name__ == "ProfileResult"
    micom = comm.micom(tradeoff=0.5)
    micom = micom[0] if isinstance(micom, list) else micom
    assert type(micom).__name__ == "MicomResult"
    assert len(comm.member_model(comm.members[0]).reactions) > 0


def test_package_exports():
    for name in ("SteadyComResult", "ProfileResult", "KappaCalibration", "ConcentrationSwitch",
                 "SteadyState", "SteadyStates", "Trajectory", "MicomResult"):
        assert hasattr(mscommunity, name), name
