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


# --------------------------------------------------------------------------- set_abundance
def _bio1(comm):
    return {met.id: coef for met, coef in comm.primary_biomass.metabolites.items()}


def test_set_abundance_keeps_the_other_bio1_terms():
    """Only the member-biomass coefficients are replaced: a community-level cost on bio1 survives (the
    integration commit rebuilt bio1 from its product and the member terms and dropped it)."""
    comm = T._community()
    atp = comm.util.model.metabolites.get_by_id("cpd00002_c1")
    comm.primary_biomass.add_metabolites({atp: -0.1})
    comm.set_abundance(_abund(0.3))
    assert _bio1(comm) == pytest.approx({"cpd11416_c0": 1.0, "cpd11416_c1": -0.3, "cpd11416_c2": -0.7,
                                         "cpd00002_c1": -0.1})
    before = _bio1(comm)
    with comm.util.model:
        comm.set_abundance(_abund(0.0))
        assert _bio1(comm) == pytest.approx({"cpd11416_c0": 1.0, "cpd11416_c2": -1.0, "cpd00002_c1": -0.1})
    assert _bio1(comm) == before


def test_set_abundance_inside_a_context_reverts_every_attribute():
    comm = T._community()
    comm.abundances_set = False
    abundances = dict(comm.abundances)
    with comm.util.model:
        comm.set_abundance(_abund(0.2))
        assert comm.abundances_set is True and comm.abundances == pytest.approx({"memA": 0.2, "memB": 0.8})
    assert comm.abundances_set is False and comm.abundances == abundances


def test_set_abundance_takes_plain_numbers():
    comm = T._community()
    comm.set_abundance({"memA": 3.0, "memB": 1.0})
    assert {m.id: m.abundance for m in comm.members} == pytest.approx({"memA": 0.75, "memB": 0.25})
    assert _bio1(comm)["cpd11416_c1"] == pytest.approx(-0.75)


@pytest.mark.parametrize("bad, match", [
    ({"memA": {"abundance": 0.9}}, "every member"),
    ({"memA": 0.9, "memB": 0.1, "memC": 0.1}, "unknown"),
    ({"memA": -0.1, "memB": 1.0}, ">= 0"),
    ({"memA": 0.0, "memB": 0.0}, "sum"),
])
def test_set_abundance_validates_before_changing_anything(bad, match):
    comm = T._community()
    state = (_bio1(comm), {m.id: m.abundance for m in comm.members}, dict(comm.abundances))
    with pytest.raises(ValueError, match=match):
        comm.set_abundance(bad)
    assert (_bio1(comm), {m.id: m.abundance for m in comm.members}, dict(comm.abundances)) == state


def test_predict_abundances_update_abundances_writes_the_prediction():
    """update_abundances=True hands set_abundance the plain {member: a} it computed, which always raised."""
    comm = MSCommunity(model=_merged(), ids=["memA", "memB"], kinetic_coeff=1e6)
    out = comm.predict_abundances(update_abundances=True, regularization=False, pfba=False)
    assert out is not None and sum(out.values()) == pytest.approx(1.0)
    assert comm.abundances == pytest.approx(out)
    bio1 = _bio1(comm)
    for m in comm.members:
        assert bio1.get(m.biomass_cpd.id, 0.0) == pytest.approx(-out[m.id])


# --------------------------------------------------------------------------- predict_abundances contract
def _unpayable():
    """memA's ATPM floor (100) cannot be paid from 10 glucose: the community is infeasible."""
    from mscommunity.commhelper import build_from_species_models
    members = [T._member("memA"), T._member("memB", periplasm=True)]
    members[0].reactions.get_by_id("rxn00062_c0").lower_bound = 100.0
    model = build_from_species_models(members, abundances={"memA": 0.5, "memB": 0.5})
    return MSCommunity(model=model, ids=["memA", "memB"], kinetic_coeff=1e6)


@pytest.mark.parametrize("regularization", [True, False])
def test_predict_abundances_returns_none_when_the_community_cannot_grow(regularization, caplog):
    comm = _unpayable()
    model = comm.util.model
    objective = str(model.objective.expression)
    rows = sorted(c.name for c in model.constraints)
    with caplog.at_level("WARNING"):
        assert comm.predict_abundances(regularization=regularization, pfba=False) is None
    assert any("predict_abundances" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    assert sorted(c.name for c in model.constraints) == rows
    assert str(model.objective.expression) == objective
