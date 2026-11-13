# -*- coding: utf-8 -*-
"""build_from_species_models must merge the same members into the same community every time.

The members' reactions were collected in a Python set of cobra objects, which hash by identity, and
ModelSEEDpy's FBAHelper.filter_cobra_set kept whichever same-id duplicate the set yielded first. So a
shared e0 exchange took a random member's bounds on each merge (Kp/Bt: 90 of 107 shared exchanges
differ between the member files, and 4 merges gave 4 bound sets), and a shared e0 metabolite took a
random member's formula and charge. Now the first member in `org_models` order supplies each shared
object, the shared exchange bounds are reconciled by a stated rule (default: their union), and every
difference is warned about and recorded in model.notes["shared_id_merge"].
"""
from __future__ import annotations

import hashlib
import logging

import pytest
from cobra import Metabolite, Model, Reaction

from mscommunity.commhelper import build_from_species_models

N_SHARED = 40              # shared nutrients; every one carries different bounds in the two members
ACETATE = "cpd00029"       # the shared metabolite whose formula and charge the members disagree on
LOGGER = "mscommunity.commhelper"


def _nutrient(k):
    return f"cpd{20000 + k:05d}"


def _member(mid, bounds_of, formula_of, charge_of, name_tag):
    """A member that takes up N_SHARED nutrients (plus acetate) through e0 exchanges into its biomass.

    bounds_of(k), formula_of(cid), charge_of(cid) give the member's own exchange bounds and its
    annotation of each e0 metabolite; name_tag marks which member's object a metabolite name came from."""
    m = Model(mid)
    bio = Reaction("bio1", lower_bound=0, upper_bound=1000)
    biomass = Metabolite("cpd11416_c0", name="biomass", compartment="c0")
    stoich = {biomass: 1}
    rxns = []
    for k, cid in enumerate([_nutrient(k) for k in range(N_SHARED)] + [ACETATE]):
        ext = Metabolite(f"{cid}_e0", name=f"{cid} ({name_tag})", compartment="e0",
                         formula=formula_of(cid), charge=charge_of(cid))
        cyt = Metabolite(f"{cid}_c0", name=cid, compartment="c0")
        ex = Reaction(f"EX_{cid}_e0")
        ex.add_metabolites({ext: -1})
        ex.bounds = bounds_of(k)
        tr = Reaction(f"rxn{10000 + k:05d}_c0", lower_bound=0, upper_bound=1000)
        tr.add_metabolites({ext: -1, cyt: 1})
        rxns += [ex, tr]
        stoich[cyt] = -0.01
    bio.add_metabolites(stoich)
    m.add_reactions(rxns + [bio])
    m.objective = "bio1"
    return m


def _kp_like():
    # like the translated iYL1228: most uptakes closed, one capped, and no e0 annotation at all
    return _member("memA", lambda k: (0.0, 1000.0) if k % 3 else (-(10.0 + k), 1000.0),
                   lambda cid: None, lambda cid: None, "A")


def _bt_like():
    # like iAH991: every exchange open both ways, annotated
    return _member("memB", lambda k: (-1000.0, 1000.0) if k % 2 else (-(5.0 + k), 500.0),
                   lambda cid: "C2H3O2" if cid == ACETATE else "C6H12O6",
                   lambda cid: -1 if cid == ACETATE else 0, "B")


def _plus_annotated_rival():
    # a third member that annotates acetate differently from memB
    return _member("memC", lambda k: (-1.0, 1.0), lambda cid: "C2H4O2" if cid == ACETATE else "C6H12O6",
                   lambda cid: 0, "C")


def _merge(models, **kwargs):
    return build_from_species_models(models, model_id="toy", build_solver="glpk", final_solver="glpk",
                                     **kwargs)


def _fingerprint(model):
    return {
        "reaction_order": [r.id for r in model.reactions],
        "metabolite_order": [m.id for m in model.metabolites],
        "bounds": {r.id: r.bounds for r in model.reactions},
        "stoichiometry": {r.id: {m.id: c for m, c in r.metabolites.items()} for r in model.reactions},
        "annotation": {m.id: (m.formula, m.charge, m.name) for m in model.metabolites},
        "lp_md5": hashlib.md5(model.solver.to_lp().encode()).hexdigest(),
    }


def _shared_exchanges():
    return [f"EX_{_nutrient(k)}_e0" for k in range(N_SHARED)]


def test_merging_the_same_members_five_times_gives_one_community():
    prints = [_fingerprint(_merge([_kp_like(), _bt_like()])) for _ in range(5)]
    for other in prints[1:]:
        for key in prints[0]:
            assert other[key] == prints[0][key], key


def test_shared_exchange_bounds_are_the_union_of_the_members_whatever_their_order():
    a, b = _kp_like(), _bt_like()
    ab, ba = _merge([a, b]), _merge([b, a])
    for k, rid in enumerate(_shared_exchanges()):
        lo_a, hi_a = a.reactions.get_by_id(rid).bounds
        lo_b, hi_b = b.reactions.get_by_id(rid).bounds
        assert ab.reactions.get_by_id(rid).bounds == (min(lo_a, lo_b), max(hi_a, hi_b))
        assert ba.reactions.get_by_id(rid).bounds == ab.reactions.get_by_id(rid).bounds
    # the input models are not touched
    assert a.reactions.get_by_id(_shared_exchanges()[1]).bounds == (0.0, 1000.0)


def test_the_first_rule_keeps_the_first_members_bounds():
    a, b = _kp_like(), _bt_like()
    ab, ba = _merge([a, b], shared_bounds="first"), _merge([b, a], shared_bounds="first")
    for rid in _shared_exchanges():
        assert ab.reactions.get_by_id(rid).bounds == a.reactions.get_by_id(rid).bounds
        assert ba.reactions.get_by_id(rid).bounds == b.reactions.get_by_id(rid).bounds
    with pytest.raises(ValueError, match="shared_bounds"):
        _merge([a, b], shared_bounds="intersection")


def test_a_shared_metabolite_is_the_first_members_object_with_missing_annotation_filled_in():
    ab, ba = _merge([_kp_like(), _bt_like()]), _merge([_bt_like(), _kp_like()])
    glc = f"{_nutrient(0)}_e0"
    # the object (its name says whose it is) comes from the first member...
    assert ab.metabolites.get_by_id(glc).name.endswith("(A)")
    assert ba.metabolites.get_by_id(glc).name.endswith("(B)")
    # ...but memA leaves formula and charge unset, which is not a disagreement: memB's fill them
    for model in (ab, ba):
        assert model.metabolites.get_by_id(glc).formula == "C6H12O6"
        assert model.metabolites.get_by_id(glc).charge == 0
        assert model.metabolites.get_by_id(f"{ACETATE}_e0").formula == "C2H3O2"
    # and every reaction points at the model's own metabolite objects, none at a dropped member's
    for model in (ab, ba):
        for rxn in model.reactions:
            for met in rxn.metabolites:
                assert met is model.metabolites.get_by_id(met.id)
    assert {r.id for r in ab.metabolites.get_by_id(glc).reactions} == {
        f"EX_{_nutrient(0)}_e0", "rxn10000_c1", "rxn10000_c2"}


def test_every_difference_is_warned_about_and_recorded(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        model = _merge([_kp_like(), _bt_like(), _plus_annotated_rival()])
    text = " ".join(rec.getMessage() for rec in caplog.records if rec.name == LOGGER)
    assert f"{N_SHARED + 1} reaction ids that several members share carry different bounds" in text
    assert "union" in text
    assert "shared metabolites have different formulas" in text
    assert "shared metabolites have different charges" in text
    record = model.notes["shared_id_merge"]
    assert record["rule"] == "union" and len(record["bounds"]) == N_SHARED + 1
    entry = record["bounds"][_shared_exchanges()[1]]
    assert entry["members"] == ["memA", "memB", "memC"]
    assert entry["bounds"] == [[0.0, 1000.0], [-1000.0, 1000.0], [-1.0, 1.0]]
    assert entry["kept"] == [-1000.0, 1000.0]
    # memA sets nothing, so the disagreement is memB's C2H3O2 / -1 against memC's C2H4O2 / 0; memB's wins
    assert record["formula"] == {f"{ACETATE}_e0": [["memB", "C2H3O2"], ["memC", "C2H4O2"]]}
    assert record["charge"] == {f"{ACETATE}_e0": [["memB", -1], ["memC", 0]]}
    assert model.metabolites.get_by_id(f"{ACETATE}_e0").formula == "C2H3O2"
    assert f"{_nutrient(0)}_e0" in record["filled"]["formula"]


def test_same_id_with_different_stoichiometry_keeps_the_first_and_warns(caplog):
    a, b = _kp_like(), _bt_like()
    odd = b.reactions.get_by_id(f"EX_{ACETATE}_e0")
    odd.add_metabolites({b.metabolites.get_by_id(f"{ACETATE}_e0"): -1})     # now 2 acetate per unit
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        model = _merge([a, b])
    assert "different stoichiometry" in " ".join(r.getMessage() for r in caplog.records)
    kept = model.reactions.get_by_id(f"EX_{ACETATE}_e0")
    assert {m.id: c for m, c in kept.metabolites.items()} == {f"{ACETATE}_e0": -1}
    # bounds are not merged across different reactions: memA's stand
    assert kept.bounds == a.reactions.get_by_id(f"EX_{ACETATE}_e0").bounds
    assert f"EX_{ACETATE}_e0" in model.notes["shared_id_merge"]["stoichiometry"]


def test_members_without_shared_ids_merge_without_a_warning(caplog):
    a = _member("memA", lambda k: (-1.0, 1.0), lambda c: None, lambda c: None, "A")
    b = Model("memB")
    met = Metabolite("cpd99999_e0", compartment="e0")
    ex, bio = Reaction("EX_cpd99999_e0", lower_bound=-1), Reaction("bio1", upper_bound=1000)
    ex.add_metabolites({met: -1})
    bio.add_metabolites({met: -1, Metabolite("cpd11416_c0", compartment="c0"): 1})
    b.add_reactions([ex, bio])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        model = _merge([a, b])
    assert not [r for r in caplog.records if r.name == LOGGER and "share" in r.getMessage()]
    assert model.notes["shared_id_merge"]["bounds"] == {}
