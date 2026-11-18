# -*- coding: utf-8 -*-
"""commcycles' single-model audit comes from ModelSEEDpy (modelseedpy.core.msenergyaudit), or,
for a ModelSEEDpy without that module, from MSCommunity's own copy of it with one warning.

These tests run in either setting: the ones that compare the two skip when the installed
ModelSEEDpy lacks the module, and the fallback is exercised in a subprocess that hides it.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mscommunity import commcycles

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = "modelseedpy.core.msenergyaudit"


def _module():
    try:
        return importlib.import_module(AUDIT)
    except ModuleNotFoundError:
        return None


def _community():
    sys.path.insert(0, HERE)
    try:
        from test_g_energy_cycles import _community as build
    finally:
        sys.path.remove(HERE)
    return build()


def test_commcycles_uses_the_modelseedpy_module_when_it_is_installed():
    mod = _module()
    if mod is None:
        assert isinstance(commcycles._ea, SimpleNamespace)
    else:
        assert commcycles._ea is mod
        assert commcycles.close_uptakes is mod.close_uptakes
        assert commcycles.NTP_PINS == mod.NTP_PINS


def test_without_the_module_commcycles_warns_once_and_uses_its_own_copy():
    """A subprocess hides the module (as an older ModelSEEDpy would lack it), imports commcycles
    and runs its tests on the two-member cycle; they must read as they do here."""
    code = f"""
import json, sys, warnings
import modelseedpy
sys.modules["{AUDIT}"] = None
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from mscommunity import commcycles
sys.path.insert(0, {HERE!r})
from test_g_energy_cycles import _community
res = commcycles.energy_from_nothing(_community(), pins="ntp")
cyc = commcycles.find_energy_cycle(_community())
print(json.dumps({{
    "warnings": [str(w.message) for w in caught if "{AUDIT}" in str(w.message)],
    "fallback": type(commcycles._ea).__name__,
    "atp": res["atp_from_nothing"], "by_member": sorted(res["atp_by_member"]),
    "electrons": res["electrons_from_nothing"],
    "cycle": sorted(r["reaction"] for r in cyc["reactions"]), "cross_fed": sorted(cyc["cross_fed"]),
}}))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(got["warnings"]) == 1 and AUDIT in got["warnings"][0]
    assert got["fallback"] == "SimpleNamespace"
    here = commcycles.energy_from_nothing(_community(), pins="ntp")
    cyc = commcycles.find_energy_cycle(_community())
    assert got["atp"] == pytest.approx(here["atp_from_nothing"])
    assert got["by_member"] == sorted(here["atp_by_member"])
    assert got["electrons"] == pytest.approx(here["electrons_from_nothing"])
    assert got["cycle"] == sorted(r["reaction"] for r in cyc["reactions"])
    assert got["cross_fed"] == sorted(cyc["cross_fed"])


def _public(ns):
    return {k for k, v in vars(ns).items()
            if not k.startswith("_") and getattr(v, "__module__", None) != "typing"
            and not isinstance(v, type(sys)) and k not in ("Reaction", "Zero", "contextmanager",
                                                            "isnan", "ln", "logger")}


def test_the_own_copy_offers_what_the_module_offers_and_reads_the_same():
    mod = _module()
    if mod is None:
        pytest.skip(f"the installed ModelSEEDpy has no {AUDIT}")
    own = commcycles._fallback_energy_audit()
    assert _public(mod) <= set(vars(own))
    for name in _public(mod):
        if not callable(getattr(mod, name)):
            assert getattr(own, name) == getattr(mod, name), name
    thermo = {"rxn90002": {"dg": 35.0, "err": 1.0, "source": "test", "reversibility": "="}}
    for ea in (mod, own):
        model = _community()
        sealed = ea.energy_from_nothing(model, pins="ntp", boundary_ions=True, lift_rows="_commKin")
        cycle = ea.find_energy_cycle(model, thermo=thermo)
        gross = ea.net_atp_yield(model, "cpd00027", acceptor=None)
        loop = ea.conformal_cycle(model, cycle["fluxes"], lp="highs")
        if ea is mod:
            first = (sealed, cycle, gross, loop)
    assert sealed == first[0]
    assert cycle == first[1]
    assert gross == first[2]
    assert loop == first[3] and loop


def test_the_private_names_scripts_import_are_still_there():
    """Analysis scripts outside the package import _seal and _maximise_tests (and read _MSID,
    _hydrolysis) from commcycles; they must take the arguments they took before the move."""
    model = _community()
    expected = commcycles.energy_from_nothing(model, electrons=False, biomass=None)
    with model:
        released = commcycles._seal(model, "lax", True)
        tests = commcycles.add_atp_tests(model)
        made = commcycles._maximise_tests(model, tests)
    assert released == expected["released"]
    assert made == pytest.approx(expected["atp_from_nothing"]) and made > 0
    with model:
        commcycles._seal(model, "strict", True, open_cpds=("cpd00067",), biomass_outlet=False)
        opened = [r for r in model.boundary if next(iter(r.metabolites)).id == "cpd00067_e0"]
        assert all(r.lower_bound == -1000 for r in opened)
        assert all(r.bounds == (0, 0) for r in model.boundary if r not in opened)
    assert commcycles._hydrolysis("cpd00002", "cpd00008") == {
        "cpd00002": -1, "cpd00001": -1, "cpd00008": 1, "cpd00009": 1, "cpd00067": 1}
    assert commcycles._MSID.match("rxn00571_c16").group(1) == "rxn00571"
