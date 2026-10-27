#!/usr/bin/env python
"""day174_reduced.py -- pilot of mscommunity.commcycles on the members of the day-174 cycle.

The codiffusion day-174 community (v7c rebuild: 88 ATP-safe members, each free of ATP from
nothing on its own) makes ATP at the flux bound with every uptake closed, through a cycle across
nine members. This pilot assembles those nine members alone (about 1/10 of the reactions, so
every LP takes seconds), then:

  1. baseline: ATP, growth and electrons from nothing; community and per-member growth on the
     medium; which bound limits the electron test
  2. the cycle: reactions, members, cross-fed metabolites, database directions, dG
  3. cut_energy_cycles with the thermodynamic gate (and with --fallback, its most-uphill second
     tier), protecting every member's standalone growth and the community's growth on the medium
  4. after: the same tests, and growth lost
  5. --looplaw: SimpleThermoPkg (loop law with a strict margin) on the uncut community, as the
     independent check that loop law alone removes the cycle, and what growth it allows

Writes <out>/day174_reduced.json. Paths are arguments; the defaults are the codiffusion layout.
"""
import argparse
import json
import os
import sys
import time
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

CODIF = "/home/freiburger/Documents/codiffusion_bioreactor"
CYCLE_MEMBERS = ["Desulfosporosinus.1", "Erysipelothrix.1", "Methanobacterium.1", "Methanosarcina.1",
                 "Ochrobactrum.1", "Petrimonas.1", "Sedimentibacter.8", "Terrisporobacter.1",
                 "midas_g_25828.1"]
GAS_FEED = {"cpd11640": 1000, "cpd00011": 1000}  # H2, CO2, as the codiffusion builds add them


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def medium_bounds(media_json, name):
    """{cpd: max uptake} for a KBase medium export ({name: {"compounds": {cpd: [maxFlux, ...]}}})
    plus the H2/CO2 gas feed."""
    raw = json.load(open(media_json))[name]["compounds"]
    out = {cpd: (fl[0] if fl[0] is not None else 100) for cpd, fl in raw.items()}
    out.update(GAS_FEED)
    return out


def apply_medium(model, bounds):
    for rxn in model.exchanges:
        met = next(iter(rxn.metabolites))
        if met.compartment.startswith("e"):
            rxn.lower_bound = -abs(bounds.get(met.id.split("_")[0], 0.0))


def gapfilled_reactions(draft, gf):
    """ModelSEED reaction ids gapfilling added to a member or opened further than the draft."""
    from mscommunity.commcycles import msid
    d = {r.id: r.bounds for r in draft.reactions}
    out = set()
    for r in gf.reactions:
        b = d.get(r.id)
        if b is None or r.lower_bound < b[0] or r.upper_bound > b[1]:
            if msid(r):
                out.add(msid(r))
    return out


def bound_limits(model, tol=1e-6):
    """Reactions sitting at a bound other than 0 or +-1000 in the current solution."""
    out = []
    for r in model.reactions:
        v = r.flux
        for b in (r.lower_bound, r.upper_bound):
            if abs(v) > tol and abs(v - b) < tol and abs(b) not in (0.0, 1000.0):
                out.append({"reaction": r.id, "flux": v, "bounds": list(r.bounds)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gf-dir", default=f"{CODIF}/media_gapfilled_models_v7c/Wolfe")
    ap.add_argument("--draft-dir", default=f"{CODIF}/models_v7c")
    ap.add_argument("--media-json", default=f"{CODIF}/modeling_files/kbase_media_207617.json")
    ap.add_argument("--medium", default="Wolfe")
    ap.add_argument("--msdb", default="/home/freiburger/Documents/ModelSEEDDatabase")
    ap.add_argument("--members", nargs="*", default=CYCLE_MEMBERS)
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--no-protect", action="store_true",
                    help="cut every gated direction even if a member then stops growing (to see whether "
                         "the gate alone can remove the cycles)")
    ap.add_argument("--max-rounds", type=int, default=25)
    ap.add_argument("--fallback", default=None, choices=(None, "most_uphill"),
                    help="cut_energy_cycles' second tier, for cycles the gate cannot break")
    ap.add_argument("--tag", default="", help="suffix of the result file")
    ap.add_argument("--looplaw", action="store_true")
    ap.add_argument("--looplaw-timeout", type=int, default=1800)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import cobra
    cobra.Configuration().solver = a.solver
    from cobra.io import read_sbml_model
    from mscommunity.commhelper import build_from_species_models
    from mscommunity import commcycles as CC

    res = {"members": a.members, "medium": a.medium, "gf_dir": a.gf_dir}
    medium = medium_bounds(a.media_json, a.medium)
    members, gapfilled = {}, {}
    for mid in a.members:
        gf = read_sbml_model(f"{a.gf_dir}/{mid}_gf.xml"); gf.id = mid; gf.solver = a.solver
        members[mid] = gf
        gapfilled[mid] = gapfilled_reactions(read_sbml_model(f"{a.draft_dir}/{mid}.xml"), gf)
    log(f"loaded {len(members)} members; gapfilled reactions per member: "
        f"{ {k: len(v) for k, v in gapfilled.items()} }")

    def member_growth(model):
        with model:
            apply_medium(model, medium)
            model.objective = "bio1"
            return model.slim_optimize(error_value=0.0)

    res["member_alone"] = {}
    for mid, m in members.items():
        res["member_alone"][mid] = {"growth": member_growth(m),
                                    **CC.energy_from_nothing(m, biomass="bio1")}
    log("members alone: " + "; ".join(f"{k} g={v['growth']:.3g} atp={v['atp_from_nothing']} "
                                       f"e-={v['electrons_from_nothing']}" for k, v in res["member_alone"].items()))

    t = time.time()
    comm = build_from_species_models(list(members.values()), "day174_cycle_members", "day174 cycle members",
                                     abundances={mid: 1.0 / len(members) for mid in members})
    comm.solver = a.solver
    names = {i + 1: mid for i, mid in enumerate(comm.notes["member_biomass_cpds"])}
    res["community"] = {"reactions": len(comm.reactions), "metabolites": len(comm.metabolites),
                        "build_s": round(time.time() - t, 1), "member_index": names}
    log(f"community: {len(comm.reactions)} reactions, built in {res['community']['build_s']}s")

    def comm_growth(model):
        with model:
            apply_medium(model, medium)
            model.objective = "bio1"
            return model.slim_optimize(error_value=0.0)

    t = time.time()
    res["before"] = CC.energy_from_nothing(comm)
    res["before"]["growth_on_medium"] = comm_growth(comm)
    res["before"]["test_s"] = round(time.time() - t, 1)
    log(f"before: {json.dumps(res['before'])}")

    # what limits the electron test (it stops at exactly 100 in every v7c community)
    with comm:
        CC.close_uptakes(comm, open_cpds=CC.ELECTRON_TEST_INPUTS)
        coefs = {}
        for rxn in comm.exchanges:
            met = next(iter(rxn.metabolites))
            cid = met.id.split("_")[0]
            if met.compartment == "e0" and cid in CC.REDUCED_PRODUCTS:
                coefs[rxn.forward_variable], coefs[rxn.reverse_variable] = CC.REDUCED_PRODUCTS[cid], -CC.REDUCED_PRODUCTS[cid]
        comm.objective = comm.problem.Objective(0, direction="max")
        comm.objective.set_linear_coefficients(coefs)
        comm.slim_optimize()
        res["electron_test_limits"] = bound_limits(comm)
    log(f"electron test limited by: {res['electron_test_limits']}")

    thermo = CC.load_msdb_thermo(a.msdb)
    t = time.time()
    res["cycle_before"] = CC.find_energy_cycle(comm, thermo=thermo, names=names, gapfilled=gapfilled)
    cyc = res["cycle_before"]
    log(f"cycle: {len(cyc['reactions'])} reactions in {cyc['members']} ({time.time() - t:.1f}s); "
        f"cross-fed {len(cyc['cross_fed'])}; gated {sum(r['infeasible'] for r in cyc['reactions'])}, "
        f"against db {sum(r['against_db'] for r in cyc['reactions'])}")

    growth0 = {mid: res["member_alone"][mid]["growth"] for mid in members}
    comm0 = res["before"]["growth_on_medium"]
    protect_log = []

    def protect(model, cuts):
        """Each touched member must still grow alone on the medium, and the community too."""
        lost = []
        by_member = {}
        for c in cuts:
            by_member.setdefault(names.get(c["member_index"]), []).append(c)
        for mid, cs in by_member.items():
            m = members.get(mid)
            if m is None:
                continue
            with m:
                for c in cs:
                    rid = c["reaction"].rsplit("_", 1)[0] + "_c0"
                    if rid in m.reactions:
                        r = m.reactions.get_by_id(rid)
                        r.bounds = (min(r.lower_bound, 0), 0) if c["direction"] == ">" else (0, max(r.upper_bound, 0))
                g = member_growth(m)
            if g <= 1e-6 < growth0[mid]:
                lost.append(mid)
        cg = comm_growth(model)
        ok = not lost and (cg > 1e-6 or comm0 <= 1e-6)
        protect_log.append({"cuts": sorted({c["msid"] + c["direction"] for c in cuts}),
                            "members_stopped": lost, "community_growth": cg, "accepted": ok})
        return ok

    t = time.time()
    report = CC.cut_energy_cycles(comm, thermo=thermo, names=names, gapfilled=gapfilled,
                                  protect=None if a.no_protect else protect, max_rounds=a.max_rounds,
                                  fallback=a.fallback)
    report["protect_log"] = protect_log
    report["seconds"] = round(time.time() - t, 1)
    res["cut"] = report
    log(f"cut: status {report['status']}, {len(report['cuts'])} reaction bounds changed "
        f"({sorted({c['msid'] + c['direction'] for c in report['cuts']})}), kept open "
        f"{[k['msid'] for k in report['kept_open']]}, {report['seconds']}s; rounds "
        f"{[(r['round'], r['atp_from_nothing'], r.get('cycle_reactions')) for r in report['rounds']]}")

    res["after"] = CC.energy_from_nothing(comm)
    res["after"]["growth_on_medium"] = comm_growth(comm)
    log(f"after: {json.dumps(res['after'])}")
    if report["status"] != "clean":
        rem = report.get("remaining_cycle") or {}
        log(f"remaining cycle: {len(rem.get('reactions', []))} reactions in {rem.get('members')}")
    json.dump(res, open(os.path.join(a.out, f"day174_reduced{a.tag}.json"), "w"), indent=1, default=str)

    if a.looplaw:
        res["looplaw"] = looplaw_check(a, members, medium, CC, log)
        json.dump(res, open(os.path.join(a.out, f"day174_reduced{a.tag}.json"), "w"), indent=1, default=str)
    log("done")


def looplaw_check(a, members, medium, CC, log):
    """Loop law with a strict margin (SimpleThermoPkg, revbin coupling) on the uncut community:
    max ATP from nothing and max growth on the medium, as MILPs."""
    from mscommunity.commhelper import build_from_species_models
    from modelseedpy.fbapkg.mspackagemanager import MSPackageManager
    out = {}
    comm = build_from_species_models(list(members.values()), "day174_cycle_members_ll", "ll",
                                     abundances={mid: 1.0 / len(members) for mid in members})
    comm.solver = a.solver
    # the ATP tests go in before the package so that they are internal reactions under the loop
    # law too: with every uptake closed, any flux is then a closed cycle, which the law forbids
    tests = CC.add_atp_tests(comm)
    t = time.time()
    MSPackageManager.get_pkg_mgr(comm).getpkg("SimpleThermoPkg").build_package({"dg_margin": 1.0})
    comm.solver.update()
    g = comm.solver.problem
    out["build_s"] = round(time.time() - t, 1)
    out["binaries"] = g.NumBinVars
    g.Params.TimeLimit = a.looplaw_timeout
    log(f"loop law built in {out['build_s']}s: {g.NumBinVars} binaries")
    for label in ("atp_from_nothing", "growth_on_medium"):
        with comm:
            if label == "atp_from_nothing":
                CC.close_uptakes(comm)
                comm.objective = comm.problem.Objective(sum(r.flux_expression for r in tests.values()),
                                                        direction="max")
            else:
                apply_medium(comm, medium)
                comm.objective = "bio1"
            t = time.time()
            v = comm.slim_optimize(error_value=float("nan"))
            out[label] = {"value": None if v != v else v, "status": str(comm.solver.status),
                          "seconds": round(time.time() - t, 1)}
            for attr in ("ObjBound", "MIPGap"):
                try:
                    out[label][attr] = float(getattr(g, attr))
                except Exception:
                    out[label][attr] = None
        log(f"loop law {label}: {out[label]}")
    return out


if __name__ == "__main__":
    main()
