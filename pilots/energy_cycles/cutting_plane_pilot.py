#!/usr/bin/env python
"""cutting_plane_pilot.py -- the loop-law optimum by covering only the cycles the optimum uses.

mscommunity.commcycles.loop_law_optimum on the nine members of the day-174 cycle (default) or on
an assembled community SBML (--community). The warm start is an orient_community optimum
(--start-reference, default "both", which always leaves every member able to grow) solved on a
copy. Reports each round (objective, cycle size, covered reactions, time), checks that the final
solution holds no cycle, and builds the hybrid: one potential profile on that solution, which
keeps its growth with LPs only. Writes <out>/<name>_cutting.json and the Gurobi log.
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
sys.path.insert(0, HERE)
from day174_reduced import CODIF, CYCLE_MEMBERS, medium_bounds, apply_medium, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default=None)
    ap.add_argument("--members", nargs="*", default=CYCLE_MEMBERS)
    ap.add_argument("--gf-dir", default=f"{CODIF}/media_gapfilled_models_v7c/Wolfe")
    ap.add_argument("--media-json", default=f"{CODIF}/modeling_files/kbase_media_207617.json")
    ap.add_argument("--medium", default="Wolfe")
    ap.add_argument("--start-reference", default="both", choices=("members", "pfba", "both"))
    ap.add_argument("--time-limit", type=int, default=1200, help="per MILP round, seconds")
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-potential", type=float, default=200.0)
    ap.add_argument("--name", default="day174_9members")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import cobra
    cobra.Configuration().solver = "gurobi"
    from cobra.io import read_sbml_model
    from cobra.flux_analysis import pfba
    from mscommunity import commcycles as CC

    t = time.time()
    member_models = None
    if a.community:
        comm = read_sbml_model(a.community)
        order = list(json.load(open(a.community.replace(".xml", ".provenance.json")))["abundances"])
        member_models = {}
        for i, mid in enumerate(order, 1):
            m = read_sbml_model(f"{a.gf_dir}/{mid}_gf.xml"); m.id = mid; m.solver = "gurobi"
            member_models[i] = m
    else:
        from mscommunity.commhelper import build_from_species_models
        members = []
        for mid in a.members:
            m = read_sbml_model(f"{a.gf_dir}/{mid}_gf.xml"); m.id = mid
            members.append(m)
        comm = build_from_species_models(members, "cycle_members", "cycle members",
                                         abundances={m.id: 1.0 / len(members) for m in members})
    comm.solver = "gurobi"
    cfg = comm.solver.configuration
    cfg.lp_method, cfg.presolve = "dual", True
    apply_medium(comm, medium_bounds(a.media_json, a.medium))
    comm.objective = "bio1"
    res = {"name": a.name, "reactions": len(comm.reactions), "load_s": round(time.time() - t, 1)}
    log(f"{a.name}: {len(comm.reactions)} reactions ({res['load_s']}s)")

    t = time.time()
    safe = comm.copy()
    oc = CC.orient_community(safe, reference=a.start_reference, member_models=member_models)
    start = pfba(safe).fluxes.to_dict()
    del safe
    res["start"] = {"reference": a.start_reference, "growth": oc["history"][-1]["objective"],
                    "cycle_in_start": len(CC.conformal_cycle(comm, start)), "seconds": round(time.time() - t, 1)}
    log(f"start: {res['start']}")

    g = comm.solver.problem
    g.Params.LogFile = os.path.join(a.out, f"{a.name}_cutting_gurobi.log")
    g.Params.OutputFlag, g.Params.LogToConsole = 1, 0
    g.Params.Threads, g.Params.MIPFocus = a.threads, 1
    t = time.time()
    fluxes, report = CC.loop_law_optimum(comm, start=start, time_limit=a.time_limit, max_rounds=a.max_rounds,
                                         max_potential=a.max_potential)
    law = report.pop("law")
    report["seconds"] = round(time.time() - t, 1)
    report["final_cycle"] = len(CC.conformal_cycle(comm, fluxes)) if fluxes else None
    for attr in ("ObjBound", "MIPGap"):
        try:
            report[attr] = float(getattr(g, attr))
        except Exception:
            report[attr] = None
    res["cutting"] = report
    log(f"cutting: {report['status']}; rounds {[(r['round'], r['objective'], r.get('cycle'), r['covered'], r['seconds']) for r in report['rounds']]}; "
        f"final cycle {report['final_cycle']}; bound {report.get('ObjBound')}; {report['seconds']}s")
    json.dump(res, open(os.path.join(a.out, f"{a.name}_cutting.json"), "w"), indent=1, default=str)

    law.remove()
    if fluxes and report["status"] == "no cycle":
        t = time.time()
        boundary = {r.id for r in comm.boundary}
        ref = {rid: v for rid, v in fluxes.items() if abs(v) > 1e-9 and rid not in boundary}
        oc = CC.orient_community(comm, reference=ref)
        res["hybrid"] = {"growth": oc["history"][-1]["objective"], "oriented": oc["profile"]["oriented"],
                         "against_reference": len(oc["profile"]["against_reference"]),
                         "seconds": round(time.time() - t, 1)}
        res["hybrid"].update(CC.energy_from_nothing(comm))
        log(f"hybrid: {json.dumps(res['hybrid'])}")
    json.dump(res, open(os.path.join(a.out, f"{a.name}_cutting.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
