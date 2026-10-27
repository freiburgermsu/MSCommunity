#!/usr/bin/env python
"""milp_start_pilot.py -- how much growth does one potential profile give up against the loop law?

orient_community (an LP) gives a growth that is feasible under the loop law, so it is a lower
bound on the loop-law growth; the unconstrained LP, which runs the cycles, is an upper bound.
This runs the exact loop law (LocalLoopLaw on loop_capable_reactions, ATP pinned) as a MILP,
started from the orient_community optimum, to see how far the incumbent and the bound move.

Without a start the same MILP on these nine members found no solution above 0 in 30 min.

Then the hybrid: the MILP's best solution becomes the reference of a new potential profile
(orient_community with reference = that solution), on a fresh copy of the community, so that
the loop-law growth the MILP found is kept in a model that needs only LPs from then on.
Writes <out>/<name>_milp.json and the Gurobi log beside it.
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
    ap.add_argument("--members", nargs="*", default=CYCLE_MEMBERS)
    ap.add_argument("--gf-dir", default=f"{CODIF}/media_gapfilled_models_v7c/Wolfe")
    ap.add_argument("--override-dir", default=None)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--media-json", default=f"{CODIF}/modeling_files/kbase_media_207617.json")
    ap.add_argument("--medium", default="Wolfe")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-potential", type=float, default=200.0)
    ap.add_argument("--time-limit", type=int, default=3600)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--name", default="day174_9members")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import cobra
    cobra.Configuration().solver = "gurobi"
    from cobra.io import read_sbml_model
    from cobra.flux_analysis import pfba
    from mscommunity.commhelper import build_from_species_models
    from mscommunity import commcycles as CC

    members = []
    for mid in a.members:
        p = f"{a.gf_dir}/{mid}_gf.xml"
        if a.override_dir and os.path.exists(f"{a.override_dir}/{mid}_gf.xml"):
            p = f"{a.override_dir}/{mid}_gf.xml"
        m = read_sbml_model(p); m.id = mid
        members.append(m)
    comm = build_from_species_models(members, "cycle_members", "cycle members",
                                     abundances={m.id: 1.0 / len(members) for m in members})
    comm.solver = "gurobi"
    apply_medium(comm, medium_bounds(a.media_json, a.medium))
    if a.rules:
        CC.apply_direction_rules(comm, json.loads(a.rules))
    comm.objective = "bio1"
    res = {"name": a.name, "reactions": len(comm.reactions), "lp_growth": comm.slim_optimize()}

    t = time.time()
    oriented = comm.copy()
    oc = CC.orient_community(oriented, rounds=a.rounds)
    start = pfba(oriented).fluxes.to_dict()
    res["profile"] = {"growth": oc["history"][-1]["objective"], "seconds": round(time.time() - t, 1),
                      "history": oc["history"]}
    log(f"LP growth {res['lp_growth']:.4g}; profile growth {res['profile']['growth']:.4g} ({res['profile']['seconds']}s)")
    del oriented

    t = time.time()
    loops = CC.loop_capable_reactions(comm, processes=a.processes)
    law = CC.LocalLoopLaw(comm, max_potential=a.max_potential)
    law.add([comm.reactions.get_by_id(r) for r in sorted(loops)])
    res["law"] = {"covered": len(law.binaries), "binaries": 2 * len(law.binaries), "pins": len(law.pinned),
                  "build_s": round(time.time() - t, 1)}
    g = comm.solver.problem
    g.Params.LogFile = os.path.join(a.out, f"{a.name}_milp_gurobi.log")
    g.Params.OutputFlag, g.Params.LogToConsole = 1, 0
    g.Params.TimeLimit, g.Params.Threads, g.Params.MIPFocus = a.time_limit, a.threads, 1
    res["law"]["start_set"] = law.start(start)
    log(f"loop law on {len(law.binaries)} reactions ({res['law']['build_s']}s); MIP start set: {res['law']['start_set']}")
    t = time.time()
    comm.slim_optimize()
    res["milp"] = {"status": str(comm.solver.status), "seconds": round(time.time() - t, 1)}
    for attr in ("ObjVal", "ObjBound", "MIPGap", "NodeCount", "SolCount"):
        try:
            res["milp"][attr] = float(getattr(g, attr))
        except Exception:
            res["milp"][attr] = None
    log(f"MILP: {res['milp']}")
    best = {r.id: r.flux for r in comm.reactions} if res["milp"].get("SolCount") else {}
    json.dump(res, open(os.path.join(a.out, f"{a.name}_milp.json"), "w"), indent=1, default=str)

    # the hybrid: one profile built on the MILP's solution
    law.remove()
    if best:
        t = time.time()
        hybrid = comm.copy()
        boundary = {r.id for r in hybrid.boundary}
        ref = {rid: v for rid, v in best.items() if abs(v) > 1e-9 and rid not in boundary}
        oc = CC.orient_community(hybrid, reference=ref, rounds=1)
        res["hybrid"] = {"growth": oc["history"][-1]["objective"], "oriented": oc["profile"]["oriented"],
                         "against_reference": len(oc["profile"]["against_reference"]),
                         "irreversible_blocked": len(oc["profile"]["irreversible_blocked"]),
                         "seconds": round(time.time() - t, 1)}
        res["hybrid"].update(CC.energy_from_nothing(hybrid))
        log(f"hybrid: {json.dumps(res['hybrid'])}")
    json.dump(res, open(os.path.join(a.out, f"{a.name}_milp.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
