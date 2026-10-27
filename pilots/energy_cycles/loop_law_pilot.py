#!/usr/bin/env python
"""loop_law_pilot.py -- mscommunity.commcycles.lazy_loop_law on a community.

Either assembles members (--members, from --gf-dir) or reads an assembled community SBML
(--community). Then:

  1. LP baseline: ATP, growth and electrons from nothing; growth on the medium
  2. --mode lazy: lazy_loop_law covers just the reactions of the cycles the solver finds, round
     by round, until the community makes no ATP from nothing (ATP hydrolysis pinned downhill);
     --mode loopset: loop_capable_reactions + LocalLoopLaw, the exact (full) loop law with one
     binary per reaction that can carry flux in the closed community
  3. under the law (MILP): growth on the medium, electrons from nothing, and solve times

Writes <out>/<name>_<mode>_loop_law.json and the Gurobi log beside it.
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
    ap.add_argument("--community", help="assembled community SBML (instead of --members)")
    ap.add_argument("--members", nargs="*", default=CYCLE_MEMBERS)
    ap.add_argument("--gf-dir", default=f"{CODIF}/media_gapfilled_models_v7c/Wolfe")
    ap.add_argument("--media-json", default=f"{CODIF}/modeling_files/kbase_media_207617.json")
    ap.add_argument("--medium", default="Wolfe")
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time-limit", type=int, default=1800, help="per MILP, seconds")
    ap.add_argument("--max-rounds", type=int, default=50)
    ap.add_argument("--no-pin", action="store_true", help="do not pin ATP hydrolysis downhill")
    ap.add_argument("--mode", choices=("lazy", "loopset"), default="lazy")
    ap.add_argument("--processes", type=int, default=8, help="FVA workers for --mode loopset")
    ap.add_argument("--mip-focus", type=int, default=None, help="Gurobi MIPFocus")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import cobra
    cobra.Configuration().solver = a.solver
    from cobra.io import read_sbml_model
    from mscommunity import commcycles as CC

    t = time.time()
    if a.community:
        comm = read_sbml_model(a.community)
        name = a.name or os.path.basename(a.community).replace(".xml", "")
    else:
        from mscommunity.commhelper import build_from_species_models
        members = []
        for mid in a.members:
            m = read_sbml_model(f"{a.gf_dir}/{mid}_gf.xml"); m.id = mid
            members.append(m)
        comm = build_from_species_models(members, "cycle_members", "cycle members",
                                         abundances={m.id: 1.0 / len(members) for m in members})
        name = a.name or f"{len(members)}_members"
    comm.solver = a.solver
    try:
        comm.solver.problem.Params.Threads = a.threads
    except Exception:
        pass
    res = {"name": name, "reactions": len(comm.reactions), "metabolites": len(comm.metabolites),
           "load_s": round(time.time() - t, 1), "pin_atp": not a.no_pin, "time_limit": a.time_limit}
    log(f"{name}: {len(comm.reactions)} reactions, loaded in {res['load_s']}s")
    medium = medium_bounds(a.media_json, a.medium)

    def growth(model):
        with model:
            apply_medium(model, medium)
            model.objective = "bio1"
            t0 = time.time()
            v = model.slim_optimize(error_value=float("nan"))
            return {"value": None if v != v else v, "status": str(model.solver.status),
                    "seconds": round(time.time() - t0, 1)}

    t = time.time()
    res["lp"] = CC.energy_from_nothing(comm)
    res["lp"]["growth_on_medium"] = growth(comm)
    res["lp"]["seconds"] = round(time.time() - t, 1)
    log(f"LP: {json.dumps(res['lp'])}")

    try:
        g = comm.solver.problem
        g.Params.LogFile = os.path.join(a.out, f"{name}_{a.mode}_gurobi.log")
        g.Params.OutputFlag = 1
        g.Params.LogToConsole = 0
        if a.mip_focus is not None:
            g.Params.MIPFocus = a.mip_focus
    except Exception:
        pass
    t = time.time()
    if a.mode == "lazy":
        law, report = CC.lazy_loop_law(comm, max_rounds=a.max_rounds, time_limit=a.time_limit,
                                       pin_atp=not a.no_pin)
        report["seconds"] = round(time.time() - t, 1)
        report["potentials"] = len(law.potentials)
        res["lazy"] = report
        log(f"lazy loop law: {report['status']}, {report['covered']} reactions covered "
            f"({100 * report['covered'] / len(comm.reactions):.2f}%), {len(law.pinned)} ATP pins, "
            f"{report['seconds']}s; rounds {[(r['round'], r['atp_from_nothing'], r['covered'], r['seconds']) for r in report['rounds']]}")
    else:
        loops = CC.loop_capable_reactions(comm, processes=a.processes)
        res["loopset"] = {"reactions": len(loops), "seconds": round(time.time() - t, 1),
                          "share": len(loops) / len(comm.reactions)}
        t = time.time()
        law = CC.LocalLoopLaw(comm, pin_atp=not a.no_pin)
        law.add([comm.reactions.get_by_id(r) for r in sorted(loops)])
        res["loopset"]["build_s"] = round(time.time() - t, 1)
        log(f"loop set: {json.dumps(res['loopset'])}")
    json.dump(res, open(os.path.join(a.out, f"{name}_{a.mode}_loop_law.json"), "w"), indent=1, default=str)

    comm.solver.configuration.timeout = a.time_limit
    res["law"] = {"growth_on_medium": growth(comm)}
    log(f"under the law, growth on the medium: {res['law']['growth_on_medium']}")
    if a.mode == "loopset":
        t = time.time()
        res["law"]["atp_from_nothing"] = CC.energy_from_nothing(comm, electrons=False, biomass=None)["atp_from_nothing"]
        res["law"]["atp_seconds"] = round(time.time() - t, 1)
    t = time.time()
    res["law"].update(CC.electrons_from_nothing(comm))
    res["law"]["electrons_seconds"] = round(time.time() - t, 1)
    try:
        g = comm.solver.problem
        res["law"]["binaries"] = g.NumBinVars
    except Exception:
        pass
    log(f"under the law: {json.dumps(res['law'])}")
    json.dump(res, open(os.path.join(a.out, f"{name}_{a.mode}_loop_law.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
