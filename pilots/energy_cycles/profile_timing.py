#!/usr/bin/env python
"""profile_timing.py -- how long potential_profile's LP takes on a whole community, per backend.

Loads the community once, takes its pFBA optimum on the medium as the reference (timed), and
solves the profile with each --lp backend, recording time, objective and how many reactions
each orients. Writes <out>/<name>_profile_timing.json.
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
from day174_reduced import CODIF, medium_bounds, apply_medium, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default=f"{CODIF}/comm_models_v7c/day174_Jb_comm.xml")
    ap.add_argument("--media-json", default=f"{CODIF}/modeling_files/kbase_media_207617.json")
    ap.add_argument("--medium", default="Wolfe")
    ap.add_argument("--lp", nargs="*", default=["gurobi", "highs-ipm"])
    ap.add_argument("--timeout", type=int, default=1800, help="per backend, seconds (HiGHS only)")
    ap.add_argument("--name", default="day174_full")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    import cobra
    cobra.Configuration().solver = "gurobi"
    from cobra.io import read_sbml_model
    from cobra.flux_analysis import pfba
    from mscommunity import commcycles as CC
    t = time.time()
    comm = read_sbml_model(a.community)
    res = {"reactions": len(comm.reactions), "load_s": round(time.time() - t, 1), "backends": {}}
    cfg = comm.solver.configuration
    cfg.lp_method, cfg.presolve = "dual", True
    apply_medium(comm, medium_bounds(a.media_json, a.medium))
    comm.objective = "bio1"
    t = time.time()
    boundary = {r.id for r in comm.boundary}
    ref = {rid: v for rid, v in pfba(comm).fluxes.items() if abs(v) > 1e-9 and rid not in boundary}
    res["pfba_s"] = round(time.time() - t, 1)
    log(f"loaded in {res['load_s']}s; pFBA reference {len(ref)} reactions in {res['pfba_s']}s")
    for lp in a.lp:
        t = time.time()
        try:
            dg, rep = CC.potential_profile(comm, ref, lp=lp)
            res["backends"][lp] = {"seconds": round(time.time() - t, 1), "objective": rep["objective"],
                                   "oriented": rep["oriented"], "against_reference": len(rep["against_reference"]),
                                   "irreversible_blocked": len(rep["irreversible_blocked"])}
        except Exception as exc:
            res["backends"][lp] = {"seconds": round(time.time() - t, 1), "error": str(exc)[:300]}
        log(f"{lp}: {res['backends'][lp]}")
        json.dump(res, open(os.path.join(a.out, f"{a.name}_profile_timing.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
