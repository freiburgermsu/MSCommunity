#!/usr/bin/env python
"""full_community_profile.py -- orient_community on a whole assembled community (SBML).

The codiffusion day-174 v7c community: 88 members, about 90k reactions. Members' references come
from their own models (member_reference with member_models; the order of the provenance
"abundances" is the build order, so member i is the i-th id). On the medium:

  before: ATP, growth and electrons from nothing; growth on the medium (LP)
  orient_community (--rounds), timed
  after: the same, and each member's growth inside the community

Writes <out>/<name>_full_profile.json; --save writes the oriented community SBML there too.
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
    ap.add_argument("--provenance", default=None, help="default: <community>.provenance.json")
    ap.add_argument("--gf-dir", default=f"{CODIF}/media_gapfilled_models_v7c/Wolfe")
    ap.add_argument("--media-json", default=f"{CODIF}/modeling_files/kbase_media_207617.json")
    ap.add_argument("--medium", default="Wolfe")
    ap.add_argument("--rules", default=None, help="JSON {msid: allowed direction} applied first")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--name", default="day174_full")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import cobra
    cobra.Configuration().solver = "gurobi"
    from cobra.io import read_sbml_model, write_sbml_model
    from mscommunity import commcycles as CC

    prov = json.load(open(a.provenance or a.community.replace(".xml", ".provenance.json")))
    order = list(prov["abundances"])
    t = time.time()
    comm = read_sbml_model(a.community)
    res = {"name": a.name, "community": a.community, "reactions": len(comm.reactions),
           "members": len(order), "load_s": round(time.time() - t, 1)}
    log(f"{a.name}: {len(comm.reactions)} reactions, {len(order)} members, loaded in {res['load_s']}s")
    cfg = comm.solver.configuration
    cfg.lp_method, cfg.presolve = "dual", True
    apply_medium(comm, medium_bounds(a.media_json, a.medium))

    def growth(model):
        model.objective = "bio1"
        t0 = time.time()
        v = model.slim_optimize(error_value=float("nan"))
        return (None if v != v else v), round(time.time() - t0, 1)

    t = time.time()
    res["before"] = CC.energy_from_nothing(comm)
    res["before"]["test_s"] = round(time.time() - t, 1)
    res["before"]["growth_on_medium"], res["before"]["growth_s"] = growth(comm)
    log(f"before: {json.dumps(res['before'])}")
    if a.rules:
        rules = json.loads(a.rules)
        res["rules"] = {"rules": rules, "changed": len(CC.apply_direction_rules(comm, rules))}

    t = time.time()
    members = {}
    for i, mid in enumerate(order, 1):
        m = read_sbml_model(f"{a.gf_dir}/{mid}_gf.xml"); m.id = mid; m.solver = "gurobi"
        members[i] = m
    res["member_load_s"] = round(time.time() - t, 1)
    t = time.time()
    comm.objective = "bio1"
    oc = CC.orient_community(comm, rounds=a.rounds, member_models=members)
    prof = oc["profile"]
    res["profile"] = {k: prof[k] for k in ("reactions", "metabolites", "objective", "oriented")}
    res["profile"].update(pins=len(prof["pinned"]), against_reference=prof["against_reference"],
                          irreversible_blocked=prof["irreversible_blocked"], bounds_changed=len(oc["changed"]),
                          history=oc["history"], seconds=round(time.time() - t, 1),
                          member_growth_alone={order[i - 1]: g for i, g in oc["member_growth_alone"].items()})
    log(f"profile: {json.dumps({k: v for k, v in res['profile'].items() if k != 'member_growth_alone'})}")

    t = time.time()
    res["after"] = CC.energy_from_nothing(comm)
    res["after"]["test_s"] = round(time.time() - t, 1)
    res["after"]["growth_on_medium"], res["after"]["growth_s"] = growth(comm)
    log(f"after: {json.dumps(res['after'])}")
    json.dump(res, open(os.path.join(a.out, f"{a.name}_full_profile.json"), "w"), indent=1, default=str)
    if a.save:
        write_sbml_model(comm, os.path.join(a.out, f"{a.name}_oriented.xml"))
    log("done")


if __name__ == "__main__":
    main()
