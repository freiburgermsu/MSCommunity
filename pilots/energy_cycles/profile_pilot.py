#!/usr/bin/env python
"""profile_pilot.py -- orient a community by one potential profile (mscommunity.commcycles).

Either assembles members (--members from --gf-dir, any of them replaced by a model in
--override-dir) or reads an assembled community SBML (--community). On the medium:

  1. LP baseline: ATP, growth and electrons from nothing; growth on the medium
  2. --rules: direction curations applied first (e.g. the gate's verdicts)
  3. orient_community: member_reference (each member alone, pFBA) -> potential_profile (ATP
     hydrolysis pinned downhill, weighted L1) -> orient_by_potentials, --rounds times, each round
     adding the last optimum to the reference
  4. after: ATP, growth and electrons from nothing; growth on the medium; per-member growth
     inside the community; what was oriented

Writes <out>/<name>_profile.json.
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
    ap.add_argument("--override-dir", default=None, help="member models that replace the --gf-dir ones")
    ap.add_argument("--rules", default=None, help="JSON {msid: allowed direction} applied first")
    ap.add_argument("--media-json", default=f"{CODIF}/modeling_files/kbase_media_207617.json")
    ap.add_argument("--medium", default="Wolfe")
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--rounds", type=int, default=1, help="orient_community rounds")
    ap.add_argument("--reference", default="members", choices=("members", "pfba", "both"))
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
        members, sources = [], {}
        for mid in a.members:
            p = f"{a.gf_dir}/{mid}_gf.xml"
            if a.override_dir and os.path.exists(f"{a.override_dir}/{mid}_gf.xml"):
                p = f"{a.override_dir}/{mid}_gf.xml"
            m = read_sbml_model(p); m.id = mid
            members.append(m); sources[mid] = p
        comm = build_from_species_models(members, "cycle_members", "cycle members",
                                         abundances={m.id: 1.0 / len(members) for m in members})
        name = a.name or f"{len(members)}_members"
    comm.solver = a.solver
    names = {i + 1: mid for i, mid in enumerate(comm.notes.get("member_biomass_cpds", {}))}
    res = {"name": name, "reactions": len(comm.reactions), "load_s": round(time.time() - t, 1),
           "member_index": names, "sources": sources if not a.community else a.community}
    log(f"{name}: {len(comm.reactions)} reactions, loaded in {res['load_s']}s")
    apply_medium(comm, medium_bounds(a.media_json, a.medium))

    def growth(model):
        model.objective = "bio1"
        return model.slim_optimize(error_value=0.0)

    res["before"] = CC.energy_from_nothing(comm)
    res["before"]["growth_on_medium"] = growth(comm)
    log(f"before: {json.dumps(res['before'])}")
    if a.rules:
        rules = json.loads(a.rules)
        res["rules"] = {"rules": rules, "changed": CC.apply_direction_rules(comm, rules)}
        res["after_rules"] = CC.energy_from_nothing(comm, electrons=False, biomass=None)
        res["after_rules"]["growth_on_medium"] = growth(comm)
        log(f"rules {rules}: {len(res['rules']['changed'])} bounds changed; {json.dumps(res['after_rules'])}")

    t = time.time()
    comm.objective = "bio1"
    oc = CC.orient_community(comm, rounds=a.rounds, reference=a.reference)
    report, changed = oc["profile"], oc["changed"]
    report["seconds"] = round(time.time() - t, 1)
    report["history"] = oc["history"]
    report["member_growth_alone"] = {names.get(i, i): g for i, g in (oc["member_growth_alone"] or {}).items()}
    report["bounds_changed"] = len(changed)
    report["members_oriented"] = {}
    for rid, _, _ in changed:
        idx = CC.member_index(comm.reactions.get_by_id(rid))
        key = str(names.get(idx, idx))
        report["members_oriented"][key] = report["members_oriented"].get(key, 0) + 1
    report["changed"] = changed
    res["profile"] = report
    against = report["against_reference"]
    log(f"profile: {report['oriented']} of {report['reactions']} reactions oriented, {len(changed)} bounds "
        f"changed, {[len(x) for x in against] if against and isinstance(against[0], list) else len(against)} against the reference, "
        f"{len(report['irreversible_blocked'])} irreversible blocked, {len(report['pinned'])} pins; "
        f"{report['seconds']}s; rounds {[(h['round'], h['objective'], h['oriented']) for h in oc['history']]}")

    res["after"] = CC.energy_from_nothing(comm)
    res["after"]["growth_on_medium"] = growth(comm)
    sol = comm.optimize()
    res["after"]["member_growth_in_community"] = {
        names.get(CC.member_index(r), r.id): sol.fluxes[r.id] for r in comm.reactions
        if r.id.startswith("bio") and r.id != "bio1" and CC.member_index(r)}
    log(f"after: {json.dumps(res['after'])}")
    json.dump(res, open(os.path.join(a.out, f"{name}_profile.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
