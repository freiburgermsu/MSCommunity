#!/usr/bin/env python
"""regapfill_members.py -- can members that grow only through an uphill direction be gapfilled
without it?

cut_energy_cycles' gate rules out directions that no concentrations could drive (dG'min > 0).
On day 174, two of them carry a member's standalone growth: Terrisporobacter.1 grows on Wolfe
only with rxn00571 forward (nitrite + NAD+ -> nitrate + NADH, dG'0 +34.8 kcal/mol) and
Sedimentibacter.8 only with rxn00288 forward (succinate + FAD -> fumarate + FADH2 with free FAD,
+20.3). The cross-member cycle runs through both. A cut at community level cannot fix that: the
members were gapfilled onto the uphill direction and have to be gapfilled again without it.

For each member: the draft (models_v7c), with the direction rules applied to the draft and to
its template (as a template curation would), gapfilled ATP-safe on the medium -- the codiffusion
`--atp-safe` recipe: MSATPCorrection's expansion filter, then MSGapfill checked against the
empty-medium ATP test only. Reports growth, what was added, and the member's ATP and electrons
from nothing; writes the regapfilled models to <out>/models/ so that the community can be
re-cut with them (day174_reduced.py --gf-dir-override).
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

TEMPLATES = "/home/freiburger/Documents/ModelSEEDTemplates-codif-curated/templates/v7.0"
TEMPLATE_FILE = {"gram_pos": "GramPosModelTemplateV7.json", "gram_neg": "GramNegModelTemplateV7.json",
                 "archaea": "ArchaeaTemplateV6.json"}
# the gate's verdicts on day 174 (results/day174_reduced.json): direction allowed
RULES = {"rxn00571": "<", "rxn00288": "<", "rxn00359": ">"}


def curated_template(path, rules):
    """The template with each rule's reaction restricted to the allowed direction (direction,
    GapfillDirection and the bounds that follow them)."""
    from modelseedpy.core.mstemplate import MSTemplateBuilder
    d = json.load(open(path))
    changed = []
    for r in d["reactions"]:
        rule = rules.get(r["id"].split("_")[0])
        if rule is None:
            continue
        old = (r.get("direction"), r.get("GapfillDirection"))
        r["direction"] = r["GapfillDirection"] = rule
        if rule == ">":
            r["lower_bound"], r["upper_bound"] = 0, max(r.get("upper_bound", 1000), 0) or 1000
            r["maxrevflux"] = 0
        else:
            r["lower_bound"], r["upper_bound"] = min(r.get("lower_bound", -1000), 0) or -1000, 0
            r["maxforflux"] = 0
        changed.append((r["id"], old, rule))
    return MSTemplateBuilder.from_dict(d).build(), changed


def atp_safe_gapfill(model, template, core_path, media, rounds=3):
    """codiffusion build_prfba_community_models._mg_one with --atp-safe, for one member."""
    from modelseedpy.core.msatpcorrection import MSATPCorrection
    from modelseedpy.core.msgapfill import MSGapfill
    from modelseedpy.core.mstemplate import MSTemplateBuilder
    core = MSTemplateBuilder.from_dict(json.load(open(core_path))).build()
    before = {r.id for r in model.reactions}
    atp = MSATPCorrection(model, core)
    atp.evaluate_growth_media(no_gapfilling=True)
    atp.determine_growth_media()
    atp.expand_model_to_genome_scale()
    tests = [x for x in atp.build_tests() if x["media"].id == "empty"]
    added_atpase = atp.atp_hydrolysis.id if atp.atp_hydrolysis.id not in before else None
    added = {}
    for _ in range(rounds):
        model.objective = "bio1"
        g = model.slim_optimize(error_value=0.0)
        if g > 1e-6:
            break
        gf = MSGapfill(model, default_gapfill_templates=[template], test_conditions=tests)
        sol = gf.run_gapfilling(media=media, target="bio1", minimum_obj=0.01, binary_check=False,
                                prefilter=True)
        if not sol:
            break
        gf.integrate_gapfill_solution(sol, cumulative_solution=[])
        added.update(sol.get("new", {}))
    if added_atpase and added_atpase in model.reactions:
        model.remove_reactions([added_atpase])
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="*", default=["Terrisporobacter.1", "Sedimentibacter.8"])
    ap.add_argument("--rules", default=json.dumps(RULES), help="JSON {msid: allowed direction}")
    ap.add_argument("--draft-dir", default=f"{CODIF}/models_v7c")
    ap.add_argument("--gf-dir", default=f"{CODIF}/media_gapfilled_models_v7c/Wolfe")
    ap.add_argument("--assignment", default=f"{CODIF}/modeling_files/template_assignment_v7.json")
    ap.add_argument("--media-json", default=f"{CODIF}/modeling_files/kbase_media_207617.json")
    ap.add_argument("--medium", default="Wolfe")
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
    rules = json.loads(a.rules)

    import cobra
    cobra.Configuration().solver = a.solver
    from cobra.io import read_sbml_model, write_sbml_model
    from modelseedpy import MSMedia
    from mscommunity import commcycles as CC

    bounds = medium_bounds(a.media_json, a.medium)
    assign = json.load(open(a.assignment))
    res = {"rules": rules, "members": {}}
    for mid in a.members:
        t = time.time()
        row = {}
        old = read_sbml_model(f"{a.gf_dir}/{mid}_gf.xml"); old.solver = a.solver
        with old:
            apply_medium(old, bounds)
            row["growth_before"] = old.slim_optimize(error_value=0.0)
            CC.apply_direction_rules(old, rules)
            row["growth_before_with_rules_no_regapfill"] = old.slim_optimize(error_value=0.0)
        draft = read_sbml_model(f"{a.draft_dir}/{mid}.xml"); draft.id = mid; draft.solver = a.solver
        row["draft_changed"] = CC.apply_direction_rules(draft, rules)
        cls = assign[mid]["template"]
        template, row["template_changed"] = curated_template(os.path.join(TEMPLATES, TEMPLATE_FILE[cls]), rules)
        media = MSMedia.from_dict(bounds); media.id = media.name = a.medium
        try:
            added = atp_safe_gapfill(draft, template, os.path.join(TEMPLATES, "Core-V6.json"), media)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            added = {}
        row["added"] = {k: v for k, v in added.items()}
        with draft:
            apply_medium(draft, bounds)
            draft.objective = "bio1"
            row["growth_after"] = draft.slim_optimize(error_value=0.0)
        row.update(CC.energy_from_nothing(draft, biomass="bio1"))
        row["uses_ruled_out_directions"] = [r.id for r in draft.reactions
                                            if rules.get(CC.msid(r) or "") == "<" and r.upper_bound > 0
                                            or rules.get(CC.msid(r) or "") == ">" and r.lower_bound < 0]
        row["seconds"] = round(time.time() - t, 1)
        write_sbml_model(draft, os.path.join(a.out, "models", f"{mid}_gf.xml"))
        res["members"][mid] = row
        log(f"{mid}: growth {row['growth_before']:.3g} -> rules only {row['growth_before_with_rules_no_regapfill']:.3g}"
            f" -> regapfilled {row['growth_after']:.3g}; {len(added)} added; ATP from nothing "
            f"{row['atp_from_nothing']}; e- {row['electrons_from_nothing']}; {row.get('error', '')}")
        json.dump(res, open(os.path.join(a.out, "regapfill_members.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
