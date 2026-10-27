# Energy-generating cycles across community members: pilots

Code: `mscommunity/commcycles.py` (tests: `tests/test_g_energy_cycles.py`). Data: the codiffusion
bioreactor v7c rebuild (`codiffusion_bioreactor/atp_tests/README.md` sections 4-5): 2,138
member-medium models, every one of them free of ATP and electrons from nothing on its own, and
communities that still make ATP at the flux bound with every uptake closed.

## The problem, on the nine members of the day-174 cycle (`day174_reduced.py`)

The day-174 community's cycle (found by `codiffusion_bioreactor/atp_tests/community_cycle_directions.py`)
runs through nine members. Assembled alone, from their Wolfe-gapfilled models (11,098 reactions):

| | nine members alone | community of the nine |
|---|---|---|
| ATP from nothing | 0 in each | 4,257 (8 of 9 members' ATP tests) |
| electrons from CO2/H2O/H+ | 0 in each | 0 |
| growth from nothing | 0 | 0 |
| growth on Wolfe (+H2/CO2) | 1.24 - 3.13 | 19.91 (equal abundances) |

The cycle the solver finds: 116 reactions in 7 members, 14 shared-pool metabolites passed around
(H2, CO2, acetate, formate, ethanol, methane, glutamate, ...).

## Method 1: cut directions thermodynamics rules out (`cut_energy_cycles`)

The gate closes a direction only if no concentrations in 1 uM - 20 mM could drive it
(dG'min - 2 sigma > 0, ModelSEED database dG'0). Result:

- **It finds template curations.** Three directions: `rxn00571` forward (nitrite + NAD+ -> nitrate
  + NADH, dG'0 +34.8 kcal/mol, gapfilled into Terrisporobacter.1), `rxn00359` reverse (PAPS
  formation from sulfate, +20.6, gapfilled into Desulfosporosinus.1), `rxn00288` forward
  (succinate + free FAD -> fumarate + FADH2, +20.3). The first two are '=' in the curated
  GramPos/GramNeg templates and should be `rxn00571 '<'`, `rxn00359 '>'`; `rxn00288`'s free-FAD
  form is the doubtful part (SDH runs forward to quinones).
- **Two of them carry a member's own growth.** Terrisporobacter.1 and Sedimentibacter.8 grow on
  Wolfe only through `rxn00571` / `rxn00288` forward; closing them stops their growth, and with it
  the community's. They were gapfilled onto uphill steps. `regapfill_members.py` regapfills them
  from their drafts (codiffusion's ATP-safe recipe) with the three directions closed in draft and
  template: both grow (1.75 -> 1.50, 2.42 -> 1.43), with no ATP or electrons from nothing. So the
  fix for these is curation + regapfill, not a community-level patch.
- **It cannot finish.** With all three cut (`--no-protect`) the community still makes 3,600 ATP
  from nothing, through a 106-reaction cycle whose steps are each feasible at some concentrations
  (glutamate DH with NADP+, proline DH, formaldehyde DH reversed, ATP sulfurylase, PFOR with NAD,
  threonine DH reversed, heterodisulfide reductase reversed, ... +4 to +12 kcal/mol at dG'0).
- **The "most uphill" fallback over-cuts** (`--fallback most_uphill`, protecting every member's
  growth): 43 directions cut (232 reaction bounds across members), 20 kept open because a member
  needs them, and still 1,054 ATP from nothing. Not recommended.

## Method 2: the loop law (MILP)

Three-state loop law (flux forward -> dG <= -1, reverse -> dG >= 1, no flux -> dG free) with ATP
hydrolysis pinned downhill in every member (`LocalLoopLaw`).

- On every reaction that can carry flux in the closed community (3,723, `loop_capable_reactions`,
  2 min), without a start: no solution above 0 in 30 min (the full SimpleThermoPkg loop law on all
  11,107 reactions likewise; `loop_law_pilot.py`, `day174_reduced.py --looplaw`). Started from a
  method-3 solution (`milp_start_pilot.py`): optimal, 19.91, gap 0, in 15 min.
- **Cutting planes (`loop_law_optimum`, `cutting_plane_pilot.py`)**: solve; if the solution holds a
  cycle (`conformal_cycle`, one LP over its active reactions, the ATP pin included), cover that
  cycle's reactions and solve again, warm-started from a cycle-free method-3 flux. 12 rounds,
  110 s, 1,850 reactions covered: **19.91, no cycle left** -- the loop-law optimum.
- The loop-law growth equals the LP growth here: the cycle exists, but growth on Wolfe does not
  need it.
- The pin matters (test `test_the_atp_pin_stops_a_covered_cycle_feeding_biomass`): without it the
  loop law still lets a cycle hand its ATP to biomass.

## Method 3: one potential profile (`orient_community`, LPs only)

One metabolite-potential vector for the whole community, with ATP hydrolysis pinned downhill, from
two LPs: first honour a reference flux as far as it holds no cycle, then orient as few reactions as
possible. Each reaction then keeps only the directions the profile allows. No orientation from
such a profile can run an energy-generating cycle (around a cycle the dGs sum to 0, every allowed
step has dG <= 0, ATP hydrolysis has dG < 0), including one that feeds biomass. Everything after it
is ordinary LP.

| reference | growth on Wolfe | ATP / e- from nothing | time |
|---|---|---|---|
| (none: the LP) | 19.91 | 4,257 / 0 | |
| each member alone (`"members"`) | 3.90 | 0 / 0 | 17 s |
| same, gate curations + 2 regapfilled members (single-LP profile) | 4.33 | 0 / 0 | 5 s |
| members (10x) and the LP's pFBA optimum (`"both"`) | 11.02 | 0 / 0 | 23 s |
| the LP's own pFBA optimum (`"pfba"`) | 19.61 (-1.5%) | 0 / 0 | 10 s |
| the cutting-plane loop-law optimum | **19.91** | 0 / 0 | 9 s after the 110 s cutting |

The members-alone reference is far too narrow: it fixes the directions each member uses alone and
shuts most cross-feeding. The pFBA optimum is close here, but where the optimum leans on the cycle
it is not safe (below). The loop-law optimum is exact and, with cutting planes, cheap at this size.

Two numerical lessons, both now in the code: take dG from the LP's p - q split, and treat
|dG| < 1e-5 as 0 (solver tolerance once shut a member's glucose uptake); and honour the reference
in a first LP of its own (a single weighted LP let many small orientations outweigh two
1000-flux reactions of a cycle-free reference, and dropped the hybrid from 19.91 to 7.4).

## The whole day-174 community (`full_community_profile.py`, `cutting_plane_pilot.py`)

90,721 reactions, 88 members. Unoriented LP: growth on Wolfe 18.21; ATP from nothing 30,066
(summed over the members' ATP tests, most at the 1,000 bound); electrons from CO2/H2O/H+ 100
(every member alone makes 0).

| | growth on Wolfe | ATP / e- from nothing | time |
|---|---|---|---|
| LP | 18.21 | 30,066 / 100 | 1 min |
| profile, reference "members" (3 rounds, before the two numerical fixes) | 5.90 | 0 / 0 | 53 min (scipy HiGHS) |
| profile, reference "pfba" | 6.17 (16 optimum reactions oriented against, 217 irreversible closed) | 0 / 0 | 18 min |
| profile, reference "both" | **11.83** (76 optimum reactions oriented against, 268 irreversible closed) | 0 / 0 | 14 min |
| cutting-plane loop law from the "both" start (11.32) | round 1 (MILP, 1,800 s limit): incumbent 11.81, bound 18.21 after 9 min; running (`cutting_full.log`, `results/day174_full_cutting.json`) | | |

- The electrons from nothing go too. Every member alone makes 0 and the community made 100, so
  the 100 are cross-member as well; that the orientation removes them along with the ATP is
  consistent with the cycle driving them, though the orientation also closes 268 irreversible
  directions, any of which could shut a pump on its own.
- Unlike the nine-member pilot, the full optimum leans on the cycles: the profile has to orient
  against 16-76 of its reactions, and "pfba" keeps only a third of the growth. "both" keeps
  every member growing and about two thirds. So neither reference is the better one in general:
  "pfba" when the optimum holds few cycles, "both" when it leans on them.
- The nine-member result (loop-law growth = LP growth) is about growth on Wolfe only. It does not
  make the cycles harmless: anything ATP limits -- the CH4 intervals of the measured runs above
  all -- is computed with free ATP until the community is oriented under those constraints.
- Time: the profile LP itself takes about a minute per stage with Gurobi's barrier; most of the
  14-18 min is cobra's pFBA on 90k reactions (11 min) and the after-checks. scipy's HiGHS simplex
  took about 20 min per profile LP.

## What to use

1. `energy_from_nothing` (or `MSCommunity.energy_cycles`, or `build_from_species_models(...,
   check_energy_cycles=True)`) on every assembled community: one LP.
2. Under the conditions you will simulate: `orient_community(copy, reference="both")` for a safe,
   cycle-free start; `loop_law_optimum(model, start=...)` for the loop-law optimum; then
   `orient_community(model, reference=<that optimum>)` -- one profile that keeps the optimum and
   makes every later simulation an ordinary, cycle-free LP. Where the MILP is too slow, orient
   on `reference="both"` (keeps every member growing; better when the optimum leans on cycles)
   or `"pfba"` (better when it holds few), and compare their growth with the unoriented LP.
3. `cut_energy_cycles(auto=False)` to list the directions thermodynamics rules out, as template
   curations; members that need them to grow have to be regapfilled.

Not covered: the ATP pin is on ATP only (a cycle making GTP and handing it to biomass is excluded
only where GTP is linked to ATP), and electrons created by reactions that do not conserve charge
are a curation problem, which `electrons_from_nothing` reports but nothing here removes. One
profile serves one set of conditions; build it again when the medium or the measured constraints
change.
