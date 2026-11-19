# Changelog

User-visible changes to MSCommunity. Commits are on branch `community-energy-cycles`.

## Unreleased (since f1753d5)

### Added

- New modules:
  - `mscommunity.commmembers`: member bookkeeping shared by everything below. Which reactions
    belong to which member (`member_reactions`, `reactions_by_member`, `member_transporters`),
    per-member uptake expressions and net exchange, member rates, opening and closing the member
    biomass drains, `lifted_kinetic_rows`, member biomass variables `X_m`, per-member uptake caps
    (`cap_member_uptake`) and maintenance rows (`set_member_maintenance`) on an abundance or `X`
    basis, `refresh_caps`, `remove_member_caps`, `set_member_abundances`, `member_table`.
  - `mscommunity.commcomposition`: SteadyCom balanced growth (`steadycom`, Chan et al. 2017),
    `calibrate_kappa` (each member's kappa_sat at its solo optimum), `kappa_sat_coupled`,
    `composition_profile`.
  - `mscommunity.commmicom`: MICOM's cooperative tradeoff (`cooperative_tradeoff`, `MicomResult`,
    `qp_backend`).
  - `mscommunity.commchemostat`: chemostat steady states of every surviving member subset, with
    stability, invasion rates and the supply x yield decomposition (`chemostat`, `member_block`).
  - `mscommunity.commdynamic`: multi-species dynamic FBA in two forms, `"members"` (DMMM) and
    `"community_strict"` (one community LP per step), plus `member_model`, `simulate_models` and
    `calibrate_monoculture`.
- New `MSCommunity` methods wrapping those modules: `member_table`, `member_uptake_expression`,
  `member_net_exchange`, `cap_member_uptake`, `set_member_maintenance`, `refresh_member_caps`,
  `steadycom`, `calibrate_kappa`, `kappa_sat_coupled`, `composition_profile`, `chemostat`,
  `member_model`, `dynamic`, `calibrate_monoculture`, and `micom_legacy`.
- Package exports: `SteadyComResult`, `ProfileResult`, `KappaCalibration`, `ConcentrationSwitch`,
  `SteadyState`, `SteadyStates`, `Trajectory`, `MicomResult`.
- `MSCommunity(kinetic_coeff="calibrate")` sets kappa from the members' solo optima instead of
  taking a number. It measures each member's kappa_sat with its forced fluxes (such as an ATPM
  floor) in place, through `calibrate_kappa(release_forced=False)`. It then installs the largest of
  these times (1 + 1e-6). It raises `ValueError` when no member grows alone. With floors, the
  kinetic row also sets a minimum growth near each member's solo optimum. So a coupled community
  whose members cannot all reach their solo optima at once is infeasible while the rows are on.
- `calibrate_kappa(release_forced=...)`. The default `True` releases forced member bounds, as
  `steadycom` does. `False` keeps them as rows `LB X_m <= v <= UB X_m`.
- `dynamic(form="community_strict", a_min=1e-4)`. A member with less than `a_min` of the culture is
  solved by its own member LP for that step. Every community rate is clamped to the member's solo
  maximum on the current pool. When the community LP is infeasible, the forced bounds of the
  member(s) that cannot pay them are released for that step. Each of these events is reported in
  `Trajectory.warnings`, except a clamp within solver tolerance (1e-4 relative + 1e-6 1/h). Every
  event, including those small clamps, is counted per member in `Trajectory.stats["notes"]`.
- The `t_start` integration option of `dynamic` and `simulate_models` sets the time of `X0` and
  `S0`. `calibrate_monoculture` uses it to start the simulation at the first measured time.

### Changed

- `micom()` now runs MICOM itself (Diener et al. 2020): taxon-unit bounds, stage 1 max
  sum_m a_m mu_m, stage 2 min sum_m mu_m^2 within the tradeoff, MICOM's crossover and optional
  pFBA. The signature is `micom(media=None, tradeoff=0.5, *, abundances=None, pfba=False,
  min_growth=0.0)`, with a default tradeoff of 0.5 (it was 0.6). It returns `list[MicomResult]`,
  one per medium, not cobra Solutions. Abundances are an input, and the composition is restored
  exactly on exit, also inside `with model:`. `comm.solution`, `memGrowths` and `comm_growth` are
  left as they were, so read the rates from the `MicomResult`. The old behaviour (the declared
  composition with every member at `tradeoff x mu_max`) is `micom_legacy()`, which raises a
  `DeprecationWarning`.
- `regularization()` raises `FeasibilityError` when the solve that sets the `min_comm_growth` floor
  fails. It tries the warm solve, then a cold basis, then a canonical copy. Before, the floor was
  skipped without a trace. `predict_abundances()` catches the error and keeps its documented
  contract: it logs a WARNING with the reason and returns `None`. It does the same when the
  determining solve is not optimal.
- `predict_abundances()` sets the solver timeout before its first solve and restores it on every
  exit. When the kinetic rows forbid growth and they are dropped for the call, a warning is logged
  and `self.kinetic_rows_dropped` is set.
- `set_abundance()`:
  - It accepts `{id: a}` as well as `{id: {"abundance": a, ...}}`.
  - It checks that every member is present, that each value is finite and >= 0, and that the total
    is positive. This check runs before anything is changed, and a failure raises `ValueError`.
  - It replaces only the member-biomass coefficients of bio1 and keeps every other term.
  - Inside `with model:`, bio1, the members' abundances, `comm.abundances` and `abundances_set` all
    revert on exit.
  - It does not rewrite the commmembers cap rows. `commmembers.set_member_abundances` and
    `refresh_caps` do that, and they also revert those row bounds on exiting a model context.
- `predict_abundances(update_abundances=True)` works. It used to always raise.

### Removed

- The `icecream` dependency. Its `ic()` made every later cobra model load or copy slow.

### Fixed (review of 4d8974c)

- `kinetic_coeff="calibrate"` measured kappa with forced fluxes released and installed it on the
  model where they still apply. Every member with an ATPM floor was then infeasible alone. When no
  member grew alone, it raised `TypeError`.
- `micom(abundances=...)` restored bio1 incorrectly. It left a zero-abundance member's term in
  bio1, and inside `with model:` it raised `KeyError`. It also left `comm.memGrowths` and
  `comm.comm_growth` in taxon units.
- In `dynamic(form="community_strict")`, a rare member's rate was solver noise divided by its
  abundance: 285.7 1/h at X = 1e-10 for a member that grows at 2.857 1/h. One member's unpayable
  forced flux idled every member for the whole run.
- `set_member_abundances` inside `with model:` left the cap rows at the context's composition
  after exit.
- `calibrate_monoculture` ignored the series' start time. A series sampled from 2 h fitted vmax
  1.687 for a truth of 2.0.
- An idle member never took up the substrate its maintenance burns.
- `simulate_models(copy=False)` left Gurobi's thread count changed on the caller's models.
