# -*- coding: utf-8 -*-
from modelseedpy.fbapkg.mspackagemanager import MSPackageManager
from modelseedpy.core.msmodelutl import MSModelUtil
from modelseedpy.core.exceptions import ObjectAlreadyDefinedError, FeasibilityError, NoFluxError
from modelseedpy.core.msgapfill import MSGapfill
from modelseedpy.core.fbahelper import FBAHelper
#from modelseedpy.fbapkg.gapfillingpkg import default_blacklist
from modelseedpy.core.msatpcorrection import MSATPCorrection
from mscommunity.commhelper import build_from_species_models
from mscommunity.commkineticpkg import CommKineticPkg, member_kinetic_reactions
from mscommunity.mscommviz import interactions as mscommsim_interactions
from mscommunity.batched_lp import (
    BatchedSolution,
    CommunityProblem,
    LPInstance,
    get_batched_solver,
    media_to_bounds,
)
from cobra.io import save_matlab_model, write_sbml_model
from itertools import combinations, permutations
from cobra.core.dictlist import DictList
from collections import Counter
from optlang.symbolics import Zero
from cobra.flux_analysis import pfba
from cobra import Reaction, Model
from numpy import array, logspace, linspace
from os import makedirs, path, environ
from math import isclose, isnan
from pandas import DataFrame
from pprint import pprint
from math import exp
import logging
from functools import partial

logger = logging.getLogger(__name__)

# QP-capable cobra solvers -> optlang.available_solvers capability key. Used by the
# community split-determinizer (min sum mu_i^2) to pick a backend: Gurobi/CPLEX are
# exact; 'hybrid' (HiGHS+OSQP) and 'osqp' are the strictly-open-source path (ADMM,
# ~1e-3 accuracy). GLPK/SciPy reject quadratic objectives outright.
_QP_CAPABLE = {"gurobi": "GUROBI", "cplex": "CPLEX", "hybrid": "OSQP", "osqp": "OSQP"}


def _pick_qp_backend(current=None, prefer=None):
    """Name of an available QP-capable cobra solver: an explicit `prefer`, then the
    `current` solver if it is already QP-capable (avoids a swap), then commercial
    (gurobi/cplex), then strictly open-source (hybrid=HiGHS+OSQP, then osqp). Returns
    None if no QP backend is installed."""
    import optlang
    avail = optlang.available_solvers
    order = ([prefer] if prefer else []) + ([current] if current in _QP_CAPABLE else []) \
            + ["gurobi", "cplex", "hybrid", "osqp"]
    seen = set()
    for s in order:
        if s in seen:
            continue
        seen.add(s)
        if avail.get(_QP_CAPABLE.get(s, ""), False):
            return s
    return None


def _select_biomass_cpd(entry, preferred, fallback):
    """Resolve one member-biomass metabolite from a `member_biomass_cpds` note entry.

    `commhelper.build_from_species_models` stores that note as
    {model_id: LIST of biomass metabolites} (it setdefault(...).append()s every
    cpd11416/"biomass"-named metabolite it renames), and those metabolites belong
    to the pre-copy model. `preferred` maps id -> metabolite of the live community
    model for the member-biomass compounds that feed the primary biomass;
    `fallback` maps id -> any metabolite of the live model. Returns None when the
    entry matches nothing."""
    entries = list(entry) if isinstance(entry, (list, tuple, set)) else [entry]
    hits = [preferred[cpd.id] for cpd in entries if cpd.id in preferred]
    if not hits:  hits = [fallback[cpd.id] for cpd in entries if cpd.id in fallback]
    return hits[0] if hits else None


class CommunityMember:
    def __init__(self, community, biomass_cpd, ID=None, index=None, abundance=0, model=None):
        print(ID, "biomass compound:", biomass_cpd)
        self.community, self.biomass_cpd = community, biomass_cpd
        try:     self.index = int(self.biomass_cpd.compartment[1:])
        except:  self.index = index
        self.abundance = abundance
        if self.biomass_cpd in self.community.primary_biomass.metabolites:
            self.abundance = abs(self.community.primary_biomass.metabolites[self.biomass_cpd])
        if ID is not None:  self.id = ID
        elif "species_name" in self.biomass_cpd.annotation:
            self.id = self.biomass_cpd.annotation["species_name"]
        else:  self.id = f"Species{self.index}"

        logger.info(f"Making atp hydrolysis reaction for species: {self.id}")
        if not model:
            self.model = Model()
        atp_hydrolysis_rxnComp = f"c{self.index}"
        try:
            self.atp_hydrolysis = self.community.util.model.reactions.get_by_id(f"rxn00062_{atp_hydrolysis_rxnComp}")
            print(f"skipping atp hydrolysis rxn00062_{atp_hydrolysis_rxnComp} for {self.id}")
        except:
            atp_rxn = self.community.util.add_atp_hydrolysis(atp_hydrolysis_rxnComp)
            self.atp_hydrolysis = atp_rxn["reaction"]
            if not model:
                self.model.add_reactions([self.atp_hydrolysis.copy()])
            print(f"created atp hydrolysis reaction rxn00062_{atp_hydrolysis_rxnComp} for {self.id}")
        self.biomass_drain = self.primary_biomass = None
        if not model:
            self.reactions = []
        # Classify every reaction of the community model against this member.
        #
        # Two defects used to live here.  (1) The biomass-drain test sat inside an
        # `if "bio" in rxn.id` branch, so a drain named by the ModelSEED convention
        # -- SK_cpd11416_c<i>, which is what build_from_species_models carries over
        # from the member models -- was never recognised, and the constructor then
        # built a REDUNDANT DM_cpd11416_c<i> beside it.  (2) The loop `break`ed as
        # soon as the primary biomass was found, so every reaction after it was left
        # unclassified: the drain (which follows bio<i> by three positions in a
        # ModelSEED community model) could not be seen even in principle, and
        # `self.reactions` -- and therefore the member's standalone `self.model` --
        # was silently truncated to whatever preceded the biomass reaction.
        for rxn in self.community.util.model.reactions:
            mets = {met.id: met for met in rxn.metabolites}
            if self.biomass_cpd.id in mets:
                met = mets[self.biomass_cpd.id]
                ## this member's own biomass reaction produces one unit of its biomass
                if rxn.metabolites[met] == 1 and len(rxn.metabolites) > 1:
                    self.primary_biomass = rxn
                    continue
                ## a drain on this member's biomass: a boundary reaction that consumes
                ## it and nothing else, whatever prefix it happens to carry
                if len(rxn.metabolites) == 1 and rxn.metabolites[met] < 0:
                    self.biomass_drain = rxn
                    continue
            ## biomass reactions (this member's, another member's, or the community's)
            ## never count as member metabolism
            if "bio" in rxn.id:  continue
            rxnComp = FBAHelper.rxn_compartment(rxn)
            if rxnComp is None:  print(f"The reaction {rxn.id} compartment {rxnComp} is undefined.")
            elif rxnComp[1:] == '': print("no compartment", rxn, rxnComp)
            elif int(rxnComp[1:]) == self.index:  self.reactions.append(rxn)

        if self.primary_biomass is None:  print(f"No biomass reaction found for species {self.id}")
        if not self.biomass_drain:
            print(f"Making biomass drain reaction for species: {self.id}")
            self.biomass_drain = Reaction(id=f"DM_{self.biomass_cpd.id}", name=f"DM_{self.biomass_cpd.name}", lower_bound=0, upper_bound=100)
            self.community.util.model.add_reactions([self.biomass_drain])
            self.biomass_drain.add_metabolites({self.biomass_cpd: -1})
            self.biomass_drain.annotation["sbo"] = 'SBO:0000627'
        # the drain's as-built bounds, kept so that MSCommunity can close the drain for
        # the coupled community solve and still restore it where a member is measured
        # on its own (see MSCommunity._solo_max_batch)
        self.biomass_drain_bounds = tuple(self.biomass_drain.bounds)
        # reactions = self.reactions + [self.primary_biomass, self.biomass_drain]
        # print(Counter([rxn.id for rxn in reactions]))
        # TODO the best way of tracking the models may be to run build_from_species_models inside the MSCommunity class
        ## where the models are available and build available.
        self.model.add_reactions([rxn.copy() for rxn in self.reactions + [self.primary_biomass, self.biomass_drain]])
        self.model.add_reactions([rxn.copy() for rxn in self.community.util.exchange_list()])
        self.model.medium = self.community.util.model.medium
        self.model.objective = self.model.reactions.get_by_id(self.primary_biomass.id).flux_expression

    def disable_species(self):
        for reaction in self.community.util.model.reactions:
            reaction_index = FBAHelper.rxn_compartment(reaction)[1:]
            if int(reaction_index) == self.index:  reaction.upper_bound = reaction.lower_bound = 0

    def compute_max_biomass(self):
        if self.primary_biomass is None:  logger.critical("No biomass reaction found for species "+self.id)
        self.community.util.add_objective(self.primary_biomass.flux_expression)
        if self.community.lp_filename:  self.community.print_lp(f"{self.community.lp_filename}_{self.id}_Biomass")
        return self.community.model.optimize()

    def compute_max_atp(self):
        if not self.atp_hydrolysis: logger.critical("No ATP hydrolysis found for species:" + self.id)
        self.community.util.add_objective(Zero, coef={self.atp_hydrolysis.forward_variable: 1})
        if self.community.lp_filename:  self.community.print_lp(f"{self.community.lp_filename}_{self.id}_ATP")
        return self.community.model.optimize()


class MSCommunity:
    def __init__(self, model=None, member_models: list = None, abundances=None, ids=None, kinetic_coeff=750,
                 flux_limit=300, probs=None, climit=None, o2limit=None, lp_filename=None, printing=False, eleLimits=None, ID=None,
                 close_member_drains=False):
        """A community model with one CommunityMember per member biomass compound.

        model / member_models  the merged community model, or the member models to merge
                               (commhelper.build_from_species_models)
        abundances             {member id: a} (or {id: {"abundance": a, ...}}), normalised to sum 1
                               and written into bio1; default equal abundances
        kinetic_coeff          kappa of the community-kinetics rows installed at construction, or
                               "calibrate" to set it to the smallest single kappa at which every
                               member still reaches its own solo optimum (calibrate_kappa)
                               (<member>_commKin: sum_r |v_r| <= kappa * v_bio_m over the member's
                               metabolism). They are ALWAYS installed. The default 750 can forbid
                               measured monocultures: it is below the solo saturation kappa of
                               P. fluorescens on acetate (751.0) and of both SynCom members (1103.6,
                               1421.6), so at 750 those members cannot grow alone at their measured
                               rates. Calibrate kappa for the system, or lift the rows
                               (commmembers.lifted_kinetic_rows) where they are not meant to bind.
        close_member_drains    False (default): each member's biomass drain stays open, so a member
                               can make biomass and discard it. Member growth is then free of the
                               declared abundances (what predict_abundances and _solo_max_batch
                               need), but each unit drained buys (kappa - 1) units of kinetic budget:
                               on a two-member SynCom 83 % of one member's biomass left through its
                               drain at the lowest feasible kappa, and closing the drains raised
                               that kappa from 1285 to 1481.
                               True: drains shut (0, 0), so member biomass leaves only through bio1
                               and v_bio_m = a_m * v_bio1: every member grows at mu_m = mu_com, an
                               identity of the formulation that carries no information about who
                               grows faster (commmembers.member_rates). Use it when the declared
                               composition is meant to BIND.
        flux_limit             unused (kept for old callers)
        probs, climit, o2limit, lp_filename, printing, eleLimits, ID  as before."""
        # `flux_limit` is UNUSED: nothing in the package reads it. It is kept only so
        # existing positional/keyword callers do not break; pass it or not, it has no
        # effect. `probs` was a mutable {} default shared by every instance (mutating
        # one community's rxnProbs would silently edit the default) -> None sentinel.
        #
        # `close_member_drains` shuts each member's biomass drain, so member biomass can
        # leave only through bio1 and every member's growth is pinned to
        # abundance * bio1. An open drain lets a member synthesise biomass and discard
        # it, so its own biomass flux exceeds abundance * bio1 and the CommKinetics
        # right-hand side (kinetic_coef * member biomass) inflates with it: each unit
        # drained buys (kinetic_coef - 1) units of flux budget. Measured on a two-member
        # SynCom, 83% of one member's biomass went out the drain at the lowest feasible
        # kinetic coefficient, and closing the drains raised that coefficient from 1285
        # to 1481.
        #
        # It defaults to False because the drain is load-bearing elsewhere in this
        # class: `_solo_max_batch` and `predict_abundances*` deliberately let a member
        # grow while the others are switched off, which zeroes bio1 and leaves the drain
        # as the only outlet for that member's biomass. (`_solo_max_batch` reopens the
        # target's drain for its own LPs, so it works either way.) Set it True whenever
        # the declared abundance vector is meant to BIND -- any run where you read
        # member growth rates or a kinetic coefficient off the solution.
        assert model is not None or member_models is not None, "Either the community model and the member models must be defined."
        probs = probs if probs is not None else {}
        self.lp_filename = lp_filename
        self.printing = printing
        self.gapfillings = {}

        #Define Data attributes as None
        for attr in ['solution', 'biomass_cpd', 'primary_biomass', 'biomass_drain', 'threshold', 'msgapfill', 
                     'element_uptake_limit', 'msdb_path', 'comm_growth', 'threshold', "memGrowths", "member_fluxes",
                     "suboptimal_solution"]:
            setattr(self, attr, None)
        self.kinCoef = 1e6 if isinstance(kinetic_coeff, str) else kinetic_coeff  # "calibrate": a placeholder until calibrated below
        # Determinism instrumentation: whether the last determining coculture
        # solve had to fall back from pFBA/QP to the plain (solver-degenerate)
        # LP, and how many times that has happened on this community. Callers
        # read these to flag rows whose per-member split is NOT converged.
        self.pfba_fell_back = False
        self.pfba_fallback_count = 0
        # whether the last predict_abundances call dropped the *_commKin rows (see there)
        self.kinetic_rows_dropped = False
        # defining the models
        if model is None and member_models is not None:
            model = build_from_species_models(member_models, abundances=abundances, printing=printing)
        self.id = ID or model.id
        # self.modelID_names = model.notes["modelID_names"]
        self.util = MSModelUtil(model, True, None, climit, o2limit)
        msid_cobraid_hash = self.util.msid_hash()  # dict of list() of metabolite objects by their msid
        if "cpd11416" not in msid_cobraid_hash:  raise KeyError("Could not find biomass compound for the model.")
        other_biomass_cpds = []
        for self.biomass_cpd in msid_cobraid_hash["cpd11416"]:
            if "c0" in self.biomass_cpd.id:
                for rxn in self.util.model.reactions:
                    if self.biomass_cpd not in rxn.metabolites:  continue
                    # print(self.biomass_cpd, rxn, end=";\t")
                    if rxn.metabolites[self.biomass_cpd] == 1 and len(rxn.metabolites) > 1:
                        if self.primary_biomass:  raise ObjectAlreadyDefinedError(
                            f"The primary biomass {self.primary_biomass} is already defined,"
                            f"hence, the {rxn.id} cannot be defined as the model primary biomass.")
                        if printing:  print('primary biomass defined', rxn.id)
                        self.primary_biomass = rxn
                    elif rxn.metabolites[self.biomass_cpd] < 0 and len(rxn.metabolites) == 1:  self.biomass_drain = rxn
            elif 'c' in self.biomass_cpd.compartment:   other_biomass_cpds.append(self.biomass_cpd)
        
        # lookups into the LIVE (possibly copied) community model, since
        # model.notes["member_biomass_cpds"] holds pre-copy metabolite objects
        memberBioCPDs = {cpd.id: cpd for cpd in other_biomass_cpds}
        allModelCPDs = {met.id: met for met in self.util.model.metabolites}
        if ids is None:
            if member_models is not None:   ids = [mem.id for mem in member_models]
            else:  ids = [f"Species{i}" for i in range(len(other_biomass_cpds))]
        # memberIDs_biomass = dict(zip(ids,
        if not abundances:
            if member_models is None:
                abundances = {ids[memIndex]: {"biomass_compound": bioCpd, "abundance": 1/len(other_biomass_cpds)}
                              for memIndex, bioCpd in enumerate(other_biomass_cpds)}
            else:
                abundances = {}
                for memID, bioCPDs in model.notes["member_biomass_cpds"].items():
                    abundances[memID] = {"abundance": 1/len(other_biomass_cpds)}
                    met = _select_biomass_cpd(bioCPDs, memberBioCPDs, allModelCPDs)
                    if met is None:   print(f"The {memID} bioCPD was not captured")
                    else:  abundances[memID].update({"biomass_compound": met})
        elif not isinstance(list(abundances.values())[0], dict):  # plain floats ("abundance" in <float> raised TypeError)
            abundances = {memID:{"abundance": abund,
                                 "biomass_compound": _select_biomass_cpd(
                                     model.notes["member_biomass_cpds"][memID], memberBioCPDs, allModelCPDs)}
                            for memID, abund in abundances.items()}

        # print()   # this returns the carriage after the tab-ends in the biomass compound printing
        self.members = DictList(CommunityMember(self, info["biomass_compound"], ID, index+1, info["abundance"])
                                for index, (ID, info) in enumerate(abundances.items()))
        # self.members = DictList(
        #     CommunityMember(community=self, biomass_cpd=biomass_cpd, name=ids[memIndex], abundance=abundances[memIndex])
        #     for memIndex, biomass_cpd in enumerate(other_biomass_cpds))
        self.set_abundance(abundances)

        # close the member biomass drains (see the note in the docstring above)
        self.close_member_drains = close_member_drains
        if close_member_drains:
            for member in self.members:
                if member.biomass_drain is not None:
                    member.biomass_drain.bounds = (0, 0)

        # assign the MSCommunity constraints and objective
        self.rxnProbs = probs
        self.pkgmgr = MSPackageManager.get_pkg_mgr(self.util.model)
        kinetic_pkg = CommKineticPkg(self.util.model)
        self.pkgmgr.addpkgobj(kinetic_pkg)
        calibrate = isinstance(kinetic_coeff, str)
        if calibrate and kinetic_coeff != "calibrate":
            raise ValueError(f"kinetic_coeff must be a number or 'calibrate', not {kinetic_coeff!r}")
        if calibrate:
            # the smallest single kappa that forbids no member's solo optimum (commcomposition);
            # computed before the rows exist (calibrate_kappa then uses the terms add_commkinetics
            # would write), because the package cannot rebuild rows it has already registered
            from mscommunity.commcomposition import calibrate_kappa
            kinetic_coeff = max(calibrate_kappa(self, per_member=False).values())
            self.kinCoef = kinetic_coeff
            logger.info("kinetic_coeff calibrated to %s", kinetic_coeff)
        kinetic_pkg.build_package(kinetic_coeff, self, self.rxnProbs)
        if eleLimits is not None:
            self.pkgmgr.getpkg("ElementUptakePkg").build_package(eleLimits)
        # if kinetic_coeff is not None:   self.add_commkinetics(kinetic_coeff, probs)
        

    #Manipulation functions
    def set_abundance(self, abundances):
        old_abundance = {mem.id: mem.abundance for mem in self.members}
        old_abundances = getattr(self, "abundances", None)
        old_set = getattr(self, "abundances_set", False)
        #calculate the normalized biomass
        total_abundance = sum(list([content["abundance"] for content in abundances.values()]))
        # map abundances to all species
        for modelID, content in abundances.items():
            if modelID in self.members:  self.members.get_by_id(modelID).abundance = content["abundance"]/total_abundance
        self.abundances = {mem.id: mem.abundance for mem in self.members}
        #remake the primary biomass reaction based on abundances  #TODO what is the purpose of this?
        if self.primary_biomass is None:  logger.critical("Primary biomass reaction not found in community model")
        # Write bio1 with the model's own metabolite objects, outside cobra's change tracking, and
        # register one undo step that restores the previous stoichiometry and abundances. A member's
        # stored biomass_cpd can be a different object with the same id (the SynCom build), and a
        # zero abundance drops the metabolite from bio1; cobra's per-metabolite undo record raised
        # KeyError on either inside `with model:`.
        model = self.util.model
        mets = model.metabolites
        all_metabolites = {self.primary_biomass.products[0]: 1}
        all_metabolites.update({mets.get_by_id(mem.biomass_cpd.id): -abundances[mem.id]["abundance"]/total_abundance
                                for mem in self.members})
        context = model._contexts[-1] if getattr(model, "_contexts", None) else None
        if context is not None:
            previous = (dict(self.primary_biomass.metabolites), dict(old_abundance), old_abundances, old_set)
            context(partial(self._restore_abundance_state, *previous))
        self._write_bio1(all_metabolites)
        self.abundances_set = True

    def _write_bio1(self, stoichiometry):
        """Replace bio1's stoichiometry without recording cobra undo steps (set_abundance records its own)."""
        model = self.util.model
        contexts, model._contexts = getattr(model, "_contexts", []), []
        try:
            self.primary_biomass.add_metabolites(
                {met: -coef for met, coef in self.primary_biomass.metabolites.items()}, combine=True)
            self.primary_biomass.add_metabolites(stoichiometry, combine=True)
        finally:
            model._contexts = contexts

    def _restore_abundance_state(self, stoichiometry, member_abundance, abundances, abundances_set):
        self._write_bio1(stoichiometry)
        for mem in self.members:
            if mem.id in member_abundance:  mem.abundance = member_abundance[mem.id]
        self.abundances = abundances
        self.abundances_set = abundances_set

    def set_objective(self, target=None, targets=None, weights=None, minimize=False):
        targets = targets or [self.util.model.reactions.get_by_id(target or self.primary_biomass.id).flux_expression]
        if weights is not None:   targets = [t*w for t, w in zip(targets, weights)]
        self.util.model.objective = self.util.model.problem.Objective(sum(targets), direction="max" if not minimize else "min")

    def constrain(self, element_uptake_limit=None, thermo_params=None, msdb_path=None):
        if element_uptake_limit:
            self.element_uptake_limit = element_uptake_limit
            self.pkgmgr.getpkg("ElementUptakePkg").build_package(element_uptake_limit)
        if thermo_params:
            if msdb_path:
                self.msdb_path = msdb_path
                thermo_params.update({'modelseed_db_path':msdb_path})
                self.pkgmgr.getpkg("FullThermoPkg").build_package(thermo_params)
            else:  self.pkgmgr.getpkg("SimpleThermoPkg").build_package(thermo_params)

    def interactions(self, solution=None, media=None, msdb=None, msdb_path=None, filename=None, figure_format="svg",
                     node_metabolites=True, flux_threshold=1, visualize=True, ignore_mets=None):
        return mscommsim_interactions(self, solution or self.solution, media, flux_threshold, msdb, msdb_path,
                                        visualize, filename, figure_format, node_metabolites, True, ignore_mets)

    def add_commkinetics(self, kinCoef=750, probs={}):  #, abundances):
        self.rxnProbs = probs
        self.kinCoef = kinCoef
        for member in self.members:
            ## remove existing instance of CommKinetics
            consName = f"{member.id}_commKin"
            if consName in self.util.model.constraints:
                print(f"Removing {consName} from {self.util.model.id}")
                self.util.model.remove_cons_vars(self.util.model.constraints[consName])
            ## define the CommKinetics constraint:  kinCoef * bio_f,i > kinCoef * bio_r,i + sum(rxn_i * prob_r)
            ## membership comes from commkineticpkg.member_kinetic_reactions, the single
            ## definition shared with CommKineticPkg -- the two used to apply different
            ## exclusion rules ("bio" not in the id here, identity there), which disagreed
            ## on the community biomass reaction and on the member biomass drains.
            coef = {member.primary_biomass.forward_variable: -kinCoef, member.primary_biomass.reverse_variable: kinCoef}
            for rxn in member_kinetic_reactions(self, member):
                coef[rxn.forward_variable] = coef[rxn.reverse_variable] = self.rxnProbs.get(rxn.id, 1)
            self.util.create_constraint(self.util.model.problem.Constraint(Zero, name=consName, ub=0), coef=coef, printing=True)

    #Utility functions
    def print_lp(self, filename=None):
        filename = filename or self.lp_filename
        # a bare relative filename has no dirname; makedirs("") raises FileNotFoundError
        if path.dirname(filename):  makedirs(path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as out:  out.write(str(self.util.model.solver))  ;  out.close()

    def to_sbml(self, export_name):
        makedirs(path.dirname(export_name), exist_ok=True)
        write_sbml_model(self.util.model, export_name)

    #Analysis functions
    def gapfill(self, media = None, target = None, minimize = False, default_gapfill_templates=None, default_gapfill_models=None,
                test_conditions=None, reaction_scores=None, blacklist=None, suffix = None, solver:str="glpk"):
        default_gapfill_templates = default_gapfill_templates or []
        default_gapfill_models = default_gapfill_models or []
        test_conditions, blacklist = test_conditions or [], blacklist or []
        reaction_scores = reaction_scores or {}
        if not target:  target = self.primary_biomass.id
        self.set_objective(target, minimize)
        gfname = FBAHelper.mediaName(media) + "-" + target
        if suffix:  gfname += f"-{suffix}"
        # MSGapfill's 7th positional is now `atp_gapfilling`, not solver: set the solver
        # on the model instead of passing it through (a truthy string silently switched
        # every community gapfill into ATP-gapfilling mode).
        if solver:
            self.util.model.solver = solver
        self.gapfillings[gfname] = MSGapfill(self.util.model, default_gapfill_templates, default_gapfill_models,
                                             test_conditions, reaction_scores, blacklist)
        gfresults = self.gapfillings[gfname].run_gapfilling(media, target)
        assert gfresults, f"Gapfilling of {self.util.model.id} in {gfname} towards {target} failed."
        return self.gapfillings[gfname].integrate_gapfill_solution(gfresults)

    def energy_cycles(self, electrons=True, release_floors=True, closure="lax", biomass_outlet=True,
                      boundary_ions=False, pins=None):
        """ATP, growth and electrons the community makes from nothing (commcycles.energy_from_nothing).
        Each is 0 for a community without energy-generating cycles, including the ones that run
        across members, which no single-member test can see.

        The community-kinetics rows are lifted for the test (energy_from_nothing does it itself
        now, and lists them in report["kinetic_rows_lifted"]): they cap each member's flux at
        kinetic_coeff x its growth, so with nothing coming in they hold every flux at 0 and would
        hide the cycle -- which still runs, up to that cap, whenever the community grows.
        Forced fluxes (maintenance-ATP floors) are released for the test too (release_floors),
        else the sealed community is infeasible rather than 0; report["released"] lists them.
        closure "lax" lets the boundary still remove matter, "strict" closes it both ways.
        boundary_ions=True adds the arms that reopen H+ and Pi (a route through them is invisible
        to the sealed test); pins="ntp" adds GTP, CTP, UTP and PPi hydrolysis tests."""
        from mscommunity.commcycles import energy_from_nothing
        return energy_from_nothing(self.util.model, biomass=self.primary_biomass.id, electrons=electrons,
                                   release_floors=release_floors, closure=closure,
                                   biomass_outlet=biomass_outlet, boundary_ions=boundary_ions, pins=pins)

    def orient_by_thermodynamics(self, reference="pfba", rounds=1, media=None, **profile_args):
        """Remove every energy-generating cycle, including those across members, by giving each
        reaction only the directions one metabolite-potential profile allows (ATP hydrolysis
        pinned downhill): commcycles.orient_community, changing this community's bounds in place.
        The profile is built on the community's current medium and objective, from `reference`
        ("pfba": its own optimum; "both": each member alone, then the optimum; "members": each
        member alone only, not recommended; or a flux dict such as a loop-law solution), whose
        objective it keeps where the reference holds no cycle. "pfba" and "both" are the
        recommended references. The solves run on a canonical copy (canonical=True), so the
        same configuration and threads give the same orientation; pass threads= for that.
        Returns the report; report["changed"] lists every bound changed."""
        from mscommunity.commcycles import orient_community
        if media is not None:
            self.util.add_medium(media)
        self.set_objective()
        return orient_community(self.util.model, reference=reference, rounds=rounds, **profile_args)

    # ---- member bookkeeping (mscommunity.commmembers) ------------------------------------
    def member_table(self):
        """One row per member: abundance, biomass and drain, ATPM, compartments, transporters by
        compound, and the caps installed (commmembers.member_table)."""
        from mscommunity.commmembers import member_table
        return member_table(self)

    def member_uptake_expression(self, member, cpd, **kwargs):
        """The member's net (or gross=True) uptake of compound `cpd` from e0, per gDW of community
        (commmembers.member_uptake_expression)."""
        from mscommunity.commmembers import member_uptake_expression
        return member_uptake_expression(self, member, cpd, **kwargs)

    def member_net_exchange(self, member, solution=None, **kwargs):
        """{cpd: net uptake from e0} for one member at a solution (default: the last one),
        per gDW of community (commmembers.member_net_exchange)."""
        from mscommunity.commmembers import member_net_exchange
        return member_net_exchange(self, solution if solution is not None else self.solution, member, **kwargs)

    def cap_member_uptake(self, member, cpd=None, *, q, **kwargs):
        """Cap the member's gross uptake at q x a_m (basis="abundance") or q x X_m (basis="X"),
        q in mmol per gDW of THAT member per h, on its own transporters
        (commmembers.cap_member_uptake). Abundance caps follow set_abundance only through
        refresh_member_caps()."""
        from mscommunity.commmembers import cap_member_uptake
        return cap_member_uptake(self, member, cpd, q=q, **kwargs)

    def set_member_maintenance(self, member, A_m, **kwargs):
        """ATP maintenance of the member >= A_m x a_m (or x X_m), A_m per gDW of that member
        (commmembers.set_member_maintenance)."""
        from mscommunity.commmembers import set_member_maintenance
        return set_member_maintenance(self, member, A_m, **kwargs)

    def refresh_member_caps(self):
        """Rewrite abundance-basis caps and maintenance rows from the current abundances."""
        from mscommunity.commmembers import refresh_caps
        return refresh_caps(self)

    # ---- composition (mscommunity.commcomposition) ---------------------------------------
    def steadycom(self, **kwargs):
        """Balanced growth with member biomass variables X_m (SteadyCom, Chan et al. 2017),
        reporting composition RANGES, solo rates and corner/relabelling warnings. A max-growth
        balanced-growth model that reproduces the monocultures cannot predict coexistence below
        the fastest solo rate. See commcomposition.steadycom."""
        from mscommunity.commcomposition import steadycom
        return steadycom(self, **kwargs)

    def calibrate_kappa(self, per_member=False, **kwargs):
        """kappa_sat per member: the smallest kinetic coefficient at which that member alone still
        reaches its kappa-free solo optimum (commcomposition.calibrate_kappa)."""
        from mscommunity.commcomposition import calibrate_kappa
        return calibrate_kappa(self, per_member, **kwargs)

    def kappa_sat_coupled(self, abundances, **kwargs):
        """kappa at which the coupled (fixed-composition) form saturates at a declared composition.
        predict_abundances at this kappa returns that composition (commcomposition)."""
        from mscommunity.commcomposition import kappa_sat_coupled
        return kappa_sat_coupled(self, abundances, **kwargs)

    def composition_profile(self, **kwargs):
        """mu_com(a) on a grid (objective="coupled") or the composition range near the
        composition-free optimum (objective="free"); report it with any composition answer
        (commcomposition.composition_profile)."""
        from mscommunity.commcomposition import composition_profile
        return composition_profile(self, **kwargs)

    # ---- chemostat and dynamic simulation -----------------------------------------------
    def chemostat(self, D, feed, *, kinetics, **kwargs):
        """Steady states (mu = D) of every surviving member subset, with stability, invasion rates
        and the supply x yield decomposition (commchemostat.chemostat)."""
        from mscommunity.commchemostat import chemostat
        return chemostat(self, D, feed, kinetics=kinetics, **kwargs)

    def member_model(self, member, repair=None, subset=None, **kwargs):
        """The member's own stand-alone model, cut from the community (commdynamic.member_model)."""
        from mscommunity.commdynamic import member_model
        return member_model(self, member, repair=repair, subset=subset, **kwargs)

    def dynamic(self, t_end, dt, X0, S0, *, kinetics, **kwargs):
        """Multi-species dynamic FBA with member biomass states (form "members" or
        "community_strict"); returns a Trajectory (commdynamic.dynamic)."""
        from mscommunity.commdynamic import dynamic
        return dynamic(self, t_end, dt, X0, S0, kinetics=kinetics, **kwargs)

    def calibrate_monoculture(self, member, timeseries, params=("vmax",), *, repair=None, **kwargs):
        """Fit a member's uptake kinetics to its monoculture time course
        (commdynamic.calibrate_monoculture on member_model(member))."""
        from mscommunity.commdynamic import calibrate_monoculture, member_model
        mem = self.members.get_by_id(member) if isinstance(member, str) else member
        return calibrate_monoculture(member_model(self, mem, repair=repair), timeseries, params,
                                     member_id=mem.id, **kwargs)

    def to_neural_network(self, inputs=None, media=None, **surrogate_args):
        """This community as the ANN surrogate of Song et al. (2025): uptake limits on the shared
        e0 exchanges in, each member's biomass, the community biomass and the actual exchange
        fluxes out, as one closed-form tanh network trained on sampled FBA solutions
        (commsurrogate.community_to_nn). `inputs` defaults to every exchange of the medium; the
        community is sampled as configured, objective and constraints included. Returns the
        FBASurrogate, whose `scores` hold the R^2 of each output on held-out samples."""
        from mscommunity.commsurrogate import community_to_nn
        if media is not None:
            self.util.add_medium(media)
        return community_to_nn(self, inputs, **surrogate_args)

    def test_individual_species(self, media=None, interacting=True, run_atp=True, run_biomass=True):
        assert run_atp or run_biomass, ValueError("Either the run_atp or run_biomass arguments must be True.")
        # self.pkgmgr.getpkg("KBaseMediaPkg").build_package(media)
        if media is not None:  self.util.add_medium(media)
        data = {"Species": [], "Biomass": [], "ATP": []}
        for individual in self.members:
            data["Species"].append(individual.id)
            with self.util.model:
                if not interacting:
                    for other in self.members:
                        if other != individual:  other.disable_species()
                if run_biomass:  data["Biomass"].append(individual.compute_max_biomass())
                if run_atp:  data["ATP"].append(individual.compute_max_atp())
        return DataFrame(data)

    def atp_correction(self, core_template, atp_medias, max_gapfilling=None, gapfilling_delta=0):
        self.atp = MSATPCorrection(self.util.model, core_template, atp_medias, "c0", max_gapfilling, gapfilling_delta)

    def _growth_fraction(self):
        growth_multiple = 0.999 * exp(-0.03 * len(self.members))
        if self.printing:  print(f"The growth multiple is {growth_multiple}")
        return growth_multiple

    def regularization(self, linear=True, growth_constraint=True, batch_backend="cpu", batch_workers=1):
        """Install the scaffolding of predict_abundances(regularization=True) on the live model:

        growth_constraint  min_comm_growth: bio1 >= growth_fraction * max(bio1), where
                           growth_fraction = 0.999 exp(-0.03 n_members). bio1's coefficients are
                           the DECLARED abundances, so this floor pulls the predicted composition
                           towards the declared one (see predict_abundances).
        linear             <member>_regularization: v_bio_m >= ratio * solo_max_m for every member
                           with a solo capacity, ratio the strictest feasible rung of
                           0.7, 0.5, 0.3, 0.2, 0.1, 0.05. Solo capacities are measured first,
                           with any min_comm_growth suspended (_solo_max_batch).
        linear=False installs no per-member rows (the least-squares form is not implemented).

        The solve that sets min_comm_growth is checked on its STATUS. GLPK can report a spurious
        'infeasible' from a warm basis (CommPhitting maltose: the same LP is optimal at 0.2800 on
        Gurobi and on GLPK from a cold basis); skipping the floor then silently returned the 70 %
        solo floor (a_ec 0.6994 against 0.8187). A non-optimal status is re-solved from a cold
        basis (commcycles.reset_start) and then on a canonical copy (commcycles.canonical_copy);
        if all three fail, FeasibilityError is raised. An optimal max(bio1) <= 1e-6 installs no
        floor and logs a warning. Each rung of the ratio ladder that is not optimal is likewise
        re-solved once from a cold basis before it is rejected. Returns None."""
        from mscommunity.commcycles import canonical_copy, reset_start
        self.util.remove_constraint("_regularization")
        # PATCH 5: the per-member solo capacities MUST be measured with the community
        # growth floor suspended. min_comm_growth pins bio1 >= f*max, but bio1 consumes
        # EVERY member's biomass metabolite and nothing else produces them, so a solo
        # LP -- which pins the other members' biomass reactions to (0, 0) -- forces
        # v_bio1 = 0 and contradicts any strictly positive floor: every solo LP came
        # back infeasible and every member's capacity was recorded as garbage/zero, so
        # no per-member floor was ever installed. Measure first (and defensively drop a
        # floor left over from an earlier call, restoring it via the model context),
        # then install the floor for the ratio loop and the determining solve below.
        solo_max = None
        if linear:
            leftoverFloor = [cons for cons in self.util.model.constraints if cons.name == "min_comm_growth"]
            with self.util.model:
                if leftoverFloor:  self.util.model.remove_cons_vars(leftoverFloor)
                solo_max = self._solo_max_batch(backend=batch_backend, workers=batch_workers)
            # the context manager restores any suspended floor on exit
        # PATCH 1: min_comm_growth binds bio1 (community biomass), so its LB
        # must come from the max of bio1 — not the max of whatever objective
        # the caller currently has set (e.g. sum-of-member-biomasses).
        if growth_constraint:
            self.util.remove_constraint("min_comm_growth")
            model = self.util.model
            ogObj_for_bio1 = model.objective
            model.objective = model.problem.Objective(self.primary_biomass.flux_expression, direction="max")
            # MS5: the floor used to be skipped whenever this solve was not optimal, without a
            # trace. Read the status, and retry from a cold basis, then on a canonical copy.
            statuses = {}
            try:
                commMax_bio1 = model.slim_optimize(error_value=float("nan"))
                statuses["warm"] = status = model.solver.status
                if status != "optimal":
                    logger.warning("regularization: max bio1 on %s is %s; re-solving from a cold basis",
                                   model.id, status)
                    reset_start(model)
                    commMax_bio1 = model.slim_optimize(error_value=float("nan"))
                    statuses["cold basis"] = status = model.solver.status
                if status != "optimal":
                    logger.warning("regularization: max bio1 on %s is %s from a cold basis; re-solving "
                                   "on a canonical copy", model.id, status)
                    work = canonical_copy(model)
                    work.objective = work.problem.Objective(
                        work.reactions.get_by_id(self.primary_biomass.id).flux_expression, direction="max")
                    commMax_bio1 = work.slim_optimize(error_value=float("nan"))
                    statuses["canonical copy"] = status = work.solver.status
            finally:
                model.objective = ogObj_for_bio1
            if status != "optimal":
                raise FeasibilityError(
                    f"regularization: the solve that sets min_comm_growth (max {self.primary_biomass.id}) "
                    f"on {model.id} is not optimal ({', '.join(f'{k}: {v}' for k, v in statuses.items())}); "
                    "the community-growth floor cannot be installed, and without it predict_abundances "
                    "would return the per-member solo floors instead of its regularized composition.")
            if len(statuses) > 1:
                logger.warning("regularization: max bio1 on %s is %.6g (%s)", model.id, commMax_bio1,
                               ", ".join(f"{k}: {v}" for k, v in statuses.items()))
            if commMax_bio1 > 1e-6:
                self.util.create_constraint(model.problem.Constraint(
                    self.primary_biomass.flux_expression, name="min_comm_growth",
                    lb=commMax_bio1 * self._growth_fraction()), printing=True)
            else:
                logger.warning("regularization: max bio1 on %s is %.3g <= 1e-6; no min_comm_growth floor "
                               "is installed", model.id, commMax_bio1)
        if linear:
            # PATCH 4: relative (per-member) regularization. The previous
            # scheme bounded |mu_i - mu_j| <= threshold absolutely, which
            # forced near-equal growth and collapsed real asymmetries
            # (parasitism / commensalism / dominance). Instead, compute each
            # member's solo capacity (what it could reach if the others were
            # forced to zero growth in the same community model) and require
            # each viable member to keep at least `ratio` of its own solo
            # max. Members with zero solo capacity get no constraint, so a
            # surviving member can grow at its full rate while a non-viable
            # partner stays at zero.
            # The N solo-max LPs all share the community S — only bounds and
            # objective differ — so route them through the batched solver. They are
            # solved above, BEFORE min_comm_growth is installed (see PATCH 5).
            # Iterate from tight to loose so we land on the strictest feasible
            for ratio in [0.7, 0.5, 0.3, 0.2, 0.1, 0.05]:
                self.util.remove_constraint("_regularization")
                for mem in self.members:
                    if solo_max[mem.id] > 1e-6:
                        consName = f"{mem.id}_regularization"
                        coef = {mem.primary_biomass.forward_variable: 1}
                        self.util.create_constraint(self.util.model.problem.Constraint(
                            Zero, name=consName, lb=ratio * solo_max[mem.id]), coef=coef, printing=True)
                # BUG FIX (Layer 0 #2): we only need to know whether this ratio is
                # FEASIBLE, not a vertex of the degenerate max-sum face. The old
                # `.optimize()` materialised a full Solution whose per-member flux
                # split is solver-dependent (GLPK vs Gurobi pick different vertices),
                # and in borderline cases that vertex pick could nudge which ratio
                # bucket is accepted. slim_optimize() reads only the objective /
                # status, so the ratio chosen is a pure feasibility decision and the
                # floors handed downstream are solver-independent.
                self.util.model.slim_optimize()
                if self.util.model.solver.status != "optimal":
                    # a rung is rejected on its status, so a spurious warm-basis 'infeasible'
                    # (MS5) would silently loosen the floors: confirm it from a cold basis
                    reset_start(self.util.model)
                    self.util.model.slim_optimize()
                if self.util.model.solver.status == "optimal":
                    print(f"The model {self.util.model.id} is regularized: each viable member keeps >={ratio*100:.0f}% of its solo capacity")
                    break
            else:
                logger.warning("regularization: no rung of the ratio ladder is feasible on %s; the 5 %% "
                               "floors stay installed and the next solve is infeasible", self.util.model.id)
        else:
            # TODO create the least-squares method employed in MICOM
            pass
        commCurrent = self.util.model.slim_optimize()
        logger.debug("Community growth after regularization (%s)", commCurrent)
        return None

    def micom(self, media=None, tradeoff=0.5, *, abundances=None, pfba=False, min_growth=0.0):
        """MICOM's cooperative tradeoff (Diener, Gibbons & Resendis-Antonio 2020, mSystems
        5:e00606-19): member growth rates at a GIVEN composition.

        MICOM takes abundance as input and returns growth rates. It is a rate-allocation rule,
        not an abundance predictor: nothing here changes or predicts the composition. When no
        member constraint binds at the stage-2 point, its rates are

            mu_m = f * F * a_m / sum_k a_k^2

        (f = tradeoff, F = the stage-1 optimum), so mu_1 / mu_2 = a_1 / a_2 exactly and f only
        rescales every rate together: the declared composition times one scalar. A member rate
        departs from this only where a bound, a cap, a kinetic row or the medium binds
        (result.l2_deviation measures it). On the CommPhitting maltose community at the measured
        composition this gives E. coli 0.3155 f and P. fluorescens 0.0470 f 1/h.

        The formulation (mscommunity.commmicom, which also lists what is not reproduced):
            taxon units  every member reaction's bounds are multiplied by its abundance a_m, so
                         community fluxes are a_m times the per-gDW-member (taxon-unit) flux;
            biomass      bio1 is shut and every member biomass drain opened: member biomass is
                         free and community growth = sum_m v_bio_m = sum_m a_m mu_m;
            stage 1      F = max sum_m v_bio_m;
            stage 2      min sum_m mu_m^2 = sum_m (v_bio_m / a_m)^2
                         subject to tradeoff * F <= sum_m v_bio_m <= F;
            crossover    MICOM's LP, only when the stage-2 QP is not optimal;
            pfba=True    min total flux with each mu_m held at its stage-2 value.
        The *_commKin rows (which MICOM does not have) stay as configured; for MICOM proper run
        it inside `with commmembers.lifted_kinetic_rows(comm):`.

        media       None (the current medium), one medium (dict / KBase media) or a list of them
        tradeoff    f in [0, 1]; MICOM recommends 0.5 and its own default is 1.0. At f = 1 the
                    answer is the L2-minimal point of the stage-1 optimal face, which on the
                    CommPhitting builds is a unique vertex with no L2 content (maltose: E. coli
                    0, P. fluorescens 2.166 1/h)
        abundances  {member id: a} for every member (normalised); default the community's
                    current abundances. Written into bio1 and the commmembers caps for the call
                    and put back afterwards
        min_growth  minimum specific growth rate per member (1/h; a number or {member: rate}),
                    v_bio_m >= a_m * min_growth in both stages, as MICOM's min_growth

        Returns a list with one commmicom.MicomResult per medium: community_growth (= sum_m
        a_m mu_m at the stage-2 point), community_growth_max (F), growth_rates {member: mu_m},
        objective (sum_m mu_m^2), status, and the cobra solution (its .fluxes is also
        result.fluxes); self.solution is set to the last solution. Every reaction bound, member
        drain, the composition, the objective and the solver are restored on exit; the medium
        too. RuntimeError when no QP-capable solver is installed.

        The pre-2026-11 behaviour (the declared composition with mu_m = tradeoff * mu_max) is
        micom_legacy."""
        from mscommunity.commmicom import cooperative_tradeoff, qp_backend
        # The second stage minimises a quadratic, so a QP-capable backend is required: keep the
        # incumbent when it can already do QP, otherwise swap to the best available backend and
        # always swap back (qp_backend; hard-coding "hybrid" blew up on a Gurobi incumbent).
        with qp_backend(self):
            results = cooperative_tradeoff(self, media, tradeoff, abundances=abundances, pfba=pfba,
                                           min_growth=min_growth)
        for result in results:
            if result.solution is not None:
                self._set_solution(result.solution)
        return results

    def micom_legacy(self, media, tradeoff=0.6):
        """DEPRECATED: the micom() of MSCommunity before 2026-11, kept callable to reproduce old
        results. It is not MICOM: it maximises bio1 at the declared composition, holds bio1 at
        tradeoff x that maximum and minimises sum_m v_bio_m^2, so it returns the declared
        composition with every member at mu_m = tradeoff * mu_max (mu_max the bio1 maximum).
        Returns a list of cobra Solutions, one per medium. The comm_tradeoff row it installs is
        now removed on exit (it used to stay on bio1). Use micom() for MICOM's tradeoff."""
        import warnings
        warnings.warn("MSCommunity.micom_legacy is deprecated: it returns the declared composition "
                      "with every member at mu_m = tradeoff * mu_max (the bio1 maximum), which is "
                      "neither MICOM nor a prediction. Use micom() for MICOM's cooperative tradeoff.",
                      DeprecationWarning, stacklevel=2)
        from mscommunity.commmicom import qp_backend
        media = [media] if type(media) == dict else media
        with qp_backend(self):
            try:
                return self._micom_inner(media, tradeoff)
            finally:
                self.util.remove_constraint("comm_tradeoff")

    def _micom_inner(self, media, tradeoff):
        """micom_legacy's solves (unchanged)."""
        solutions = []
        for m in media:
            # add the objectives
            if self.abundances is None:
                self.predict_abundances()
            # obj = sum([mem.primary_biomass.forward_variable*mem.abundance for mem in self.members])
            obj = self.primary_biomass.forward_variable
            self.util.model.objective = self.util.model.problem.Objective(obj, direction="max")
            ## maximize the community growth
            sol = self.run_fba(m, pfba=False)
            ### set tradeoff
            biomass = sol.fluxes[self.primary_biomass.id]
            # biomass = sum([sol.fluxes[mem.primary_biomass.id]*mem.abundance for mem in self.members])
            tradeoff_growth = biomass*tradeoff
            logger.debug("micom: community growth %s, tradeoff %s", biomass, tradeoff)
            self.util.remove_constraint("comm_tradeoff")
            self.util.create_constraint(self.util.model.problem.Constraint(
                self.primary_biomass.forward_variable, name="comm_tradeoff",
                ub=None, lb=tradeoff_growth), printing=True)
            ## minimize the sum of the absolute differences between the member growths
            obj = sum([mem.primary_biomass.forward_variable**2 for mem in self.members])
            self.util.model.objective = self.util.model.problem.Objective(obj, direction="min")
            sol = self.run_fba(m, pfba=False)
            # TODO hard-code a subsequent pFBA optimization that fixes biomasses of all members 
            solutions.append(sol)
        return solutions


    # TODO evaluate the comparison of this method with MICOM
    # TODO implement a community tradeoff and then the objective should be a fraction of the total summed biomass with a minimization of variance
    def predict_abundances(self, media=None, pfba=True, timeout=60,
                           environName=None, regularization=True, update_abundances=False,
                           determinize=False, qp_backend=None):
        """Member composition read off one community solve: maximise the sum of the member
        biomass fluxes (member drains as configured; open by default) and return
        {member: v_bio_m / sum_k v_bio_k}, or None when the community does not grow.

        What the answer is depends on the options, and in neither case is it a prediction of
        the composition from metabolism alone:

        regularization=True (default)
            Returns a composition NEAR THE DECLARED ABUNDANCE: a prior, not a prediction. The
            min_comm_growth floor sits on bio1, whose coefficients are the declared abundances,
            and the per-member floors are the strictest feasible rung of a fixed ratio ladder
            (see regularization). Measured on the CommPhitting builds: acetate 0.1106 for a
            declared 0.1176, maltose 0.8187 for 0.870, maltose + 4HB 0.5400 for 0.574.
        regularization=False
            With the community-kinetics (kappa) rows off, the answer is a vertex of a nearly
            flat face, chosen by a ~0.2 % yield difference (SynCom: 0.0, R12 alone). With the
            kappa rows on, it is a kappa-set composition: kappa decides it (SynCom 0.657 at
            kappa 1363, 0.454 at kappa 2000), so a kappa calibrated at a declared composition
            returns that composition.

        media         applied first (and the previous medium restored on exit)
        timeout       solver time limit in s, set before the FIRST solve of the call and restored
                      on exit (an unbounded first slim_optimize stalled 3 of 7 calls for 4-7 min in
                      one long-lived process); a solve that hits it is treated as not optimal
        update_abundances  write the result into the community (set_abundance)
        determinize   replace a degenerate LP split by the unique QP centroid (_determinize_split)

        When the kappa rows (*_commKin) forbid growth on the medium but the community grows
        without them, they are dropped for this call only: a WARNING is logged and
        self.kinetic_rows_dropped is set True (it is reset to False at the start of every call),
        and the composition returned then comes from the model WITHOUT kappa rows. Objective,
        medium, timeout and the regularization rows are restored on exit, also when a solve
        raises (regularization raises FeasibilityError when its growth floor cannot be set)."""
        logger.debug("predict_abundances on %s: regularization=%s", self.util.model.id, regularization)
        self.kinetic_rows_dropped = False
        # store the original parameters
        ogObj = self.util.model.objective
        ogMedia = self.util.model.medium
        ogTimeout = self.util.model.solver.configuration.timeout
        # the time limit covers every solve of the call, this first slim_optimize included
        self.util.model.solver.configuration.timeout = timeout
        try:
            # apply environment media BEFORE regularization so constraints are calibrated correctly
            if media is not None:
                self.util.add_medium(media)
            slimOpt = self.util.model.slim_optimize()
            # PATCH 3: if kinetic constraints zero out growth on this medium, fall
            # back to a no-kinetics simulation rather than returning None. This
            # rescues low-yield carbons (e.g. Acetate) where Σ|flux| ≤ kinCoef·bio
            # is unsatisfiable at any positive biomass for kinCoef=750. Use the
            # cobra model's context manager so removed constraints auto-restore
            # on exit (so the package's internal tracking stays consistent).
            if isclose(0, slimOpt, abs_tol=1e-3):
                kin_cons = [c for c in self.util.model.constraints if "_commKin" in c.name]
                if kin_cons:
                    with self.util.model:
                        self.util.model.remove_cons_vars(kin_cons)
                        slimOpt_no_kin = self.util.model.slim_optimize()
                        if not isclose(0, slimOpt_no_kin, abs_tol=1e-3):
                            self.kinetic_rows_dropped = True
                            logger.warning(
                                "predict_abundances: the kinetic rows (%d *_commKin) forbid growth of %s on %s "
                                "(slim_optimize %.3g); they are dropped for this call (%.3g without them), so "
                                "the composition returned comes from the model WITHOUT kappa rows",
                                len(kin_cons), self.util.model.id, environName, slimOpt, slimOpt_no_kin)
                            return self._predict_inner(pfba, timeout, regularization, update_abundances,
                                                        ogObj, ogMedia, ogTimeout, determinize, qp_backend)
                    # context exits here, kinetic constraints auto-restored
                    logger.warning("predict_abundances: %s doesn't grow even without kinetics on %s",
                                   self.util.model.id, environName)
                    self.util.model.objective = ogObj
                    self.util.model.medium = ogMedia
                    self.util.model.solver.configuration.timeout = ogTimeout
                    return None
                else:
                    logger.warning("predict_abundances: %s doesn't grow, with a slim_optimize of %s in %s media",
                                   self.util.model.id, slimOpt, environName)
            return self._predict_inner(pfba, timeout, regularization, update_abundances,
                                        ogObj, ogMedia, ogTimeout, determinize, qp_backend)
        except BaseException:
            # leave the caller's model as it was found, not half-regularized
            self.util.remove_constraint("_regularization")
            self.util.remove_constraint("min_comm_growth")
            self.util.model.objective = ogObj
            self.util.model.medium = ogMedia
            raise
        finally:
            self.util.model.solver.configuration.timeout = ogTimeout

    def _predict_inner(self, pfba, timeout, regularization, update_abundances,
                       ogObj, ogMedia, ogTimeout, determinize=False, qp_backend=None):
        # maximize the sum of all member biomass reactions
        self.set_objective(targets=[species.primary_biomass.forward_variable for species in self.members])
        self.util.model.solver.configuration.timeout = timeout
        if regularization:   threshold = self.regularization(linear=True)
        else:   self.util.remove_constraint("_regularization")
        # BUG FIX (Layer 0 #1): the old `except: ... pfba=False` silently reverted
        # to the plain, solver-degenerate LP with no trace — masking exactly the
        # hard pairs (and it would swallow a future QP-backend failure too). Catch
        # only real solve errors, log the cause, and record the fallback so callers
        # can flag the row as NOT determinism-converged.
        self.pfba_fell_back = False
        try:
            sol = self.run_fba(None, pfba)
        except Exception as e:
            self._note_fallback(
                "pFBA failed for %s (%s: %s); falling back to plain LP — this "
                "row's per-member split is NOT determinism-converged",
                self.util.model.id, type(e).__name__, e)
            sol = self.run_fba(None, pfba=False)
        # Determinize the per-member split to the unique strictly-convex-QP optimum
        # while the regularization floors are still installed. No-op on the ~98.8%
        # of cells whose split is already unique; on the flat-face ~1.2% it replaces
        # the arbitrary LP vertex with the solver-independent centroid.
        if determinize:
            sol = self._determinize_split(sol, qp_backend)
        self.util.remove_constraint("_regularization")
        self.util.remove_constraint("min_comm_growth")
        self.util.model.solver.configuration.timeout = ogTimeout
        self.util.model.objective = ogObj
        self.util.model.medium = ogMedia
        return self._compute_relative_abundance_from_solution(sol, True, update_abundances)

    def _note_fallback(self, message, *args):
        """Record that THIS row's per-member split is not determinism-converged.

        `pfba_fell_back` is the per-row flag (reset at the top of `_predict_inner`)
        and `pfba_fallback_count` counts ROWS, not events: a row that both loses pFBA
        and fails to determinize is counted once. Previously only `_predict_inner`'s
        pFBA handler touched the counter, so the determinizer's own failure paths were
        invisible and the count under-reported the true non-converged rate."""
        if not self.pfba_fell_back:
            self.pfba_fell_back = True
            self.pfba_fallback_count += 1
        logger.warning(message, *args)

    def _determinize_split(self, sol, qp_backend=None, tol=1e-6, fix_fluxes=True):
        """Refine the per-member coculture-growth split to the UNIQUE strictly-convex
        QP optimum so it is solver-independent. Called from _predict_inner while the
        regularization floors + sum-of-member objective are still installed.

        On the deterministic-construction base ~98.8% of (pair, medium) cells already
        have a unique max-sum-member split; ~1.2% sit on a flat alternate-optima face
        where GLPK and Gurobi pick different vertices (Δ up to ~3.4). A cheap 2-LP
        probe on member[0]'s biomass detects the flat face; only then do we minimise
        sum(mu_i^2) on a QP backend (Gurobi/CPLEX exact, else open-source OSQP/hybrid)
        to select the unique centroid, validate it is feasible on the exact native
        solver, then pin mu* and pFBA on the native solver for a deterministic full
        flux vector. Returns the (possibly new) solution; sets pfba_fell_back if a QP
        backend is needed but unavailable or returns an infeasible point."""
        members = list(self.members)
        if len(members) < 2:
            return sol
        model = self.util.model
        native = model.solver.interface.__name__.rsplit(".", 1)[-1].replace("_interface", "")
        sum_vars = [m.primary_biomass.forward_variable for m in members]
        try:
            C = float(sum(sol.fluxes[m.primary_biomass.id] for m in members))
        except Exception:
            return sol
        if isnan(C) or C <= 1e-9:
            return sol
        # Pin the community total near its max, with a tiny slack BELOW it. Some
        # degenerate cells are a NEAR-flat ridge rather than an exactly-flat face:
        # GLPK and Gurobi stop at points whose totals differ within their optimality
        # tolerance (~1e-5). An exact Σ=C pin collapses that ridge (the probe then
        # reads the split as "unique"); the slack lets the QP traverse the ridge to
        # the centroid. Growth is reduced by at most det_slack (relative ~1e-6).
        det_slack = max(1e-6 * abs(C), 1e-8)
        self.util.remove_constraint("_det_total")
        self.util.create_constraint(model.problem.Constraint(sum(sum_vars), name="_det_total", lb=C - det_slack, ub=None))
        og_obj = model.objective
        try:
            # --- degeneracy probe: range of mu_0 on the fixed-total optimal face ---
            v0 = members[0].primary_biomass.forward_variable
            model.objective = model.problem.Objective(v0, direction="min"); lo = model.slim_optimize()
            model.objective = model.problem.Objective(v0, direction="max"); hi = model.slim_optimize()
            model.objective = og_obj
            if lo is None or hi is None or isnan(lo) or isnan(hi) or (hi - lo) <= tol:
                return sol  # split already unique -> no-op (the common ~98.8% case)

            # --- strictly-convex QP for the unique centroid split ---
            # Solved by HiGHS native QP via matrix extraction (see _solve_qp_highs):
            # it does NOT swap the live model's solver, so it is safe inside the
            # PATCH-3 `with self.util.model:` context (kinetics-disabled cells) and
            # needs no QP-capable optlang backend (GLPK is LP-only). Exact + open-source.
            mu = self._solve_qp_highs(sum_vars)
            if mu is None:
                self._note_fallback("QP determinizer (HiGHS) unavailable or non-optimal "
                                    "for %s; leaving the solver-dependent LP split.",
                                    self.util.model.id)
                return sol

            # --- pin mu* on the native solver (validate feasibility) ---
            def _pin(ptol):
                for m in members:
                    self.util.remove_constraint(f"_det_fix_{m.id}")
                    self.util.create_constraint(model.problem.Constraint(
                        m.primary_biomass.forward_variable, name=f"_det_fix_{m.id}",
                        lb=mu[m.id] - ptol, ub=mu[m.id] + ptol))
                model.slim_optimize()
                return model.solver.status == "optimal"
            if not _pin(1e-6) and not _pin(1e-4):
                self._note_fallback("QP split infeasible on native solver for %s; "
                                    "keeping LP split.", self.util.model.id)
                return sol

            # --- pFBA on native with mu* pinned -> deterministic full flux vector ---
            if fix_fluxes:
                try: final = self.util.run_fba(None, True)
                except Exception: final = model.optimize()
            else:
                final = model.optimize()
            return final
        finally:
            for m in members:
                self.util.remove_constraint(f"_det_fix_{m.id}")
            self.util.remove_constraint("_det_total")
            try: model.objective = og_obj
            except Exception: self.set_objective(targets=sum_vars)

    def _solve_qp_highs(self, hess_vars):
        """min sum(v**2 for v in hess_vars) subject to ALL current model constraints
        and variable bounds, via HiGHS native QP (highspy) by extracting the problem
        matrices — WITHOUT swapping the live model's solver. This is what makes the
        determinizer safe inside the PATCH-3 `with model:` context and independent of
        whether the cobra solver supports quadratics (GLPK does not). Exact and
        strictly open-source. Returns {member_id: mu} or None if highspy is
        unavailable or the QP is not optimal."""
        try:
            import highspy
            import numpy as np
        except Exception:
            return None
        model = self.util.model
        INF = highspy.kHighsInf
        variables = list(model.solver.variables)
        vidx = {v.name: i for i, v in enumerate(variables)}
        n = len(variables)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("log_to_console", False)
        # Hard wall-clock cap so a pathological/degenerate community QP can never
        # hang the whole scoring run (some KBase-reconstructed communities sent
        # HiGHS into a non-terminating solve). On time-out the status is not
        # kOptimal, so we return None below and fall back to the LP split
        # (flagged via pfba_fell_back) instead of blocking forever.
        h.setOptionValue("time_limit", float(environ.get("QP_TIME_LIMIT", "20")))
        for v in variables:
            h.addVar(-INF if v.lb is None else float(v.lb),
                      INF if v.ub is None else float(v.ub))
        for con in model.solver.constraints:
            items = list(con.get_linear_coefficients(list(con.variables)).items())
            if not items:
                continue
            idx = np.array([vidx[v.name] for v, _ in items], dtype=np.int32)
            val = np.array([float(cf) for _, cf in items], dtype=np.float64)
            lo = -INF if con.lb is None else float(con.lb)
            up =  INF if con.ub is None else float(con.ub)
            h.addRow(lo, up, len(idx), idx, val)
        # objective = sum v^2 = 0.5 x^T Q x with Q_ii = 2 on the member biomass vars
        diag = {vidx[v.name] for v in hess_vars}
        q_start = np.zeros(n + 1, dtype=np.int32)
        q_idx, q_val = [], []
        for col in range(n):
            q_start[col] = len(q_idx)
            if col in diag:
                q_idx.append(col); q_val.append(2.0)
        q_start[n] = len(q_idx)
        try:
            h.passHessian(n, len(q_idx), highspy.HessianFormat.kTriangular,
                          q_start, np.array(q_idx, dtype=np.int32), np.array(q_val, dtype=np.float64))
            h.run()
        except Exception as e:
            logger.warning("HiGHS QP solve failed for %s: %s", self.util.model.id, e)
            return None
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
        cv = h.getSolution().col_value
        return {m.id: float(cv[vidx[m.primary_biomass.forward_variable.name]]) for m in self.members}

    def run_fba(self, media=None, pfba=False, fva_reactions=None):
        # print("pfba =", pfba)
        if media is not None:
            self.util.add_medium(media)
        return self._set_solution(self.util.run_fba(None, pfba, fva_reactions))

    def _comm_growth(self):
        self.comm_growth = 0
        self.member_fluxes, self.memGrowths = {}, {}
        for mem in self.members:
            self.member_fluxes[mem.id] = array([self.solution.fluxes[rxn.id] for rxn in mem.reactions])
            self.memGrowths[mem.id] = self.solution.fluxes[mem.primary_biomass.id]
            self.comm_growth += self.memGrowths[mem.id] * mem.abundance

    def _compute_relative_abundance_from_solution(self, solution=None, skipNoGrowth=True, update_abundances=False):
        if solution is not None:  self._set_solution(solution)
        total_growth = sum([self.solution.fluxes[member.primary_biomass.id] for member in self.members])
        message = f"The total community growth is {total_growth}"
        if self.printing:  print(message)
        if isclose(0, total_growth, abs_tol=1e-3):
            if not skipNoGrowth:   NoFluxError(f"No community growth: {total_growth} in {self.util.model.id}")
            else:    print(message)  ;  return None
        abundances = {member.id: self.solution.fluxes[member.primary_biomass.id]/total_growth for member in self.members}
        if update_abundances:
            self.set_abundance(abundances)
            if self.printing:  print(f"Updated abundances: {self.abundances}")
        for mem in self.members:
            logger.debug("%s grows %s with abundance %s", mem.id, self.solution.fluxes[mem.primary_biomass.id], mem.abundance)
        return abundances

    def _set_solution(self, solution, dump_diagnostics=None):
        if solution.status != "optimal":
            # This used to be silent AND fatal at once: the FeasibilityError was
            # constructed but never raised (so a sub-optimal solution was consumed as
            # if optimal), while the unconditional diagnostic dump crashed on the bare
            # relative "erronous_model.lp" (print_lp -> makedirs("")). Be loud instead,
            # and only dump when the caller opted in through `lp_filename` (or asked
            # explicitly). Logging rather than raising keeps predict_abundances'
            # documented tolerance of non-growing media intact -- callers detect the
            # bad row through the NaN/zero growth that _compute_relative_abundance_
            # from_solution already screens for.
            self.suboptimal_solution = True
            logger.error("The %s solution is sub-optimal, with a(n) %s status; the fluxes and "
                         "abundances derived from it are NOT trustworthy.",
                         self.util.model.id, solution.status)
            if dump_diagnostics or (dump_diagnostics is None and self.lp_filename):
                dumpDir = path.dirname(self.lp_filename) if self.lp_filename else ""
                try:
                    self.print_lp(path.join(dumpDir, "erronous_model.lp"))
                    save_matlab_model(self.util.model, path.join(
                        dumpDir, f"{self.util.model.name or self.util.model.id}.mat"))
                except Exception as e:
                    logger.warning("Could not dump diagnostics for %s: %s", self.util.model.id, e)
        else:  self.suboptimal_solution = False
        self.solution = solution
        self.exchange_fluxes = {ex.id: self.solution.fluxes[ex.id] for ex in self.util.model.reactions if "EX_" in ex.id}
        self._comm_growth()
        if self.printing:
            print("Member Biomass Fluxes:", self.memGrowths)  # TODO weight the member growths by their abundances
            print("Max member Fluxes:", {memID: growth*self.kinCoef for memID, growth in self.memGrowths.items()})
            print("Total fluxes:", {memID: sum(abs(fluxes)) for memID, fluxes in self.member_fluxes.items()})
        # logger.info(self.util.model.summary())
        return self.solution

    def return_member_models(self):
        ## applicability in disaggregating Filipe's Nitrate reducing community model for the SBI ENIGMA team.
        return [member.model for member in self.members]

    def add_medium(self, media):
        self.util.add_medium(media)

    # --- Batched-LP entry points -----------------------------------------
    # The community S matrix is fixed across samples / conditions; only
    # bounds and objective vary. `extract_problem` snapshots that shared
    # structure once and `solve_batch` routes a list of `LPInstance`s
    # through a pluggable backend (cpu reference today, gpu later) without
    # any call-site changes.

    def extract_problem(self):
        """Snapshot the live community model as a `CommunityProblem`."""
        return CommunityProblem.from_model(self.util.model)

    def solve_batch(self, instances, backend="cpu", problem=None, **backend_kwargs):
        """Solve N independent LPs that share this community's S matrix."""
        if problem is None:
            problem = self.extract_problem()
        solver = get_batched_solver(backend, **backend_kwargs)
        return solver.solve(problem, instances)

    def _solo_max_batch(self, backend="cpu", workers=1):
        """Per-member solo growth max, batched over members.

        Each member's LP zeros out the other members' biomass reactions and
        maximizes its own — same S, different bounds and objective.
        """
        instances = []
        for target in self.members:
            bounds = {}
            for other in self.members:
                if other.id != target.id:
                    bounds[other.primary_biomass.id] = (0.0, 0.0)
            # "Solo" means this member growing without its partners, so its biomass has
            # to be able to leave the system on its own: zeroing the other members'
            # biomass also zeroes the community biomass reaction, which is the only
            # other outlet for this member's biomass compound. Restore its drain for
            # this LP even when the coupled community model runs with drains closed --
            # otherwise every solo capacity is 0 and the regularization floors vanish.
            if target.biomass_drain is not None:
                bounds[target.biomass_drain.id] = tuple(
                    getattr(target, "biomass_drain_bounds", (0.0, 1000.0)))
            instances.append(LPInstance(
                id=target.id,
                bounds=bounds,
                objective={target.primary_biomass.id: 1.0},
                sense="max",
            ))
        results = self.solve_batch(instances, backend=backend, workers=workers)
        solo = {}
        for r in results:
            # The status is the ONLY trustworthy signal here: on a non-optimal solve
            # GLPK leaves the objective expression evaluated at the STALE incumbent of
            # whatever was solved before (a positive number), while Gurobi returns
            # None -- so reading objective_value alone made the recorded solo capacity
            # solver-dependent garbage. Anything not optimal means "no capacity".
            if r.status != "optimal":
                logger.warning("The solo-capacity LP of %s in %s is %s; recording zero capacity "
                               "(its objective_value %r is not meaningful).",
                               r.id, self.util.model.id, r.status, r.objective_value)
                solo[r.id] = 0
                continue
            v = r.objective_value
            solo[r.id] = v if (v is not None and not isnan(v) and v > 1e-6) else 0
        return solo

    def predict_abundances_batch(self, medias, pfba=False, backend="cpu", workers=1):
        """Predict member abundances across many media in one batched solve.

        Each medium becomes an `LPInstance` whose bounds patch the exchange
        reactions of the shared community model. The objective is the sum
        of member primary biomass fluxes (matches `_predict_inner`).
        """
        problem = self.extract_problem()
        obj_patch = {mem.primary_biomass.id: 1.0 for mem in self.members}
        instances = []
        for i, media in enumerate(medias):
            bounds = media_to_bounds(self.util.model, media)
            instances.append(LPInstance(
                id=f"sample_{i}",
                bounds=bounds,
                objective=obj_patch,
                sense="max",
                pfba=pfba,
            ))
        results = self.solve_batch(instances, backend=backend, problem=problem, workers=workers)
        out = []
        for r in results:
            if r.status != "optimal":
                out.append(None)
                continue
            total = sum(r.fluxes.get(mem.primary_biomass.id, 0.0) for mem in self.members)
            if isclose(0, total, abs_tol=1e-3):
                out.append(None)
                continue
            out.append({
                mem.id: r.fluxes.get(mem.primary_biomass.id, 0.0) / total
                for mem in self.members
            })
        return out

