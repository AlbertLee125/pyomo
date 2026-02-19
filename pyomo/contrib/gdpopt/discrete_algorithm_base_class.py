#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from collections import namedtuple
import itertools as it
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt

from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, get_main_elapsed_time
from pyomo.core import minimize, TransformationFactory, Objective, value
from pyomo.core.base import ComponentUID
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr.logical_expr import ExactlyExpression
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    add_algebraic_variable_list,
    add_boolean_variable_lists,
    add_disjunct_list,
    add_transformed_boolean_variable_list,
)

from pyomo.contrib.gdpopt.discrete_search_enums import DirectionNorm, SearchPhase




class DiscreteDataManager:
    """Manage explored points in a discrete search space.

    The manager stores per-point metadata (feasibility, objective, provenance)
    using only built-in Python containers.

    Attributes
    ----------
    point_info : dict[tuple[int, ...], dict[str, object]]
        Mapping from an external-variable point to a metadata dictionary.
    external_var_info_list : list
        External variable descriptors (e.g., bounds) used for point validation.
    solution_cache : dict[tuple[int, ...], dict[str, dict[str, object]]]
        Optional cached variable values for previously evaluated points.
        The payload is a dict (e.g., ``{"algebraic": {...}, "boolean": {...}}``)
        whose inner keys are stable component identifiers (``ComponentUID``
        string representations).
    """

    def __init__(self, external_var_info_list=None):
        """Create a new data manager.

        Parameters
        ----------
        external_var_info_list : list, optional
            External variable descriptors (e.g., a list of ``ExternalVarInfo``)
            used to validate candidate points.
        """
        self.point_info: dict[tuple[int, ...], dict[str, object]] = {}
        self.external_var_info_list = external_var_info_list
        # Optional per-point solution cache. Keys are points in the external
        # variable space; values are dicts mapping stable component UIDs
        # (strings) to cached values. This is used by discrete algorithms to
        # avoid re-solving previously evaluated points when updating the
        # incumbent solution.
        self.solution_cache: dict[tuple[int, ...], dict[str, dict[str, object]]] = {}

    def store_solution(self, point: tuple[int, ...], solution):
        """Store a cached solution payload for an external-variable point.

        Parameters
        ----------
        point : tuple[int, ...]
            External-variable point.
        solution : dict
            Solution payload.

            Expected format is a dict containing (some or all of) the keys:

            - ``"algebraic"``: mapping from ``str(ComponentUID(var))`` to a
              numeric value
            - ``"boolean"``: mapping from ``str(ComponentUID(var))`` to a bool
              (or a numeric 0/1)

        Notes
        -----
        This cache is intentionally independent from Pyomo's ``SolverResults``
        solution objects. GDPopt meta-solvers frequently return empty
        ``results.solution`` containers, so caching provides a lightweight
        way to recover a best-known assignment for transfer to the original
        model.
        """
        self.solution_cache[tuple(point)] = solution

    def get_solution(self, point: tuple[int, ...]):
        """Return cached solution payload for a point, if present.

        Parameters
        ----------
        point : tuple[int, ...]
            External-variable point.

        Returns
        -------
        dict or None
            The cached payload for ``point``, or ``None`` if no cache entry
            exists.
        """
        return self.solution_cache.get(tuple(point))

    def set_external_info(self, external_var_info_list):
        """Set bounds/structure information for external variables.

        Parameters
        ----------
        external_var_info_list : list
            External variable descriptors (e.g., a list of ``ExternalVarInfo``).
        """
        self.external_var_info_list = external_var_info_list

    def add(
        self,
        point: tuple[int, ...],
        feasible: bool,
        objective: float,
        source: str,
        iteration_found: int,
    ):
        """Register a visited point and its metadata.

        Parameters
        ----------
        point : tuple[int, ...]
            External-variable point.
        feasible : bool
            Whether the point is feasible.
        objective : float
            Objective value (or penalty value for infeasible points).
        source : str
            Provenance label (e.g., "Neighbor", "Anchor").
        iteration_found : int
            Iteration counter at which the point was first evaluated.
        """
        self.point_info[point] = {
            "feasible": feasible,
            "objective": objective,
            "source": source,
            "iteration_found": iteration_found,
        }

    def is_visited(self, point: tuple[int, ...]) -> bool:
        """Check whether a point has already been evaluated.

        Parameters
        ----------
        point : tuple[int, ...]
            External-variable point.

        Returns
        -------
        bool
            ``True`` if the point is present in the registry.
        """
        return point in self.point_info

    def get_info(self, point: tuple[int, ...]) -> dict[str, object] | None:
        """Get stored metadata for a point.

        Parameters
        ----------
        point : tuple[int, ...]
            External-variable point.

        Returns
        -------
        dict[str, object] or None
            Metadata dictionary if present; otherwise ``None``.
        """
        return self.point_info.get(point)

    def get_cached_value(self, point: tuple[int, ...]) -> float | None:
        """Get the cached objective value for a visited point.

        Parameters
        ----------
        point : tuple[int, ...]
            External-variable point.

        Returns
        -------
        float or None
            Cached objective value, or ``None`` if the point is unknown.
        """
        info = self.point_info.get(point)
        if info:
            return info["objective"]
        return None

    def is_valid_point(self, point: tuple[int, ...]) -> bool:
        """Check whether a point lies within configured bounds.

        Parameters
        ----------
        point : tuple[int, ...]
            External-variable point.

        Returns
        -------
        bool
            ``True`` if the point is within all bounds, or if no bound
            information is available.
        """
        if not self.external_var_info_list:
            return True

        return all(
            info.LB <= val <= info.UB
            for val, info in zip(point, self.external_var_info_list)
        )

    def get_best_solution(
        self, sense=None
    ) -> tuple[tuple[int, ...] | None, float | None]:
        """Return the best feasible point found so far.

        Parameters
        ----------
        sense : {minimize, maximize}, optional
            Objective sense used to determine "best". Defaults to minimize.

        Returns
        -------
        (tuple[int, ...] or None, float or None)
            ``(best_point, best_objective)`` if any feasible point exists;
            otherwise ``(None, None)``.
        """
        from pyomo.core import minimize as _minimize

        if sense is None:
            sense = _minimize

        feasible_candidates = {
            pt: data["objective"]
            for pt, data in self.point_info.items()
            if data["feasible"]
        }

        if not feasible_candidates:
            return None, None

        if sense == _minimize:
            best_point = min(feasible_candidates, key=feasible_candidates.get)
        else:
            best_point = max(feasible_candidates, key=feasible_candidates.get)
        return best_point, feasible_candidates[best_point]


ExternalVarInfo = namedtuple(
    'ExternalVarInfo',
    [
        'exactly_number',  # number of external variables for this type
        'Boolean_vars',  # list with names of the ordered Boolean variables to be reformulated
        'UB',  # upper bound on external variable
        'LB',  # lower bound on external variable
    ],
)


class _GDPoptDiscreteAlgorithm(_GDPoptAlgorithm):
    """Base class for GDPopt discrete algorithms.

    Notes
    -----
    This base class centralizes the common discrete-search machinery and uses
    :class:`~pyomo.contrib.gdpopt.discrete_algorithm_base_class.DiscreteDataManager`
    to track evaluated points.

    Developer notes (relative to LDSDA)
            - Direction generation and subproblem-result handling are reused.
            - Neighbor validity checks and external-variable extraction were
                refactored to use the data manager.
            - Discrete-point evaluation and neighbor generation were factored into
                dedicated helper methods.
    """

    def __init__(self, **kwds):
        """Initialize the discrete algorithm base.

        Parameters
        ----------
        **kwds
            Forwarded to the GDPopt base algorithm constructor.
        """
        super().__init__(**kwds)
        self.data_manager = DiscreteDataManager()

    # def _cache_point_solution(self, point, util_block):
    #     """Cache variable values for a point.

    #     Parameters
    #     ----------
    #     point : iterable[int]
    #         External-variable point.
    #     util_block : Block
    #         Utility block on the model instance whose values should be cached.
    #         This is typically a cloned subproblem's util block.

    #     Returns
    #     -------
    #     None

    #     Notes
    #     -----
    #     Values are cached using stable identifiers (``ComponentUID`` string
    #     representations) so that values can later be replayed onto the
    #     *original* model's variable lists when updating the incumbent.

    #     The cache currently includes:

    #     - Algebraic variables from ``util_block.algebraic_variable_list``
    #     - Boolean variables from ``util_block.boolean_variable_list``
    #     """
    #     point = tuple(point)
    #     algebraic = {
    #         str(ComponentUID(v)): v.value for v in getattr(util_block, 'algebraic_variable_list', [])
    #     }
    #     boolean = {
    #         str(ComponentUID(v)): v.value for v in getattr(util_block, 'boolean_variable_list', [])
    #     }
    #     self.data_manager.store_solution(point, {'algebraic': algebraic, 'boolean': boolean})

    def _cache_point_solution(self, point, solved_model):
        """Cache variable values for a point.

        Parameters
        ----------
        point : iterable[int]
            External-variable point.
        solved_model : Block
            A solved model instance (typically the cloned subproblem) that
            contains primal variable values.

        Notes
        -----
        Cache keys are based on *original-model* variable ComponentUID strings.
        Values are retrieved from the corresponding components on `solved_model`
        using `ComponentUID(orig_var).find_component(solved_model)`.

        This avoids mismatches between clone/transformation-specific UIDs and
        the original model's UIDs when replaying cached solutions.
        """
        point = tuple(point)

        algebraic = {}
        for ov in getattr(self.original_util_block, "algebraic_variable_list", []):
            uid = str(ComponentUID(ov))
            sv = ComponentUID(ov).find_component(solved_model)
            algebraic[uid] = None if sv is None else sv.value

        boolean = {}
        for ob in getattr(self.original_util_block, "boolean_variable_list", []):
            uid = str(ComponentUID(ob))
            sb = ComponentUID(ob).find_component(solved_model)
            val = None
            if sb is not None:
                # Prefer associated binary value when available because logical
                # transformations may not populate BooleanVar.value consistently.
                try:
                    abin = sb.get_associated_binary()
                except Exception:
                    abin = None
                if abin is not None and abin.value is not None:
                    val = abin.value
                else:
                    val = sb.value
            boolean[uid] = val

        self.data_manager.store_solution(point, {"algebraic": algebraic, "boolean": boolean})

    def _load_incumbent_from_solution_cache(self, point, logger=None):
        """Load incumbent buffers from a cached solution payload.

        Parameters
        ----------
        point : iterable[int]
            External-variable point to load.
        logger : logging.Logger, optional
            Logger used for debug messages when no cached payload exists.

        Returns
        -------
        bool
            ``True`` if a cached payload was found and incumbent buffers
            (``incumbent_continuous_soln`` and ``incumbent_boolean_soln``)
            were updated; otherwise ``False``.

        Notes
        -----
        This method updates incumbent buffers *only*; it does not directly
        assign values onto the user's original model. The transfer to the
        original model occurs in the GDPopt base algorithm via
        ``_transfer_incumbent_to_original_model``.
        """
        # point = tuple(point)
        # payload = self.data_manager.get_solution(point)
        # if payload is None:
        #     if logger is not None:
        #         logger.debug("No cached solution available for point %s", point)
        #     return False

        # algebraic_map = payload.get('algebraic', {})
        # boolean_map = payload.get('boolean', {})

        # # Align with the original util-block list ordering expected by
        # # _transfer_incumbent_to_original_model.
        # self.incumbent_continuous_soln = [
        #     algebraic_map.get(str(ComponentUID(v)))
        #     for v in self.original_util_block.algebraic_variable_list
        # ]
        # self.incumbent_boolean_soln = [
        #     boolean_map.get(str(ComponentUID(v)))
        #     for v in self.original_util_block.boolean_variable_list
        # ]
        point = tuple(point)
        payload = self.data_manager.get_solution(point)
        if payload is None:
            if logger is not None:
                logger.debug("No cached solution available for point %s", point)
            return False

        algebraic_map = payload.get("algebraic", {})
        boolean_map = payload.get("boolean", {})

        new_cont = [
            algebraic_map.get(str(ComponentUID(v)))
            for v in self.original_util_block.algebraic_variable_list
        ]
        new_bool = [
            boolean_map.get(str(ComponentUID(v)))
            for v in self.original_util_block.boolean_variable_list
        ]

        # Defensive: do not clobber a valid incumbent with an unmapped cache entry
        if all(val is None for val in new_cont):
            if logger is not None:
                logger.debug(
                    "Cached payload for point %s did not map to original algebraic vars; skipping incumbent overwrite.",
                    point,
                )
            return False

        self.incumbent_continuous_soln = new_cont
        self.incumbent_boolean_soln = new_bool
        return True

    def _get_external_information(self, util_block, config):
        """Extract external-variable metadata from the working model.

        This inspects the configured logical constraints and/or disjunctions
        and creates a list of external-variable descriptors used to map GDP
        structure into a discrete (integer) search space.

        Parameters
        ----------
        util_block : Block
            GDPopt utility block attached to the working model.
        config : ConfigBlock
            GDPopt configuration block. Uses ``logical_constraint_list``,
            ``disjunction_list``, and ``starting_point``.

        Raises
        ------
        ValueError
            If a configured logical constraint is not an ``ExactlyExpression``.
        ValueError
            If an ``Exactly(N)`` constraint has ``N > 1``.
        ValueError
            If the length of ``config.starting_point`` does not match the
            number of derived external variables.
        """
        util_block.external_var_info_list = []

        reformulation_summary = []
        # Identify the variables that can be reformulated by performing a loop over logical constraints

        if config.logical_constraint_list is not None:
            for c in util_block.config_logical_constraint_list:
                if not isinstance(c.body, ExactlyExpression):
                    raise ValueError(
                        "The logical_constraint_list config should be a list of ExactlyExpression logical constraints."
                    )
                # TODO: in the first version, we don't support more than one exactly constraint.
                exactly_number = c.body.args[0]
                if exactly_number > 1:
                    raise ValueError("The function only works for exactly_number = 1")
                sorted_boolean_var_list = c.body.args[1:]
                util_block.external_var_info_list.append(
                    ExternalVarInfo(
                        exactly_number=1,
                        Boolean_vars=sorted_boolean_var_list,
                        UB=len(sorted_boolean_var_list),
                        LB=1,
                    )
                )
                reformulation_summary.append(
                    [
                        1,
                        len(sorted_boolean_var_list),
                        [boolean_var.name for boolean_var in sorted_boolean_var_list],
                    ]
                )

        if config.disjunction_list is not None:
            for disjunction in util_block.config_disjunction_list:
                sorted_boolean_var_list = [
                    disjunct.indicator_var for disjunct in disjunction.disjuncts
                ]
                util_block.external_var_info_list.append(
                    ExternalVarInfo(
                        exactly_number=1,
                        Boolean_vars=sorted_boolean_var_list,
                        UB=len(sorted_boolean_var_list),
                        LB=1,
                    )
                )
                reformulation_summary.append(
                    [
                        1,
                        len(sorted_boolean_var_list),
                        [boolean_var.name for boolean_var in sorted_boolean_var_list],
                    ]
                )

        # Set the external variable information in the data manager (after we
        # have collected all sources of external variables).
        self.data_manager.set_external_info(util_block.external_var_info_list)
        config.logger.info("Reformulation Summary:")
        config.logger.info("  Index | Ext Var | LB | UB | Associated Boolean Vars")
        self.number_of_external_variables = sum(
            external_var_info.exactly_number
            for external_var_info in util_block.external_var_info_list
        )
        if self.number_of_external_variables != len(config.starting_point):
            raise ValueError(
                "The length of the provided starting point doesn't equal the number of disjunctions."
            )

    def _fix_disjunctions_with_external_var(self, external_var_values_list):
        """Fix Boolean variables to match an external-variable point.

        Parameters
        ----------
        external_var_values_list : tuple[int, ...] or list[int]
            External-variable values. Each value is interpreted as 1-based
            index selecting the active Boolean among the associated list.

        Returns
        -------
        None
        """
        for external_variable_value, external_var_info in zip(
            external_var_values_list,
            self.working_model_util_block.external_var_info_list,
        ):
            for idx, boolean_var in enumerate(external_var_info.Boolean_vars):
                # external_variable_value is 1-based (usually)
                is_active = idx == external_variable_value - 1

                boolean_var.fix(is_active)
                if boolean_var.get_associated_binary() is not None:
                    boolean_var.get_associated_binary().fix(1 if is_active else 0)

    def _solve_discrete_point(self, point, search_type: SearchPhase | str, config):
        """Evaluate a single discrete point and register the result.

        This wrapper handles caching (skip if already visited), fixing the
        working model to the requested point, and registering feasibility and
        objective information in the data manager.

        Parameters
        ----------
        point : tuple[int, ...]
            External-variable point to evaluate.
        search_type : SearchPhase | str
            Label describing why the point is being evaluated (e.g.,
            ``SearchPhase.ANCHOR`` or ``"Anchor"``).
        config : ConfigBlock
            GDPopt configuration block.
            Contains ``infinity_output`` used as a penalty for infeasible points.

        Returns
        -------
        (bool, float)
            ``(primal_improved, objective)``.

            - ``primal_improved`` indicates whether the solve improved the
              solver's incumbent bound.
            - ``objective`` is the objective value.
            - For infeasible points, ``objective`` returns a penalty value.
            - This penalty is defined by ``config.infinity_output``.

        Notes
        -----
        The ``config.infinity_output`` serves two primary purposes:
        1. It provides a finite numerical penalty for infeasible discrete points.
        2. It acts as a feasibility threshold for solver outputs.
        If the subproblem is infeasible, the objective is set to this value.
        If the solver returns a value $\\ge$ ``infinity_output``, the point is marked infeasible.
        Using a large finite value instead of $inf$ prevents numerical errors in the master problem.
        """
        # 1. Check if already visited (optional, depending on algorithm logic)
        # Some algos might re-evaluate, but usually we skip.
        if self.data_manager.is_visited(point):
            cached_obj = self.data_manager.get_cached_value(point)
            if config.logger is not None and hasattr(self, 'log_formatter'):
                try:
                    self._log_current_state(
                        config.logger,
                        f"Skipped {search_type}",
                        point,
                        primal_improved=False,
                    )
                except Exception:
                    # Logging is best-effort only; failures here must not
                    # interrupt the optimization algorithm.
                    pass
            return False, cached_obj

        # 3. Solve the subproblem (Relies on implementation in child class)
        primal_improved, primal_bound = self._solve_GDP_subproblem(
            point, search_type, config
        )

        # 4. Normalize result and register the visit
        # NOTE: Not all discrete algorithms define an explicit infeasibility
        # penalty. When available, infinity_output is used as a finite penalty.
        if primal_bound is None:
            feasible = False
            objective = (
                config.infinity_output
                if hasattr(config, 'infinity_output')
                else float('inf')
            )
        else:
            objective = primal_bound
            if hasattr(config, 'infinity_output'):
                # You should make sure infinity_output is large enough
                feasible = primal_bound < config.infinity_output
            else:
                feasible = True

        self.data_manager.add(
            point,
            feasible=feasible,
            objective=objective,
            source=str(search_type),
            iteration_found=int(getattr(self, 'iteration', 0)),
        )

        return primal_improved, objective

    def _solve_GDP_subproblem(
        self, external_var_value, search_type: SearchPhase | str, config
    ):
        """Solve the GDP subproblem with disjunctions fixed by external variables.

        This is the discrete-point evaluation hook used by the discrete base
        class via `_solve_discrete_point`.

        Returns
        -------
        (primal_improved, primal_bound)
            primal_bound is a float objective value when solvable, or None when
            the subproblem is infeasible or fails.
        """
        # Fix working model Booleans (and associated binaries) to match the
        # proposed external point
        self._fix_disjunctions_with_external_var(external_var_value)

        subproblem = self.working_model.clone()
        TransformationFactory('core.logical_to_linear').apply_to(subproblem)

        # --- IMPORTANT: rebuild util-block variable lists on the cloned model ---
        sub_util = subproblem.component(self.original_util_block.name)
        
        # These lists are plain Python attributes and can carry stale VarData
        # references across clones. Force regeneration so update_incumbent()
        # sees variables on *this* subproblem instance.
        for attr in (
            "disjunct_list",
            "algebraic_variable_list",
            "boolean_variable_list",
            "binary_variable_list",
            "transformed_boolean_variable_list",
        ):
            if hasattr(sub_util, attr):
                delattr(sub_util, attr)

        add_disjunct_list(sub_util)
        add_algebraic_variable_list(sub_util)
        add_boolean_variable_lists(sub_util)
        add_transformed_boolean_variable_list(sub_util)
        # --- end rebuild ---

        with SuppressInfeasibleWarning():
            try:
                # Transform GDP -> algebraic model
                TransformationFactory('gdp.bigm').apply_to(subproblem)

                # Optional presolve pipeline (kept consistent with config)
                if getattr(config, 'subproblem_presolve', True):
                    fbbt(subproblem, integer_tol=config.integer_tolerance)
                    TransformationFactory('contrib.detect_fixed_vars').apply_to(
                        subproblem
                    )
                    TransformationFactory('contrib.propagate_fixed_vars').apply_to(
                        subproblem
                    )
                    TransformationFactory(
                        'contrib.deactivate_trivial_constraints'
                    ).apply_to(subproblem, tmp=False, ignore_infeasible=False)
            except InfeasibleConstraintException:
                return False, None

            minlp_args = dict(config.minlp_solver_args)
            # Ensure primal variable values are loaded onto the model for
            # incumbent updates and solution caching.
            #
            # NOTE: meta-solvers like MindtPy do not accept arbitrary solve()
            # keywords (they validate keys against an internal ConfigDict).
            # Most direct solvers already default to load_solutions=True.
            # if config.minlp_solver != 'mindtpy':
            #     minlp_args.setdefault('load_solutions', True)
            # if config.time_limit is not None and config.minlp_solver == 'gams':
            #     elapsed = get_main_elapsed_time(self.timing)
            #     remaining = max(config.time_limit - elapsed, 1)
            #     minlp_args['add_options'] = minlp_args.get('add_options', [])
            #     minlp_args['add_options'].append('option reslim=%s;' % remaining)

            # result = SolverFactory(config.minlp_solver).solve(subproblem, **minlp_args)

            # NOTE: This is a replacement to the above note
            # MindtPy is a meta-solver and may not load primal values back onto the
            # model object we pass in. In LD-BD, subproblems are continuous once the
            # discrete decisions are fixed, so solve them directly with the NLP solver.
            if config.minlp_solver == "mindtpy":
                nlp_solver = minlp_args.get("nlp_solver", None)
                if nlp_solver is None:
                    raise ValueError(
                        "gdpopt.ldbd with minlp_solver='mindtpy' requires "
                        "minlp_solver_args['nlp_solver'] to solve LD-BD subproblems."
                    )
                sub_results = SolverFactory(nlp_solver).solve(subproblem, load_solutions=True)
            else:
                # For direct MINLP solvers, ensure solutions are loaded onto subproblem
                minlp_args.setdefault("load_solutions", True)
                sub_results = SolverFactory(config.minlp_solver).solve(subproblem, **minlp_args)
            # Use the results from the solver we actually ran
            result = sub_results   

            obj = next(subproblem.component_data_objects(Objective, active=True))
            primal_bound = value(obj)

            primal_improved = self._handle_subproblem_result(
                result, subproblem, external_var_value, config, search_type
            )

        return primal_improved, primal_bound

    def _get_directions(self, dimension, config):
        """Generate neighborhood search directions.

        Parameters
        ----------
        dimension : int
            Dimensionality of the external-variable space.
        config : ConfigBlock
            Configuration block specifying the direction norm.

        Returns
        -------
        list[tuple[int, ...]]
            Direction vectors.

            - If ``config.direction_norm == DirectionNorm.L2``: standard basis
              vectors and their negatives.
            - If ``config.direction_norm == DirectionNorm.Linf``: all
              combinations from ``{-1, 0, 1}^dimension`` excluding the zero
              vector.
        """
        if config.direction_norm == DirectionNorm.L2:
            directions = []
            for i in range(dimension):
                directions.append(tuple([0] * i + [1] + [0] * (dimension - i - 1)))
                directions.append(tuple([0] * i + [-1] + [0] * (dimension - i - 1)))
            return directions
        elif config.direction_norm == DirectionNorm.Linf:
            directions = list(it.product([-1, 0, 1], repeat=dimension))
            directions.remove((0,) * dimension)  # Remove the zero direction
            return directions

        raise ValueError(
            "Unrecognized direction_norm=%r (expected %s or %s)"
            % (config.direction_norm, DirectionNorm.L2, DirectionNorm.Linf)
        )

    def _check_valid_neighbor(self, neighbor):
        """Check whether a neighbor point is valid.

        A neighbor is valid if it has not been evaluated and lies within the
        configured bounds.

        Parameters
        ----------
        neighbor : tuple[int, ...]
            Candidate neighbor point.

        Returns
        -------
        bool
            ``True`` if the neighbor is valid.
        """
        if self.data_manager.is_visited(neighbor):
            return False
        if not self.data_manager.is_valid_point(neighbor):
            return False
        return True

    def _generate_neighbors(self, current_point, config):
        """Generate valid, unvisited neighbors from a current point.

        Parameters
        ----------
        current_point : tuple[int, ...]
            Current external-variable point.
        config : ConfigBlock
            GDPopt configuration block.

        Returns
        -------
        list[tuple[tuple[int, ...], tuple[int, ...]]]
            List of ``(neighbor_point, direction)`` pairs.
        """
        directions = self._get_directions(self.number_of_external_variables, config)
        valid_neighbors = []

        for direction in directions:
            neighbor = tuple(map(sum, zip(current_point, direction)))

            # Use DataManager to check bounds and visited status
            if self.data_manager.is_valid_point(
                neighbor
            ) and not self.data_manager.is_visited(neighbor):
                valid_neighbors.append((neighbor, direction))

        return valid_neighbors

    def _handle_subproblem_result(
        self, subproblem_result, subproblem, external_var_value, config, search_type
    ):
        """Process the result of a discrete subproblem solve.

        This inspects the solver termination condition, updates bounds when
        appropriate, and logs status.

        Parameters
        ----------
        subproblem_result : SolverResults
            Solver results object.
        subproblem : ConcreteModel
            Subproblem model instance.
        external_var_value : tuple[int, ...]
            External-variable point used to create the subproblem.
        config : ConfigBlock
            GDPopt configuration block.
        search_type : str
            Label for logging (e.g., "Neighbor").

        Returns
        -------
        bool
            ``True`` if the primal bound improved; ``False`` otherwise.
        """
        if subproblem_result is None:
            return False
        if subproblem_result.solver.termination_condition in {
            tc.optimal,
            tc.feasible,
            tc.globallyOptimal,
            tc.locallyOptimal,
            tc.maxTimeLimit,
            tc.maxIterations,
            tc.maxEvaluations,
        }:
            # Cache the feasible-point solution values regardless of whether
            # the point improves the incumbent. This enables Step5 / end-stage
            # incumbent updates without re-solving.
            # try:
            #     sub_util = subproblem.component(self.original_util_block.name)
            #     self._cache_point_solution(external_var_value, sub_util)
            # except Exception:
            #     # Caching is best-effort; it must not interfere with the
            #     # optimization algorithm.
            #     pass
            # NOTE: the above approach to caching relied on the util block's variable lists, which can carry stale references across clones. The below approach uses ComponentUID lookups to avoid this issue.
            try:
                # Cache using original-model UIDs, reading values from the solved clone
                self._cache_point_solution(external_var_value, subproblem)
            except Exception:
                pass

            # NOTE: Not all solver interfaces reliably populate
            # results.problem.upper_bound / lower_bound (they may be None even
            # when a feasible/optimal solution is loaded onto the model).
            # For discrete algorithms, we need a numeric primal bound to:
            #   (1) update UB/LB,
            #   (2) record an incumbent,
            #   (3) transfer the incumbent back to the original model.
            primal_bound = (
                subproblem_result.problem.upper_bound
                if self.objective_sense == minimize
                else subproblem_result.problem.lower_bound
            )
            if primal_bound is None:
                obj = next(subproblem.component_data_objects(Objective, active=True))
                primal_bound = value(obj)
            primal_improved = self._update_bounds_after_solve(
                search_type,
                primal=primal_bound,
                logger=config.logger,
                current_point=external_var_value,
            )
            if primal_improved:
                self.update_incumbent(
                    subproblem.component(self.original_util_block.name)
                )
            return primal_improved
        return False

    def _log_header(self, logger):
        """Log the tabular header for discrete-search progress.

        Parameters
        ----------
        logger : logging.Logger
            Logger to write to.
        """
        logger.info(
            '================================================================='
            '===================================='
        )
        logger.info(
            '{:^9} | {:^15} | {:^20} | {:^11} | {:^11} | {:^8} | {:^7}\n'.format(
                'Iteration',
                'Search Type',
                'External Variables',
                'Lower Bound',
                'Upper Bound',
                'Gap',
                'Time(s)',
            )
        )

    def _log_current_state(
        self, logger, search_type, current_point, primal_improved=False
    ):
        """Log a single iteration state line.

        Parameters
        ----------
        logger : logging.Logger
            Logger to write to.
        search_type : str
            Label for the current action (e.g., "Anchor", "Neighbor").
        current_point : tuple[int, ...]
            External-variable point.
        primal_improved : bool, optional
            Whether the primal bound improved.
        """
        star = "*" if primal_improved else ""
        logger.info(
            self.log_formatter.format(
                self.iteration,
                search_type,
                str(current_point),
                self.LB,
                self.UB,
                self.relative_gap(),
                get_main_elapsed_time(self.timing),
                star,
            )
        )

    def _update_bounds_after_solve(
        self, search_type, primal=None, dual=None, logger=None, current_point=None
    ):
        """Update bounds after a subproblem solve and optionally log state.

        Parameters
        ----------
        search_type : str
            Label for logging.
        primal : float, optional
            New primal bound candidate.
        dual : float, optional
            New dual bound candidate.
        logger : logging.Logger, optional
            Logger used to log state.
        current_point : tuple[int, ...], optional
            Point associated with the update.

        Returns
        -------
        bool
            ``True`` if the primal bound improved.
        """
        primal_improved = self._update_bounds(primal, dual)
        if logger is not None:
            self._log_current_state(logger, search_type, current_point, primal_improved)

        return primal_improved
