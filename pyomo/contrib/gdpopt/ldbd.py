#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple
import traceback
from pyomo.common.config import document_kwargs_from_configdict, ConfigValue
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.ldsda import GDP_LDSDA_Solver
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    add_util_block,
    add_disjunction_list,
    add_disjunct_list,
    add_algebraic_variable_list,
    add_boolean_variable_lists,
    add_transformed_boolean_variable_list,
)
from pyomo.contrib.gdpopt.config_options import (
    _add_nlp_solver_configs,
    _add_ldsda_configs,
    _add_mip_solver_configs,
    _add_tolerance_configs,
    _add_nlp_solve_configs,
)
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, get_main_elapsed_time
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import minimize, Suffix, TransformationFactory, maximize, Any
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr.logical_expr import ExactlyExpression
from pyomo.common.dependencies import attempt_import
from pyomo.core.base import Var, Constraint, NonNegativeReals, ConstraintList, Objective, Reals, value, ConcreteModel, NonNegativeIntegers



it, it_available = attempt_import('itertools')
tabulate, tabulate_available = attempt_import('tabulate')

# Data tuple for external variables.
ExternalVarInfo = namedtuple(
    'ExternalVarInfo',
    [
        'exactly_number',  # number of external variables for this type
        'Boolean_vars',  # list with names of the ordered Boolean variables to be reformulated
        'UB',  # upper bound on external variable
        'LB',  # lower bound on external variable
    ],
)


@SolverFactory.register(
    'gdpopt.ldbdd',
    doc="The LD-BD (Logic-based Discrete Benders Decomposition solver)"
    "Generalized Disjunctive Programming (GDP) solver",
)
class GDP_LDBD_Solver(GDP_LDSDA_Solver):
    """The GDPopt (Generalized Disjunctive Programming optimizer)
    LD-BD (Logic-based Discrete Benders Decomposition (LD-SDA)) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions.
    """

    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG, default_solver='ipopt')
    _add_nlp_solve_configs(
        CONFIG, default_nlp_init_method=restore_vars_to_original_values
    )
    _add_tolerance_configs(CONFIG)
    _add_ldsda_configs(CONFIG)

    CONFIG.declare('infinity_output', ConfigValue(
        default=1e8,
        domain=float,
        description="Value to use for infeasible points instead of infinity."
    ))

    algorithm = 'LDBD'

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        return super().solve(model, **kwds)
    


    def _solve_gdp(self, model, config):
        """Solve the GDP model.

        Parameters
        ----------
        model : ConcreteModel
            The GDP model to be solved
        config : ConfigBlock
            GDPopt configuration block
        """
        logger = config.logger
        self.log_formatter = (
            '{:>9}   {:>15}   {:>20}   {:>11.5f}   {:>11.5f}   {:>8.2%}   {:>7.2f}  {}'
        )
        
        # Initialize storage for subproblem data at the beginning of the optimization
        self.master_points = []  # List to track points selected by the master problem
        self.current_point = tuple(config.starting_point)
        self.explored_point_set = set()
        self.explored_point_dict = {}  # {point_key: subproblem_obj_value}


        # Initialize attributes before they are used
        self.current_p_coefficients = []
        self.current_alpha_value = None

        # Debugging: Print or log the initial current point, explored set, and explored dictionary
        logger.info(f"Initial current point: {self.current_point}")
        logger.debug(f"Initial explored point set: {self.explored_point_set}")
        logger.debug(f"Initial explored point dictionary: {self.explored_point_dict}")

        # Create utility block on the original model so that we will be able to
        # copy solutions between
        util_block = self.original_util_block = add_util_block(model)
        add_disjunct_list(util_block)
        add_algebraic_variable_list(util_block)
        add_boolean_variable_lists(util_block)

        # We will use the working_model to perform the LDBD search.
        self.working_model = model.clone()
        # TODO: I don't like the name way, try something else?
        self.working_model_util_block = self.working_model.component(util_block.name)

        add_disjunction_list(self.working_model_util_block)
        TransformationFactory('core.logical_to_linear').apply_to(self.working_model)
        # Now that logical_to_disjunctive has been called.
        add_transformed_boolean_variable_list(self.working_model_util_block)
        self._get_external_information(self.working_model_util_block, config)

        self.directions = self._get_directions(
            self.number_of_external_variables, config
        )

        # Add the BigM suffix if it does not already exist. Used later during
        # nonlinear constraint activation.
        if not hasattr(self.working_model_util_block, 'BigM'):
            self.working_model_util_block.BigM = Suffix()
        self._log_header(logger)
        # Step 1
        # Solve the initial point
        _ = self._solve_GDP_subproblem(self.current_point, 'Initial point', config, iteration=1)

        # # Debugging: Print or log the updated sets and dictionary after solving the initial point
        # logger.debug(f"After solving initial point:")
        # logger.debug(f"Explored point set: {self.explored_point_set}")
        # logger.debug(f"Explored point dictionary: {self.explored_point_dict}")

        self.neighbor_search(config, iteration=1)

        # # Debugging: Print or log the updated sets and dictionary after neighbor search
        # print(f"After neighbor search:")
        # print(f"Explored point set: {self.explored_point_set}")
        # print(f"Explored point dictionary: {self.explored_point_dict}")

        # Step 2
        self.master_problem = self.generate_master_problem(config)
        # Generate the Benders cuts
        # Input: self.explored_point_dict = {} # {external_var_value: lower_bound}
        # Output: self.benders_cuts = [] # benders cut for master problem in each iteration
        # Main loop for generating Benders cuts and solving the master problem

        # tolerance = config.get('tolerance', 1e-5)  # Default tolerance if not set in config
        # z_diff = float('inf')
        # iteration = 1
        # Main loop for generating Benders cuts and solving the master problem
        if 'tolerance' not in config:
            config.declare('tolerance', ConfigValue(default=1e-5, domain=float, description="Convergence tolerance"))
            
        tolerance = float(value(config.tolerance))
        z_diff = float('inf')  # Initialize z_diff as infinity
        iteration = 1

        while z_diff > tolerance:
            logger.info(f"Iteration {iteration}")

            # Step 3: Perform neighbor search and accumulate the points into the dictionary
            self.neighbor_search(config, iteration)

            # Step 4: Generate Benders cuts based on the current solution of the subproblem
            self.generate_benders_cuts(config)

            # Step 6: Refine previously generated Benders cuts (Add your refinement logic here)
            self.refine_benders_cuts(config, iteration)

            # Step 5: Solve the master problem with the newly generated Benders cuts
            external_var_values, z_value = self.solve_master_problem(config)

            # Step 6: Solve the subproblem for the newly generated external variable values
            # Evaluate the subproblem objective value for the new external variable values from the master problem
            primal_improved, new_sub_obj_value = self._solve_GDP_subproblem(external_var_values, 'Benders cut generation', config, iteration)

             # Store the subproblem objective value
            point_key = tuple(external_var_values)
            self.explored_point_dict[point_key] = new_sub_obj_value
            self.current_point = external_var_values

            # Log the results of the neighbor search and the cumulative explored dictionary
            logger.info(f"Current Coordinates: {tuple(external_var_values)}")
            logger.info(f"Cumulative Dictionary: {self.explored_point_dict}")

            # Step 7: Calculate the difference between the master problem objective (z_value) and the subproblem objective
            z_diff = abs(z_value - new_sub_obj_value)
            logger.info(f"z_value: {z_value}, subproblem objective: {new_sub_obj_value}, z_diff: {z_diff}")

            # If the solution converged, stop the loop
            if z_diff <= tolerance:
                logger.info("Converged to solution.")
                break

            # Update the current point for the next iteration
            self.current_point = external_var_values

            # Optionally, you can add checks for other termination criteria (e.g., max iterations)
            if self.any_termination_criterion_met(config):
                logger.info("Termination criterion met. Exiting.")
                break

            iteration += 1

        logger.info(f"Solved in {iteration} iterations.")

    # def any_termination_criterion_met(self, config):
    #     return self.reached_iteration_limit(config) or self.reached_time_limit(config)
        
    def neighbor_search(self, config, iteration):
        """Perform neighbor search and store subproblem objective values."""
        for direction in self.directions:
            neighbor = tuple(map(sum, zip(self.current_point, direction)))
            if self._check_valid_neighbor(neighbor):
                _, sub_obj_value = self._solve_GDP_subproblem(
                    neighbor, 'Neighbor search', config, iteration
                )
                # Store the subproblem objective value
                self.explored_point_dict[neighbor] = sub_obj_value
                # if primal_improved:
                #     locally_optimal = False
                #     best_neighbor = neighbor
                #     self.best_direction = direction
        # if not locally_optimal:
        #     self.current_point = best_neighbor
        # return locally_optimal
    

    def _solve_GDP_subproblem(self, external_var_value, search_type, config, iteration):
        """Solve the GDP subproblem with disjunctions fixed according to the external variable.
        Parameters
        ----------
        external_var_value : list
            The values of the external variables to be evaluated
        search_type : str
            The type of search, neighbor search or line search
        config : ConfigBlock
            GDPopt configuration block
        iteration : int
            The current iteration number for the optimization

        Returns
        -------
        bool
            whether the primal bound is improved
        float
            The subproblem's objective value (or None if infeasible)
        """

        point_key = tuple(external_var_value)

        # Check if this subproblem has already been solved
        if point_key in self.explored_point_dict:
            subproblem_obj_value = self.explored_point_dict[point_key]
            config.logger.info(f"Retrieved stored subproblem objective for point {point_key}: {subproblem_obj_value}")
            return False, subproblem_obj_value

            
        # Solve the subproblem if it's not stored
        self.fix_disjunctions_with_external_var(external_var_value)
        subproblem = self.working_model.clone()
        TransformationFactory('core.logical_to_linear').apply_to(subproblem)

        try:
            with SuppressInfeasibleWarning():
                try:
                    fbbt(subproblem, integer_tol=config.integer_tolerance)
                    TransformationFactory('contrib.detect_fixed_vars').apply_to(subproblem)
                    TransformationFactory('contrib.propagate_fixed_vars').apply_to(subproblem)
                    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(subproblem, tmp=False, ignore_infeasible=False)
                    TransformationFactory('gdp.bigm').apply_to(subproblem)
                except InfeasibleConstraintException:
                    config.logger.info(f"Subproblem infeasible for external variable values: {external_var_value}")
                    self.store_explored_point(external_var_value, config.infinity_output)
                    print(self.explored_point_dict)
                    return False, config.infinity_output

                # Solve the subproblem
                minlp_args = dict(config.minlp_solver_args)
                if config.time_limit is not None and config.minlp_solver == 'gams':
                    elapsed = get_main_elapsed_time(self.timing)
                    remaining = max(config.time_limit - elapsed, 1)
                    minlp_args['add_options'] = minlp_args.get('add_options', [])
                    minlp_args['add_options'].append(f'option reslim={remaining};')

                result = SolverFactory(config.minlp_solver).solve(subproblem, tee=False, **minlp_args)

                # Check the solver's termination condition
                if result.solver.termination_condition in {tc.optimal, tc.feasible, tc.locallyOptimal}:
                    subproblem_obj_value = value(subproblem.obj)

                    # Check if the objective value is valid
                    if subproblem_obj_value is None or subproblem_obj_value != subproblem_obj_value:  # Check for NaN
                        config.logger.warning(f"Objective value is NaN or None for external variables {external_var_value}.")
                        subproblem_obj_value = float('nan')

                    # Store the result in the cumulative dictionary
                    self.store_explored_point(external_var_value, subproblem_obj_value)

                    # Handle the result and check for primal improvement
                    primal_improved = self._handle_subproblem_result(result, subproblem, external_var_value, config, search_type)
                    return primal_improved, subproblem_obj_value
                else:
                    config.logger.warning(f"Solver returned a non-optimal status: {result.solver.termination_condition}")
                    self.store_explored_point(external_var_value, config.infinity_output)
                    return False, config.infinity_output
        except RuntimeError as e:
            config.logger.warning(f"RuntimeError encountered for external variables {external_var_value}: {str(e)}")
            self.store_explored_point(external_var_value, None)
            return False, None

    def generate_master_problem(self, config):
        # Initialize the master problem as a ConcreteModel
        master_problem = ConcreteModel()

        # Set up the external variables (these are integer variables)
        util_block = self.working_model_util_block
        external_vars = []

        # Use initial point to set initial values for external variables in the first iteration
        for i, external_var_info in enumerate(util_block.external_var_info_list):
            var_name = f'external_var_{i}'
            initial_value = self.current_point[i]  # Set initial point value for first iteration
            var = Var(within=NonNegativeIntegers, bounds=(external_var_info.LB, external_var_info.UB), initialize=initial_value)
            setattr(master_problem, var_name, var)
            external_vars.append(getattr(master_problem, var_name))

        # Explicitly assign the external_vars list to the master_problem for future access
        master_problem.external_vars = external_vars

        # Create a placeholder for the Benders cuts
        master_problem.benders_cuts = Constraint(Any)

        # Define the objective variable 'z' and the objective function
        master_problem.z = Var(within=Reals, initialize=0)  # Objective variable 'z'

        # Define the objective function as minimizing 'z'
        master_problem.obj = Objective(expr=master_problem.z, sense=minimize)

        return master_problem

    def solve_master_problem(self, config):
        """
        Solve the master problem with the Benders cuts applied.
        Ensure that we use the correct solver, like Gurobi, for the master problem.
        """
        # Ensure that we are using the correct solver for the master problem
        milp_solver = config.mip_solver if config.mip_solver else "gurobi"

        # Create a solver object for the MILP solver
        solver = SolverFactory(milp_solver)
        
        # Solve the master problem
        result = solver.solve(self.master_problem, tee=False)

        # Check if the solver's termination condition is optimal
        if result.solver.termination_condition != tc.optimal:
            raise RuntimeError(f"Master problem did not converge to an optimal solution. "
                            f"Termination condition: {result.solver.termination_condition}")

        # Retrieve the values of the external variables from the solution
        external_var_values = [int(value(var)) for var in self.master_problem.external_vars]
        
        # Retrieve the objective value (z)
        objective_value = value(self.master_problem.z)

        # # Optionally, print the values for debugging
        # print(f"Solved master problem. Objective value (z): {objective_value}")
        # print(f"External variables: {external_var_values}")

        # Return the values of the external variables and the objective value
        return external_var_values, objective_value


    def generate_benders_cuts(self, config):
        """Generate a Benders cut for the current point obtained from the master problem."""
        # Step 1: Get the current external variable values from the master problem
        external_var_values = [int(value(var)) for var in self.master_problem.external_vars]
        point_key = tuple(external_var_values)

        # Step 2: Store the current point in master_points
        self.master_points.append(point_key)

        # Step 3: Check if the current point is infeasible
        subproblem_obj_value = self.explored_point_dict.get(point_key, config.infinity_output)
        is_infeasible = subproblem_obj_value == config.infinity_output

        if is_infeasible:
            # Handle infeasibility by generating a feasibility cut
            config.logger.info(f"Generating feasibility cut for external variables: {external_var_values}")
            # Generate feasibility cut for the current point
            benders_cut_expr = self._generate_cut_for_point(external_var_values, config, infeasible=True)
        else:
            # Handle feasible subproblem case
            config.logger.info(f"Generating Benders cut for external variables: {external_var_values}")
            # Generate Benders cut for the current point
            benders_cut_expr = self._generate_cut_for_point(external_var_values, config, infeasible=False)

        # Debugging: Ensure that benders_cut_expr and z exist and are not None
        if benders_cut_expr is None:
            config.logger.warning(f"Failed to generate a Benders cut for point {external_var_values}")
            return  # Don't add a cut if it's None

        if self.master_problem.z is None:
            config.logger.error(f"Master problem objective variable 'z' is None. This should not happen.")
            return  # Prevents adding a None variable to the constraint

        # Step 4: Add the Benders cut to the master problem
        try:
            config.logger.info(f"Adding Benders cut: {benders_cut_expr} <= {self.master_problem.z}")
            self.master_problem.benders_cuts[point_key] = benders_cut_expr <= self.master_problem.z
        except Exception as e:
            config.logger.error(f"Error when adding Benders cut for point {external_var_values}: {e}")
            raise



    def _generate_cut_for_point(self, external_var_values, config, infeasible=False):
        # Generate cut for the current point and external variable values
        num_vars = len(external_var_values)
        cut_model = ConcreteModel()

        # Define variables p and alpha
        cut_model.p = Var(range(num_vars), within=Reals, initialize=0.0)
        cut_model.alpha = Var(within=Reals, initialize=0.0)

        # Define the objective function
        cut_model.objective = Objective(
            expr=sum(
                cut_model.p[i] * external_var_values[i]
                for i in range(num_vars)
            ) + cut_model.alpha,
            sense=maximize
        )

        # Add constraints based on all previously explored feasible points
        cut_model.constraints = ConstraintList()
        for coord, rhs_value in self.explored_point_dict.items():
            if rhs_value != config.infinity_output:
                # Add constraints for feasible points
                constraint_expr = (
                    sum(
                        cut_model.p[i] * coord[i]
                        for i in range(num_vars)
                    ) + cut_model.alpha <= rhs_value
                )
                cut_model.constraints.add(constraint_expr)
                # config.logger.info(f"Added constraint: {' + '.join([f'{coord[i]}*p[{i}]' for i in range(num_vars)])} + alpha <= {rhs_value} for point {coord}")

        # Handle infeasibility
        if infeasible:
            # For infeasible points, create a feasibility cut
            infeasible_expr = sum(
                cut_model.p[i] * external_var_values[i] for i in range(num_vars)
            ) + cut_model.alpha
            cut_model.constraints.add(infeasible_expr <= config.infinity_output)
            config.logger.info(f"Generated feasibility cut for point {external_var_values}: {infeasible_expr} <= {config.infinity_output}")

        # Solve the cut-generating subproblem
        solver = SolverFactory('gurobi')
        solver.options['Method'] = 2  # Use interior point method
        result = solver.solve(cut_model, tee=False)

        # Check if the solver found an optimal solution
        if result.solver.termination_condition != tc.optimal:
            config.logger.warning(f"Cut-generating subproblem did not converge to an optimal solution for point {external_var_values}.")
            return None  # Handle this situation as needed

        # Extract p coefficients and alpha from the solution
        p_coefficients = [value(cut_model.p[i]) for i in range(num_vars)]
        alpha_value = value(cut_model.alpha)

        # Generate the Benders cut expression
        benders_cut_expr = sum(
            p_coefficients[i] * self.master_problem.external_vars[i]
            for i in range(num_vars)
        ) + alpha_value

        # Log the cut for debugging
        cut_type = 'feasibility' if infeasible else 'Benders'
        config.logger.info(f"Generated {cut_type} cut: {benders_cut_expr} <= z")
        config.logger.info(f"p coefficients: {p_coefficients}, alpha: {alpha_value}")

        return benders_cut_expr

    def refine_benders_cuts(self, config, iteration):
        """
        Refine previously generated Benders cuts for the points selected by the master problem.

        Parameters
        ----------
        config : ConfigBlock
            GDPopt configuration block.
        iteration : int
            The current iteration number for the optimization process.
        """
        # Skip refinement if it's the first iteration
        if iteration <= 1:
            config.logger.info("Skipping cut refinement on iteration 1.")
            return

        # Start refinement process from iteration 2 onwards
        config.logger.info(f"Refining Benders cuts in iteration {iteration}.")

        # Refine cuts for previous master problem solutions
        for point_key in self.master_points[:-1]:  # Exclude the current point
            external_var_values = list(point_key)
            sub_obj_value = self.explored_point_dict.get(point_key)

            # Handle infeasible points
            if sub_obj_value == config.infinity_output:
                config.logger.info(f"Regenerating feasibility cut for infeasible point {external_var_values}")
                benders_cut_expr = self._generate_cut_for_point(external_var_values, config, infeasible=True)
                if benders_cut_expr is not None:
                    self.master_problem.benders_cuts[point_key] = benders_cut_expr <= self.master_problem.z
                    config.logger.info(f"Added regenerated feasibility cut for infeasible point {point_key}.")
                else:
                    config.logger.warning(f"Failed to generate feasibility cut for point {point_key}.")
                continue  # Move to the next point
                        
            if sub_obj_value is not None:
                # Update the Benders cut using the stored subproblem objective value
                self.update_benders_cut(external_var_values, sub_obj_value, config)
            else:
                config.logger.warning(f"No subproblem objective value found for point {point_key}.")




    def store_explored_point(self, external_var_value, subproblem_obj_value):
        """Store the explored point and its corresponding objective value by iteration.

        Parameters
        ----------
        external_var_value : list
            The values of the external variables being evaluated
        subproblem_obj_value : float
            The objective value for the subproblem at the given external variable values
        iteration : int
            The iteration number for which the explored point is being stored
        """
        point_key = tuple(external_var_value)
        self.explored_point_dict[point_key] = subproblem_obj_value

    def update_benders_cut(self, external_var_values, subproblem_obj_value, config):
        """
        Update the Benders cut associated with the given external variable values.

        Parameters
        ----------
        external_var_values : list or tuple
            The external variable values corresponding to the cut to be updated.
        subproblem_obj_value : float
            The updated subproblem objective value for the given external variable values.
        config : ConfigBlock
            GDPopt configuration block.
        """
        point_key = tuple(external_var_values)
        num_vars = len(external_var_values)

        # Remove the old cut from the master problem
        if point_key in self.master_problem.benders_cuts:
            del self.master_problem.benders_cuts[point_key]
            config.logger.info(f"Removed old Benders cut for point {point_key}.")
        else:
            config.logger.warning(f"No existing Benders cut found for point {point_key} to update.")

        # Generate a new cut for this point
        benders_cut_expr = self._generate_cut_for_point(external_var_values, config)

        # Check if the Benders cut was successfully generated
        if benders_cut_expr is not None:
            # Add the updated cut to the master problem
            self.master_problem.benders_cuts[point_key] = benders_cut_expr <= self.master_problem.z
            config.logger.info(f"Added updated Benders cut for point {point_key} to the master problem.")
        else:
            config.logger.warning(f"Failed to generate Benders cut for point {external_var_values}. No cut added.")



# # Unintended Code: Multiple Cuts generated for each iteration and it refines the cuts.
# #  ___________________________________________________________________________
# #
# #  Pyomo: Python Optimization Modeling Objects
# #  Copyright (c) 2008-2022
# #  National Technology and Engineering Solutions of Sandia, LLC
# #  Under the terms of Contract DE-NA0003525 with National Technology and
# #  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# #  rights in this software.
# #  This software is distributed under the 3-clause BSD License.
# #  ___________________________________________________________________________

# from collections import namedtuple
# import traceback
# from pyomo.common.config import document_kwargs_from_configdict, ConfigValue
# from pyomo.common.errors import InfeasibleConstraintException
# from pyomo.contrib.fbbt.fbbt import fbbt
# from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
# from pyomo.contrib.gdpopt.ldsda import GDP_LDSDA_Solver
# from pyomo.contrib.gdpopt.create_oa_subproblems import (
#     add_util_block,
#     add_disjunction_list,
#     add_disjunct_list,
#     add_algebraic_variable_list,
#     add_boolean_variable_lists,
#     add_transformed_boolean_variable_list,
# )
# from pyomo.contrib.gdpopt.config_options import (
#     _add_nlp_solver_configs,
#     _add_ldsda_configs,
#     _add_mip_solver_configs,
#     _add_tolerance_configs,
#     _add_nlp_solve_configs,
# )
# from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
# from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, get_main_elapsed_time
# from pyomo.contrib.satsolver.satsolver import satisfiable
# from pyomo.core import minimize, Suffix, TransformationFactory, maximize, Any
# from pyomo.opt import SolverFactory
# from pyomo.opt import TerminationCondition as tc
# from pyomo.core.expr.logical_expr import ExactlyExpression
# from pyomo.common.dependencies import attempt_import
# from pyomo.core.base import Var, Constraint, NonNegativeReals, ConstraintList, Objective, Reals, value, ConcreteModel, NonNegativeIntegers



# it, it_available = attempt_import('itertools')
# tabulate, tabulate_available = attempt_import('tabulate')

# # Data tuple for external variables.
# ExternalVarInfo = namedtuple(
#     'ExternalVarInfo',
#     [
#         'exactly_number',  # number of external variables for this type
#         'Boolean_vars',  # list with names of the ordered Boolean variables to be reformulated
#         'UB',  # upper bound on external variable
#         'LB',  # lower bound on external variable
#     ],
# )


# @SolverFactory.register(
#     'gdpopt.ldbdd',
#     doc="The LD-BD (Logic-based Discrete Benders Decomposition solver)"
#     "Generalized Disjunctive Programming (GDP) solver",
# )
# class GDP_LDBD_Solver(GDP_LDSDA_Solver):
#     """The GDPopt (Generalized Disjunctive Programming optimizer)
#     LD-BD (Logic-based Discrete Benders Decomposition (LD-SDA)) solver.

#     Accepts models that can include nonlinear, continuous variables and
#     constraints, as well as logical conditions.
#     """

#     CONFIG = _GDPoptAlgorithm.CONFIG()
#     _add_mip_solver_configs(CONFIG)
#     _add_nlp_solver_configs(CONFIG, default_solver='ipopt')
#     _add_nlp_solve_configs(
#         CONFIG, default_nlp_init_method=restore_vars_to_original_values
#     )
#     _add_tolerance_configs(CONFIG)
#     _add_ldsda_configs(CONFIG)

#     CONFIG.declare('infinity_output', ConfigValue(
#         default=1e8,
#         domain=float,
#         description="Value to use for infeasible points instead of infinity."
#     ))

#     algorithm = 'LDBDD'

#     # Override solve() to customize the docstring for this solver
#     @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
#     def solve(self, model, **kwds):
#         return super().solve(model, **kwds)
    


#     def _solve_gdp(self, model, config):
#         """Solve the GDP model.

#         Parameters
#         ----------
#         model : ConcreteModel
#             The GDP model to be solved
#         config : ConfigBlock
#             GDPopt configuration block
#         """
#         logger = config.logger
#         self.log_formatter = (
#             '{:>9}   {:>15}   {:>20}   {:>11.5f}   {:>11.5f}   {:>8.2%}   {:>7.2f}  {}'
#         )
        
#         # Initialize storage for subproblem data at the beginning of the optimization
#         self.current_point = tuple(config.starting_point)
#         self.explored_point_set = set()
#         self.explored_point_dict = {}  # {iteration: {external_var_values_tuple: subproblem_obj_value}}

#         # Initialize attributes before they are used
#         self.current_p_coefficients = []
#         self.current_alpha_value = None

#         # Debugging: Print or log the initial current point, explored set, and explored dictionary
#         print(f"Initial current point: {self.current_point}")
#         print(f"Initial explored point set: {self.explored_point_set}")
#         print(f"Initial explored point dictionary: {self.explored_point_dict}")

#         # Create utility block on the original model so that we will be able to
#         # copy solutions between
#         util_block = self.original_util_block = add_util_block(model)
#         add_disjunct_list(util_block)
#         add_algebraic_variable_list(util_block)
#         add_boolean_variable_lists(util_block)

#         # We will use the working_model to perform the LDBD search.
#         self.working_model = model.clone()
#         # TODO: I don't like the name way, try something else?
#         self.working_model_util_block = self.working_model.component(util_block.name)

#         add_disjunction_list(self.working_model_util_block)
#         TransformationFactory('core.logical_to_linear').apply_to(self.working_model)
#         # Now that logical_to_disjunctive has been called.
#         add_transformed_boolean_variable_list(self.working_model_util_block)
#         self._get_external_information(self.working_model_util_block, config)

#         self.directions = self._get_directions(
#             self.number_of_external_variables, config
#         )

#         # Add the BigM suffix if it does not already exist. Used later during
#         # nonlinear constraint activation.
#         if not hasattr(self.working_model_util_block, 'BigM'):
#             self.working_model_util_block.BigM = Suffix()
#         self._log_header(logger)
#         # Step 1
#         # Solve the initial point
#         _ = self._solve_GDP_subproblem(self.current_point, 'Initial point', config, iteration=1)

#         # # Debugging: Print or log the updated sets and dictionary after solving the initial point
#         # print(f"After solving initial point:")
#         # print(f"Explored point set: {self.explored_point_set}")
#         # print(f"Explored point dictionary: {self.explored_point_dict}")

#         self.neighbor_search(config, iteration=1)

#         # # Debugging: Print or log the updated sets and dictionary after neighbor search
#         # print(f"After neighbor search:")
#         # print(f"Explored point set: {self.explored_point_set}")
#         # print(f"Explored point dictionary: {self.explored_point_dict}")

#         # Step 2
#         self.master_problem = self.generate_master_problem(config)
#         # Generate the Benders cuts
#         # Input: self.explored_point_dict = {} # {external_var_value: lower_bound}
#         # Output: self.benders_cuts = [] # benders cut for master problem in each iteration
#         # Main loop for generating Benders cuts and solving the master problem

#         # tolerance = config.get('tolerance', 1e-5)  # Default tolerance if not set in config
#         # z_diff = float('inf')
#         # iteration = 1
#         # Main loop for generating Benders cuts and solving the master problem
#         if 'tolerance' not in config:
#             config.declare('tolerance', ConfigValue(default=1e-5, domain=float, description="Convergence tolerance"))
            
#         tolerance = float(value(config.tolerance))
#         z_diff = float('inf')  # Initialize z_diff as infinity
#         iteration = 1

#         while z_diff > tolerance:
#             logger.info(f"Iteration {iteration}")

#             # Step 3: Perform neighbor search and accumulate the points into the dictionary
#             self.neighbor_search(config, iteration)

#             # Step 4: Generate Benders cuts based on the current solution of the subproblem
#             self.generate_benders_cuts(config)

#             # Step 6: Refine previously generated Benders cuts (Add your refinement logic here)
#             self.refine_benders_cuts(config, iteration)

#             # Step 5: Solve the master problem with the newly generated Benders cuts
#             external_var_values, z_value = self.solve_master_problem(config)

#             # Step 6: Solve the subproblem for the newly generated external variable values
#             # Evaluate the subproblem objective value for the new external variable values from the master problem
#             primal_improved, new_sub_obj_value = self._solve_GDP_subproblem(external_var_values, 'Benders cut generation', config, iteration)

#             # Log the results of the neighbor search and the cumulative explored dictionary
#             logger.info(f"Current Coordinates: {tuple(external_var_values)}")
#             logger.info(f"Cumulative Dictionary: {self.explored_point_dict}")

#             # Step 7: Calculate the difference between the master problem objective (z_value) and the subproblem objective
#             z_diff = abs(z_value - new_sub_obj_value)
#             logger.info(f"z_value: {z_value}, subproblem objective: {new_sub_obj_value}, z_diff: {z_diff}")

#             # If the solution converged, stop the loop
#             if z_diff <= tolerance:
#                 logger.info("Converged to solution.")
#                 break

#             # Update the current point for the next iteration
#             self.current_point = external_var_values

#             # Optionally, you can add checks for other termination criteria (e.g., max iterations)
#             if self.any_termination_criterion_met(config):
#                 logger.info("Termination criterion met. Exiting.")
#                 break

#             iteration += 1

#         logger.info(f"Solved in {iteration} iterations.")

#     # def any_termination_criterion_met(self, config):
#     #     return self.reached_iteration_limit(config) or self.reached_time_limit(config)
        
#     def neighbor_search(self, config, iteration):
#         """Function that evaluates a group of given points and returns the best

#         Parameters
#         ----------
#         config : ConfigBlock
#             GDPopt configuration block
#         """
#         # locally_optimal = True
#         # best_neighbor = None
#         # self.best_direction = None  # reset best direction
#         for direction in self.directions:
#             neighbor = tuple(map(sum, zip(self.current_point, direction)))
#             if self._check_valid_neighbor(neighbor):
#                 primal_improved = self._solve_GDP_subproblem(
#                     neighbor, 'Neighbor search', config, iteration
#                 )
#                 # if primal_improved:
#                 #     locally_optimal = False
#                 #     best_neighbor = neighbor
#                 #     self.best_direction = direction
#         # if not locally_optimal:
#         #     self.current_point = best_neighbor
#         # return locally_optimal
    

#     def _solve_GDP_subproblem(self, external_var_value, search_type, config, iteration):
#         """Solve the GDP subproblem with disjunctions fixed according to the external variable.
#         Parameters
#         ----------
#         external_var_value : list
#             The values of the external variables to be evaluated
#         search_type : str
#             The type of search, neighbor search or line search
#         config : ConfigBlock
#             GDPopt configuration block
#         iteration : int
#             The current iteration number for the optimization

#         Returns
#         -------
#         bool
#             whether the primal bound is improved
#         float
#             The subproblem's objective value (or None if infeasible)
#         """

#         point_key = tuple(external_var_value)

#         # Check if this subproblem has already been solved in any iteration
#         for prev_iter in range(1, iteration + 1):
#             if prev_iter in self.explored_point_dict and point_key in self.explored_point_dict[prev_iter]:
#                 subproblem_obj_value = self.explored_point_dict[prev_iter][point_key]
#                 config.logger.info(f"Retrieved stored subproblem objective for point {point_key} from iteration {prev_iter}: {subproblem_obj_value}")
#                 return False, subproblem_obj_value
            
#             # Solve the subproblem if it's not stored
#             self.fix_disjunctions_with_external_var(external_var_value)
#             subproblem = self.working_model.clone()
#             TransformationFactory('core.logical_to_linear').apply_to(subproblem)

#         try:
#             with SuppressInfeasibleWarning():
#                 try:
#                     fbbt(subproblem, integer_tol=config.integer_tolerance)
#                     TransformationFactory('contrib.detect_fixed_vars').apply_to(subproblem)
#                     TransformationFactory('contrib.propagate_fixed_vars').apply_to(subproblem)
#                     TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(subproblem, tmp=False, ignore_infeasible=False)
#                     TransformationFactory('gdp.bigm').apply_to(subproblem)
#                 except InfeasibleConstraintException:
#                     config.logger.info(f"Subproblem infeasible for external variable values: {external_var_value}")
#                     self.store_explored_point(external_var_value, config.infinity_output, iteration)
#                     return False, config.infinity_output

#                 # Solve the subproblem
#                 minlp_args = dict(config.minlp_solver_args)
#                 if config.time_limit is not None and config.minlp_solver == 'gams':
#                     elapsed = get_main_elapsed_time(self.timing)
#                     remaining = max(config.time_limit - elapsed, 1)
#                     minlp_args['add_options'] = minlp_args.get('add_options', [])
#                     minlp_args['add_options'].append(f'option reslim={remaining};')

#                 result = SolverFactory(config.minlp_solver).solve(subproblem, tee=False, **minlp_args)

#                 # Check the solver's termination condition
#                 if result.solver.termination_condition in {tc.optimal, tc.feasible, tc.locallyOptimal}:
#                     subproblem_obj_value = value(subproblem.obj)

#                     # Check if the objective value is valid
#                     if subproblem_obj_value is None or subproblem_obj_value != subproblem_obj_value:  # Check for NaN
#                         config.logger.warning(f"Objective value is NaN or None for external variables {external_var_value}.")
#                         subproblem_obj_value = float('nan')

#                     # Store the result in the cumulative dictionary
#                     self.store_explored_point(external_var_value, subproblem_obj_value, iteration)

#                     # Handle the result and check for primal improvement
#                     primal_improved = self._handle_subproblem_result(result, subproblem, external_var_value, config, search_type)
#                     return primal_improved, subproblem_obj_value
#                 else:
#                     config.logger.warning(f"Solver returned a non-optimal status: {result.solver.termination_condition}")
#                     self.store_explored_point(external_var_value, config.infinity_output, iteration)
#                     return False, config.infinity_output
#         except RuntimeError as e:
#             config.logger.warning(f"RuntimeError encountered for external variables {external_var_value}: {str(e)}")
#             self.store_explored_point(external_var_value, None, iteration)
#             return False, None

#     def generate_master_problem(self, config):
#         # Initialize the master problem as a ConcreteModel
#         master_problem = ConcreteModel()

#         # Set up the external variables (these are integer variables)
#         util_block = self.working_model_util_block
#         external_vars = []

#         # Use initial point to set initial values for external variables in the first iteration
#         for i, external_var_info in enumerate(util_block.external_var_info_list):
#             var_name = f'external_var_{i}'
#             initial_value = self.current_point[i]  # Set initial point value for first iteration
#             var = Var(within=NonNegativeIntegers, bounds=(external_var_info.LB, external_var_info.UB), initialize=initial_value)
#             setattr(master_problem, var_name, var)
#             external_vars.append(getattr(master_problem, var_name))

#         # Explicitly assign the external_vars list to the master_problem for future access
#         master_problem.external_vars = external_vars

#         # Create a placeholder for the Benders cuts
#         master_problem.benders_cuts = Constraint(Any)

#         # Define the objective variable 'z' and the objective function
#         master_problem.z = Var(within=Reals, initialize=0)  # Objective variable 'z'

#         # Define the objective function as minimizing 'z'
#         master_problem.obj = Objective(expr=master_problem.z, sense=minimize)

#         return master_problem

#     def solve_master_problem(self, config):
#         """
#         Solve the master problem with the Benders cuts applied.
#         Ensure that we use the correct solver, like Gurobi, for the master problem.
#         """
#         # Ensure that we are using the correct solver for the master problem
#         milp_solver = config.mip_solver if config.mip_solver else "gurobi"

#         # Create a solver object for the MILP solver
#         solver = SolverFactory(milp_solver)
        
#         # Solve the master problem
#         result = solver.solve(self.master_problem, tee=False)

#         # Check if the solver's termination condition is optimal
#         if result.solver.termination_condition != tc.optimal:
#             raise RuntimeError(f"Master problem did not converge to an optimal solution. "
#                             f"Termination condition: {result.solver.termination_condition}")

#         # Retrieve the values of the external variables from the solution
#         external_var_values = [int(value(var)) for var in self.master_problem.external_vars]
        
#         # Retrieve the objective value (z)
#         objective_value = value(self.master_problem.z)

#         # Optionally, print the values for debugging
#         print(f"Solved master problem. Objective value (z): {objective_value}")
#         print(f"External variables: {external_var_values}")

#         # Return the values of the external variables and the objective value
#         return external_var_values, objective_value


#     def generate_benders_cuts(self, config):
#         """Generate Benders cuts based on the stored subproblem solutions.

#         Parameters
#         ----------
#         config : ConfigBlock
#             GDPopt configuration block
#         """
#         # Step 1: Get the current external variable coordinates from the master problem
#         external_var_values = [value(var) for var in self.master_problem.external_vars]
#         point_key = tuple(external_var_values)

#         # Step 2: Check if the current point is infeasible
#         is_infeasible = any(
#             point_key in iter_dict and iter_dict[point_key] == config.infinity_output
#             for iter_dict in self.explored_point_dict.values()
#         )

#         if is_infeasible:
#             # Handle infeasibility by generating a feasibility cut
#             config.logger.info(f"Generating feasibility cut for external variables: {external_var_values}")

#             # Step 3: Set up the cut-generating subproblem for infeasible case
#             infeasible_model = ConcreteModel()

#             # Step 3a: Define variables
#             num_disjunctions = len(external_var_values)
#             infeasible_model.p = Var(range(num_disjunctions), within=Reals, initialize=0.0)
#             infeasible_model.alpha = Var(within=Reals, initialize=0.0)

#             # Step 3b: Define the objective function
#             infeasible_model.objective = Objective(
#                 expr=sum(
#                     infeasible_model.p[i] * external_var_values[i]
#                     for i in range(num_disjunctions)
#                 ) + infeasible_model.alpha,
#                 sense=maximize
#             )

#             # Step 4: Add constraints based on explored points (feasible subproblems)
#             infeasible_model.constraints = ConstraintList()
#             for iter_dict in self.explored_point_dict.values():
#                 for coord, rhs_value in iter_dict.items():
#                     if rhs_value != config.infinity_output:
#                         constraint_expr = (
#                             sum(
#                                 infeasible_model.p[i] * coord[i]
#                                 for i in range(num_disjunctions)
#                             ) + infeasible_model.alpha <= rhs_value
#                         )
#                         infeasible_model.constraints.add(constraint_expr)

#             # Step 5: Solve the cut-generating subproblem
#             solver = SolverFactory('gurobi')
#             solver.options['Method'] = 2  # Set the method to interior point
#             solver.solve(infeasible_model, tee=False)

#             # Step 6: Extract p coefficients and alpha from the solution
#             p_coefficients = [value(infeasible_model.p[i]) for i in range(num_disjunctions)]
#             alpha_value = value(infeasible_model.alpha)

#             # Step 7: Generate feasibility cut
#             benders_cut_expr = sum(
#                 p_coefficients[i] * self.master_problem.external_vars[i]
#                 for i in range(num_disjunctions)
#             ) + alpha_value

#             # Log the feasibility cut for debugging
#             config.logger.info(f"Generated feasibility cut: {benders_cut_expr} <= z")
#             config.logger.info(f"p coefficients: {p_coefficients}, alpha: {alpha_value}")

#             # Step 8: Add the feasibility cut to the master problem
#             if not hasattr(self, 'benders_cuts_dict'):
#                 self.benders_cuts_dict = {}
#             self.master_problem.benders_cuts[point_key] = benders_cut_expr <= self.master_problem.z

#         else:
#             # Handle feasible subproblem case
#             config.logger.info(f"Generating Benders cut for external variables: {external_var_values}")

#             # Step 3: Set up the cut-generating subproblem for feasible case
#             subproblem_model = ConcreteModel()

#             # Step 3a: Define variables
#             num_disjunctions = len(external_var_values)
#             subproblem_model.p = Var(range(num_disjunctions), within=Reals, initialize=0.0)
#             subproblem_model.alpha = Var(within=Reals, initialize=0.0)

#             # Step 3b: Define the objective function
#             subproblem_model.objective = Objective(
#                 expr=sum(
#                     subproblem_model.p[i] * external_var_values[i]
#                     for i in range(num_disjunctions)
#                 ) + subproblem_model.alpha,
#                 sense=maximize
#             )

#             # Step 4: Add constraints based on all explored points (feasible subproblems)
#             subproblem_model.constraints = ConstraintList()
#             for iter_dict in self.explored_point_dict.values():
#                 for coord, rhs_value in iter_dict.items():
#                     if rhs_value != config.infinity_output:
#                         constraint_expr = (
#                             sum(
#                                 subproblem_model.p[i] * coord[i]
#                                 for i in range(num_disjunctions)
#                             ) + subproblem_model.alpha <= rhs_value
#                         )
#                         subproblem_model.constraints.add(constraint_expr)

#             # Step 5: Solve the cut-generating subproblem
#             solver = SolverFactory('gurobi')
#             solver.options['Method'] = 2  # Set the method to interior point
#             solver.solve(subproblem_model, tee=False)

#             # Step 6: Extract p coefficients and alpha from the solution
#             p_coefficients = [value(subproblem_model.p[i]) for i in range(num_disjunctions)]
#             alpha_value = value(subproblem_model.alpha)

#             # Step 7: Generate Benders cut
#             benders_cut_expr = sum(
#                 p_coefficients[i] * self.master_problem.external_vars[i]
#                 for i in range(num_disjunctions)
#             ) + alpha_value

#             # Log the generated cut for debugging
#             config.logger.info(f"Generated Benders cut: {benders_cut_expr} <= z")
#             config.logger.info(f"p coefficients: {p_coefficients}, alpha: {alpha_value}")

#             # Step 8: Add the Benders cut to the master problem
#             if not hasattr(self, 'benders_cuts_dict'):
#                 self.benders_cuts_dict = {}
#             self.master_problem.benders_cuts[point_key] = benders_cut_expr <= self.master_problem.z

#     def refine_benders_cuts(self, config, iteration):
#         """
#         Refine previously generated Benders cuts.
        
#         Parameters
#         ----------
#         config : ConfigBlock
#             GDPopt configuration block.
#         iteration : int
#             The current iteration number for the optimization process.
#         """
#         # Skip refinement if it's the first iteration
#         if iteration == 1:
#             config.logger.info("Skipping cut refinement on iteration 1.")
#             return
        
#         # Start refinement process from iteration 2 onwards
#         config.logger.info(f"Refining Benders cuts in iteration {iteration}.")
        
#         # Placeholder for the actual refinement logic
#         # Example refinement steps might include:
#         # 1. Revisit stored subproblems and their cuts.
#         # 2. Modify cut coefficients or bounds.
#         # 3. Add refined cuts to the master problem.
#         for prev_iter in range(1, iteration):
#             if prev_iter in self.explored_point_dict:
#                 for external_var_values, _ in self.explored_point_dict[prev_iter].items():
#                     # Re-solve the subproblem for the stored external variable values
#                     primal_improved, sub_obj_value = self._solve_GDP_subproblem(
#                         list(external_var_values), 'Cut refinement', config, iteration
#                     )

#                     # Update the Benders cut
#                     self.update_benders_cut(external_var_values, sub_obj_value, config)


#     def store_explored_point(self, external_var_value, subproblem_obj_value, iteration):
#         """Store the explored point and its corresponding objective value by iteration.

#         Parameters
#         ----------
#         external_var_value : list
#             The values of the external variables being evaluated
#         subproblem_obj_value : float
#             The objective value for the subproblem at the given external variable values
#         iteration : int
#             The iteration number for which the explored point is being stored
#         """
#         point_key = tuple(external_var_value)
#         if iteration not in self.explored_point_dict:
#             self.explored_point_dict[iteration] = {}
#         self.explored_point_dict[iteration][point_key] = subproblem_obj_value

#     def update_benders_cut(self, external_var_values, subproblem_obj_value, config):
#         """
#         Update the Benders cut associated with the given external variable values.

#         Parameters
#         ----------
#         external_var_values : list or tuple
#             The external variable values corresponding to the cut to be updated.
#         subproblem_obj_value : float
#             The updated subproblem objective value for the given external variable values.
#         config : ConfigBlock
#             GDPopt configuration block.
#         """
#         point_key = tuple(external_var_values)
#         num_disjunctions = len(external_var_values)

#         # Step 1: Remove the old cut from the master problem using point_key
#         if point_key in self.master_problem.benders_cuts:
#             del self.master_problem.benders_cuts[point_key]
#             config.logger.info(f"Removed old Benders cut for point {point_key}.")
#         else:
#             config.logger.warning(f"No existing Benders cut found for point {point_key} to update.")

#         # Step 2: Set up the cut-generating subproblem
#         # Create a new ConcreteModel for the cut-generating subproblem
#         cut_model = ConcreteModel()

#         # Define variables p and alpha
#         cut_model.p = Var(range(num_disjunctions), within=Reals, initialize=0.0)
#         cut_model.alpha = Var(within=Reals, initialize=0.0)

#         # Define the objective function (maximize the cut)
#         cut_model.objective = Objective(
#             expr=sum(
#                 cut_model.p[i] * external_var_values[i]
#                 for i in range(num_disjunctions)
#             ) + cut_model.alpha,
#             sense=maximize
#         )

#         # Step 3: Add constraints based on all explored points
#         cut_model.constraints = ConstraintList()
#         for iter_dict in self.explored_point_dict.values():
#             for coord, rhs_value in iter_dict.items():
#                 if rhs_value != config.infinity_output:
#                     constraint_expr = (
#                         sum(
#                             cut_model.p[i] * coord[i]
#                             for i in range(num_disjunctions)
#                         ) + cut_model.alpha <= rhs_value
#                     )
#                     cut_model.constraints.add(constraint_expr)

#         # Step 4: Solve the cut-generating subproblem
#         solver = SolverFactory('gurobi')
#         solver.options['Method'] = 2  # Use interior point method
#         result = solver.solve(cut_model, tee=False)

#         # Check if the solver found an optimal solution
#         if result.solver.termination_condition != tc.optimal:
#             config.logger.warning(f"Cut-generating subproblem did not converge to an optimal solution for point {point_key}.")
#             return  # You may decide how to handle this situation

#         # Step 5: Extract p coefficients and alpha from the solution
#         p_coefficients = [value(cut_model.p[i]) for i in range(num_disjunctions)]
#         alpha_value = value(cut_model.alpha)

#         # Step 6: Generate the updated Benders cut expression
#         benders_cut_expr = sum(
#             p_coefficients[i] * self.master_problem.external_vars[i]
#             for i in range(num_disjunctions)
#         ) + alpha_value

#         # Log the updated cut for debugging
#         config.logger.info(f"Generated updated Benders cut for point {point_key}: {benders_cut_expr} <= z")
#         config.logger.info(f"Updated p coefficients: {p_coefficients}, alpha: {alpha_value}")

#         # Step 7: Add the updated cut to the master problem
#         self.master_problem.benders_cuts[point_key] = benders_cut_expr <= self.master_problem.z
#         config.logger.info(f"Added updated Benders cut for point {point_key} to the master problem.")

