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

""" Logger function for community_graph.py """

from logging import getLogger
from pyomo.core import Constraint, Objective, Var
from pyomo.common.dependencies import networkx as nx

logger = getLogger('pyomo.contrib.community_detection')


def _event_log(model, model_graph, constraint_set, type_of_graph, with_objective):
    """
    Logs information about the results of the code in generate_model_graph

    This function takes in the same Pyomo model as generate_model_graph and the model_graph generated by
    generate_model_graph (which is a NetworkX graph of nodes and edges based on the Pyomo model). Then, some relevant
    information about the model and model_graph is determined and logged using the logger.

    This function is designed to be called by generate_model_graph.

    Parameters
    ----------
    model: Block
         the Pyomo model or block to be used for community detection
    model_graph: NetworkX graph
        a NetworkX graph with nodes and edges based on the given Pyomo optimization model
    constraint_set: set
        a set of the numbers that correspond to all of the constraints in the given model
    type_of_graph: str
        a string that specifies the type of graph that is created from the model
        'c' creates a graph based on constraint nodes,
        'v' creates a graph based on variable nodes,
        'b' creates a graph based on constraint and variable nodes (bipartite graph).
    with_objective: bool
        a Boolean argument that specifies whether or not the objective function is included in the graph

    Returns
    -------
    This function returns nothing; it simply logs information that is relevant to the user.
    """

    # Collect some information that will be useful for the logger
    all_variables_count = len(
        list(model.component_data_objects(ctype=Var, descend_into=True))
    )

    active_constraints_count = len(
        list(
            model.component_data_objects(
                ctype=Constraint, active=True, descend_into=True
            )
        )
    )
    all_constraints_count = len(
        list(model.component_data_objects(ctype=Constraint, descend_into=True))
    )

    active_objectives_count = len(
        list(
            model.component_data_objects(
                ctype=Objective, active=True, descend_into=True
            )
        )
    )
    all_objectives_count = len(
        list(model.component_data_objects(ctype=Objective, descend_into=True))
    )

    number_of_nodes, number_of_edges = (
        model_graph.number_of_nodes(),
        model_graph.number_of_edges(),
    )

    # Log this information as info
    logger.info("%s variables found in the model" % all_variables_count)

    logger.info("%s constraints found in the model" % all_constraints_count)
    logger.info("%s active constraints found in the model" % active_constraints_count)

    logger.info("%s objective(s) found in the model" % all_objectives_count)
    logger.info("%s active objective(s) found in the model" % active_objectives_count)

    logger.info("%s nodes found in the graph created from the model" % number_of_nodes)
    logger.info("%s edges found in the graph created from the model" % number_of_edges)

    # Log information on connectivity and density (if density is 1 then that means the nodes are connected
    # to every other node)
    if number_of_nodes > 0:
        if nx.is_connected(model_graph):
            logger.info("The graph created from the model is connected.")
            graph_is_connected = True
        else:
            logger.info("The graph created from the model is disconnected.")
            graph_is_connected = False

        if type_of_graph == 'bipartite':
            if graph_is_connected:
                top_nodes, bottom_nodes = nx.bipartite.sets(model_graph)

                # The purpose of the next few lines is to ensure that (1) top_nodes is not an empty set (as this causes
                # errors) and that (2) the node sets are properly identified as either constraints or variables
                if len(top_nodes) == 0:
                    top_nodes, bottom_nodes = bottom_nodes, top_nodes
                if top_nodes == constraint_set:
                    constraint_nodes, variable_nodes = top_nodes, bottom_nodes
                else:
                    constraint_nodes, variable_nodes = bottom_nodes, top_nodes
            else:
                # If the graph is not connected then we must construct the constraint node set
                # and variable node set manually
                constraint_nodes = {
                    node for node in model_graph.nodes() if node in constraint_set
                }
                variable_nodes = set(model_graph) - constraint_nodes

            constraint_density = round(
                nx.bipartite.density(model_graph, constraint_nodes), 2
            )
            variable_density = round(
                nx.bipartite.density(model_graph, variable_nodes), 2
            )

            if (
                constraint_density == variable_density == 1.0
            ):  # If the graph is complete, both will equal 1
                logger.info(
                    "The bipartite graph constructed from the model is complete (graph density equals 1)"
                )
            else:
                logger.info(
                    "For the bipartite graph constructed from the model, the density for constraint nodes is %s"
                    % constraint_density
                )
                logger.info(
                    "For the bipartite graph constructed from the model, the density for variable nodes is %s"
                    % variable_density
                )

        else:
            graph_density = round(nx.density(model_graph), 2)

            if graph_density == 1.0:
                logger.info(
                    "The graph constructed from the model is complete (graph density equals 1)"
                )
            else:
                logger.info(
                    "The graph constructed from the model has a density of %s"
                    % graph_density
                )

    # Given one of the conditionals below, we will log this information as a warning
    if all_variables_count == 0:
        logger.warning("No variables found in the model")

    if all_constraints_count == 0:
        logger.warning("No constraints found in the model")
    elif active_constraints_count == 0:
        logger.warning("No active constraints found in the model")

    if all_objectives_count == 0:
        if with_objective:
            logger.info(
                "Parameter 'with_objective' is True but no objective(s) found in the model"
            )
        else:
            logger.info("No objective(s) found in the model")
    elif active_objectives_count == 0:
        if with_objective:
            logger.info(
                "Parameter 'with_objective' is True but no active objective(s) found in the model"
            )
        else:
            logger.info("No active objective(s) found in the model")

    if number_of_nodes == 0:
        logger.warning(
            "No nodes were created for the graph (based on the model and the given parameters)"
        )
    if number_of_edges == 0:
        logger.warning(
            "No edges were created for the graph (based on the model and the given parameters)"
        )
