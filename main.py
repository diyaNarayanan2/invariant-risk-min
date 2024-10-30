# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sem import ChainEquationModel
from models import *

import argparse
import torch
import numpy


# returns a pretty 1d line of all values in vector, eg "[+0.112 -2.333]"
def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"
    # .3f upto 3 decimal places float


def errors(w, w_hat):
    w = w.view(-1)
    w_hat = w_hat.view(-1)

    # declares indices of causal and non_causal elements from the true soln
    i_causal = torch.where(w != 0)[0].view(-1)
    i_noncausal = torch.where(w == 0)[0].view(-1)

    if len(i_causal):
        error_causal = (w[i_causal] - w_hat[i_causal]).pow(2).mean()
        error_causal = error_causal.item()
    else:
        error_causal = 0

    if len(i_noncausal):
        error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2).mean()
        error_noncausal = error_noncausal.item()
    else:
        error_noncausal = 0

    # calculates MSE seperately for both sets of elements and returns
    return error_causal, error_noncausal


def run_experiment(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)

    if args["setup_sem"] == "chain":
        setup_str = "chain_ones={}_hidden={}_hetero={}_scramble={}".format(
            args["setup_ones"],
            args["setup_hidden"],
            args["setup_hetero"],
            args["setup_scramble"],
        )
    elif args["setup_sem"] == "icp":
        setup_str = "sem_icp"
    else:
        raise NotImplementedError

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "ICP": InvariantCausalPrediction,
        "IRM": InvariantRiskMinimization,
    }

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(",")}

    all_sems = []
    all_solutions = []
    all_environments = []

    for rep_i in range(args["n_reps"]):
        print(f"Repitition: {rep_i} starting..:")
        if args["setup_sem"] == "chain":
            sem = ChainEquationModel(
                args["dim"],
                ones=args["setup_ones"],
                hidden=args["setup_hidden"],
                scramble=args["setup_scramble"],
                hetero=args["setup_hetero"],
            )
            print("Chain eq model created.")

            # converts args env_list into a list of float env values
            env_list = [float(e) for e in args["env_list"].split(",")]
            print("final list of environments: ")
            print(env_list)
            # stores the diff chain models made for each environment
            environments = [sem(args["n_samples"], e) for e in env_list]
            print("Corresponding environments made as follows: ")
            print(f"{len(environments)} environments have been made")
                
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)
        # environments stores the data with y poisson sampled

    # zip function jointly iterates values from sems and envs
    for sem, environments in zip(all_sems, all_environments):
        sem_solution, sem_scramble = sem.solution()
        # keeping sem-solution as the true weight of y on x

        solutions = [
            "{} SEM {} {:.5f} {:.5f}".format(setup_str, pretty(sem_solution), 0, 0)
        ]
        # creates a string statement which summarizes soln
        
        for method_name, method_constructor in methods.items():
            print(f"Starting method: {method_name}")
            print(type(method_constructor), method_name)
            method = method_constructor(environments, args)
            method_solution = sem_scramble @ method.solution()
            print("Completed: {method_name}: {type(method)}")

            # diff b/w true soln and method soln
            err_causal, err_noncausal = errors(sem_solution, method_solution)

            solutions.append(
                "{} {} {} {:.5f} {:.5f}".format(
                    setup_str,
                    method_name,
                    pretty(method_solution),
                    err_causal,
                    err_noncausal,
                )
            )
            print(solutions)

        all_solutions += solutions

    return all_solutions
    # each solution has the form:
    # "POS IRM [+0.112 -1.270] 0.011 0.233"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invariant regression")
    # Dimension of x,y,z - square matrices of data
    parser.add_argument("--dim", type=int, default=10)
    # No. of Samples: n in sem call
    parser.add_argument("--n_samples", type=int, default=100)

    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--skip_reps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)  # Negative is random
    parser.add_argument("--print_vectors", type=int, default=1)

    # Training iterations
    parser.add_argument("--n_iterations", type=int, default=10000)

    # Train process learning rate
    parser.add_argument("--lr", type=float, default=1e-3)

    # Verbose
    parser.add_argument("--verbose", type=int, default=1)

    # Methods to test
    parser.add_argument("--methods", type=str, default="ERM,IRM")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--env_list", type=str, default=".2,2.,2.8,1.3,3.")
    # default only 3 envs

    # CHAIN EQ MODEL- ARGS for Setup
    parser.add_argument("--setup_sem", type=str, default="chain")
    parser.add_argument("--setup_ones", type=int, default=1)
    parser.add_argument("--setup_hidden", type=int, default=0)
    parser.add_argument("--setup_hetero", type=int, default=1)
    parser.add_argument("--setup_scramble", type=int, default=1)
    args = dict(vars(parser.parse_args()))

    all_solutions = run_experiment(args)
    print("\n".join(all_solutions))
    # Open the file in write mode ('w')
    with open("synthetic_results4.txt", "a") as f:
        # Write the content to the file
        f.write("\n".join(all_solutions))

    # Optionally, print confirmation that the file has been written
    print("Results have been written to synthetic_results4.txt")
