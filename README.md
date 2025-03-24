# BOTH: Bayesian Online Learning for Automated Team Coordination in Human-Machine Partnerships
Human-machine teams in industrial settings require efficient Task Scheduling and Allocation (TSA) schemes to optimize productivity and adapt to varying task performance. We propose a novel approach that employs Bayesian online learning to dynamically coordinate tasks based on real-time context, outperforming traditional methods in virtual multi-round scheduling environments.

## File Structure
- ./benchmark/ -- Evaluation code for HybridNet, HetGAT and baseline heuristics
        edfutils.py has some simple class for robot teams
- ./context_bayes_opt/ -- GP based Bayesian optimizer functions
- ./gen/ -- Generate random problem instances
        mrc.py details how the constraints are sampled
- ./env/ -- OpenAI based Environment for Single Round and Multi-Round Scheduling         
- ./graph/hetgat.py -- HGA layer implementation
- ./evolutionary_algorithm.py -- Genetic Algorithm functions for generating schedules using multi-generational optimization  
- ./generate_problems.py -- Generate random problem instances with different problem size
- ./gp_scheduler.py -- some basic functions for GP based scheduler
- ./gaussian_process_scheuler.py -- main function for running BOTH

## Environment Setup
Create the environment using the `./benchmark/hybridnet/requirements.yaml` file using:

```bash
conda env create -f requirements.yaml
```

## Generate Task Scheduling and Allocation Problems:
```bash
# Small Problem Size
python generate_problems.py --folder=data/small_problem_set/problems --human=2 --robot=2 --task-min=9 --task-max=11 --start-problem=1 --end-problem=200
# Medium Problem Size
python generate_problems.py --folder=data/medium_problem_set/problems --human=2 --robot=2 --task-min=18 --task-max=22 --start-problem=1 --end-problem=200
# Large Problem Size
python generate_problems.py --folder=data/large_problem_set/problems --human=2 --robot=2 --task-min=38 --task-max=42 --start-problem=1 --end-problem=200
```

## Run BOTH
```
python gaussian_process_scheuler.py
```

