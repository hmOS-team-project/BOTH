# BOTH: Bayesian Online Learning for Automated Team Coordination in Human-Machine Partnerships
Human-machine teams in industrial settings require efficient Task Scheduling and Allocation (TSA) schemes to optimize productivity and adapt to varying task performance. We propose a novel approach that employs Bayesian online learning to dynamically coordinate tasks based on real-time context, outperforming traditional methods in virtual multi-round scheduling environments.

## File Structure
- ./benchmark/ -- Evaluation code for HybridNet, HetGAT and baseline heuristics
        edfutils.py has some simple class for robot teams
- ./context_bayes_opt/ -- GP based Bayesian optimizer functions
- ./gen/ -- Generate random problem instances
        mrc.py details how the constraints are sampled
- ./env/ -- OpenAI based Environment for Single Round and Multi-Round Scheduling
- ./evolutionary_algorithm.py -- Genetic Algorithm functions for generating schedules using multi-generational optimization           
- ./graph/hetgat.py -- HGA layer implementation
- edfutils.py has some simple class for robot teams
