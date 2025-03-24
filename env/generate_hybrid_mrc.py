# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 02:45:03 2020

@author: baltundas3

Generate problem instances and store in a folder
"""

import argparse
import random
from mrc_problem import MRCProblem

def generate_and_save(num_tasks, num_robots, num_humans, fname):
    map_width = 3
    prob_deadline = 0.25
    prob_wait_ori = 0.25
    prob_wait = prob_wait_ori / (num_tasks-1)
    
    '''
    Generate a problem and check consistency
    '''
    g = MRCProblem(num_tasks = num_tasks, num_robots = num_robots, num_humans= num_humans,
                   map_width = map_width, prob_deadline=prob_deadline, prob_wait_ori = prob_wait_ori)    
    '''
    Save the problem
    '''
    g.save_to_file(fname)
    print('Problem saved.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generates random instances (instance numbers range from start to end) \
                                    and save in a folder")
    parser.add_argument("num_robots", type=int, help="number of robots")
    parser.add_argument("num_humans", type=int, help="number of humans")
    parser.add_argument("num_tasks_max", type=int, help="max number of tasks in the instance")
    parser.add_argument("num_tasks_min", type=int, help="min number of tasks in the instance")
    parser.add_argument("folder", help="folder path")
    parser.add_argument("start", type=int, help="instance number - start")
    parser.add_argument("end", type=int, help="instance number - end")
    args = parser.parse_args()
    
    '''
    Step 0. Parameter Initialization
    '''
    num_robots = args.num_robots
    num_humans = args.num_humans
    num_tasks_max = args.num_tasks_max
    num_tasks_min = args.num_tasks_min
    
    folder = args.folder+'/'
    start_number = args.start
    end_number = args.end

    '''
    Generate multi instances from start to end
    '''
    for i in range(start_number, end_number+1):
        fname = folder + '%05d' % i
        print('Generating results for %05d' % i)
        #print(fname)
        num_tasks = random.randint(num_tasks_min, num_tasks_max)
        #print(num_tasks)
        generate_and_save(num_tasks, num_robots, num_humans, fname)