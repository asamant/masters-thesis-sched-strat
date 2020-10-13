#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
original author of ETCTime: @ggleizer

The output of the create_traffic_abstraction() function is a list of 
data structures representing the traffic model (PETC, in this case)
for multiple plants

"""
                
import time
import control as ct
import sys, os
import shortuuid
import numpy as np
import matplotlib.pyplot as plt
import pickle

''' etctime libraries '''
import abstraction as ab
import linearetc as etc


# kmax = max number of steps to consider for triggering
def create_traffic_abstractions(kmax1=30, kmax2=30):
    """Create traffic abstractions for two systems and returns list

    Args:
        kmax1 (int, optional): Number of partitions for system 1. Defaults to 30.
        kmax2 (int, optional): Number of partitions for system 2. Defaults to 30.

    Returns:
        list: Traffic model data structures for both plants
    """

    ''' Controllers taken from Mazo, Adzkiya 2016 '''

    ''' Controller 1 '''

    # Plant
    Ap = np.array([[0, 1], [-2, 3]])
    Bp = np.array([0, 1])
    
    Cp = np.eye(2)

    # Controller
    K = np.array([[1, -4]])

    plant = etc.LinearPlant(Ap, Bp, Cp)
    h = 0.01
    rho = 0.08

    controller = etc.LinearController(K, h)

    trig = etc.DirectLyapunovPETC(plant, controller, rho=rho, kmax=kmax1,
                                  h=0.01)

    ''' Build abstraction '''
    t = time.time()
    traffic = ab.TrafficModelPETC(trig,
                                  mu_threshold=0.00,
                                  min_eig_threshold=1e-3,
                                  max_delay_steps=2,
                                  no_costs=True,
                                  reduced_actions=False,
                                  early_trigger_only=False)
    # NOTE: Reduced actions seems to degrade resulting strategy severely
    # traffic.add_level_sets(0.01, 10, 100)
    print('Elapsed: %.2f seconds' % (time.time() - t))

    ''' Controller 2...'''
    
    # Plant
    Ap2 = np.array([[-0.5, 0], [0, 3.5]])
    Bp2 = np.array([1, 1])
    
    Cp2 = np.eye(2)

    # Controller
    K2 = np.array([[1.02, -5.62]])

    plant2 = etc.LinearPlant(Ap2, Bp2, Cp2)
    h = 0.01
    rho2 = 0.08

    controller2 = etc.LinearController(K2, h)

    trig2 = etc.DirectLyapunovPETC(plant2, controller2, rho=rho2, kmax=kmax2,
                                   h=0.01)
    
    '''... and its abstraction'''
    t = time.time()
    traffic2 = ab.TrafficModelPETC(trig2,
                                   mu_threshold=0.00,
                                   min_eig_threshold=1e-3,
                                   max_delay_steps=2,
                                   no_costs=True,
                                   reduced_actions=False,
                                   early_trigger_only=False)
    # NOTE: Reduced actions seems to degrade resulting strategy severely
    # traffic.add_level_sets(0.01, 10, 100)
    print('Elapsed: %.2f seconds' % (time.time() - t))

    traffic_list = [traffic, traffic2]
    return traffic_list


# TEST SCRIPT
if __name__ == "__main__":

    file_dir = os.path.dirname(os.path.abspath(__file__))
    
    k_vals = [40, 45]

    for val in k_vals:

        PFILE = os.path.join(file_dir, f"traffics_k{val}.dat")

        kmax1 = val
        kmax2 = val

        traffic_list = create_traffic_abstractions(kmax1, kmax2)

        with open(PFILE, "wb") as f:
            pickle.dump(traffic_list, f)
            print(f"Successfully saved traffic list to {PFILE}!")
