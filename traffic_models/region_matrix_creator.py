#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
original author of ETCTime: ggleizer

script to write matrices to a file for determining 
the region the current state of the plant is in

LOGIC BEHIND THE ENTIRE SCRIPT:

In G. Gleizer's IFAC 2020 paper, a bunch of QCQP problems 
is solved to determine the current region of the plant's 
state.
This is represented by determining if x.T @ Q @ x <= 0 
for all of a region's Q matrices (each sample step has 
its Q matrix)

This script stores all of the Q matrices for each region 
in a dictionary which can be iterated over, the regions 
sorted in ascending order.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

''' etctime libraries '''
import traffic_models.abstraction as ab
import traffic_models.linearetc as etc


def generate_region_matrices_dicts(traffic):


    # create a dictionary of all regions in the traffic model.
    region_dict = dict()

    # its corresponding JSON-compatible version
    region_dict_json = dict()
    
    # Traffic "region dictionary". Needed for LabVIEW usage 
    # (LV's JSON parsing is quite primitive).
    #
    # The element "regions" contains a list of all regions.
    # The rest is a dictionary of size of the Q matrix array 
    # per region. 
    # 
    # For instance, if there are three regions, 
    # say, 4, 6, 8, this dict would look like:
    # {"regions": "4, 6, 8", "4": "1", "6" : "1", "8" : "3"}.
    region_size_dict_json = dict()

    regions = sorted(traffic.regions)
    region_size_dict_json["regions"] = \
                [region[0] for region in regions]

    for k in regions:

        region_dict[k] = list() 

        # Regions are tuples with one element.
        region_dict_json[k[0]] = list() 
        
        # for each region add all the associated Q matrices
        Q_form_set = traffic._add_constraints_for_region_i(k, 
                                                    set())
        
        for Q_form in Q_form_set:
            region_dict[k].append(Q_form.Q)
            region_dict_json[k[0]].append(Q_form.Q.tolist())
        # endfor

        region_size_dict_json[k[0]] = len(Q_form_set)
    # endfor

    return region_dict, region_dict_json, region_size_dict_json


def create_region_matrix_JSON_files(traffic, output_path='', filename_suffix="1"):

    # First generate the dictionaries
    _, region_dict_json, region_size_dict_json = \
        generate_region_matrices_dicts(traffic)

    # Then dump in JSON files   
    with open(os.path.join(output_path, 
                f"traffic_{filename_suffix}_dict.json"), "w") as f:
        json.dump(region_dict_json, f, sort_keys=True)
        print(f"Successfully saved traffic_{filename_suffix}_dict.json!")
        
    with open(os.path.join(output_path, 
                f"traffic_{filename_suffix}_size_dict.json"), "w") as f:
        json.dump(region_size_dict_json, f)
        print(f"Successfully saved traffic_{filename_suffix}_size_dict.json!")


if __name__ == '__main__':

    # first load the traffic files
    dir_path = os.path.dirname(os.path.realpath(__file__))
    traffics = os.path.join(dir_path, 'traffics_k40.dat')
    traffics2 = os.path.join(dir_path, 'traffics_k45.dat')    
    
    with open(traffics, "rb") as f:
        traffics_k40 = pickle.load(f)

    with open(traffics2, "rb") as f:
        traffics_k45 = pickle.load(f)

    # assuming both traffic lists have the same length 
    for i in range(0, len(traffics_k40)):

        # Create JSON files for each traffic model
        create_region_matrix_JSON_files(traffics_k40[i], f"k40_{i}")
        create_region_matrix_JSON_files(traffics_k45[i], f"k45_{i}")