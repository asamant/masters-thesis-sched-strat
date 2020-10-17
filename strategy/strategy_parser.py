#!/usr/bin/env python3

import strategy
import os
import re
from itertools import islice
import json

# FIRST THE UTIL FUNCTION DEFINITIONS

# https://stackoverflow.com/questions/22281059/set-object-is-not-json-serializable
# To save sets as JSON


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def extract_limits(condition, cl_low, cl_high):

    # NOTE: UPPAAL doesn't have operators of the kind "> or >=", but only "< or <="

    lower_lim = cl_low
    upper_lim = cl_high

    # First define the regex strings
    regex_string_lower_limit = '.*([0-9]+)([<>]=?|==).*'
    regex_string_upper_limit = '.*([<>]=?|==)([0-9]+).*'

    # Check for upper limit
    match_obj = re.match(regex_string_upper_limit, condition)
    if match_obj:
        oper = match_obj.group(1)
        val = int(match_obj.group(2))

        if oper == "==":
            return (val, val)
        elif oper == "<=":
            upper_lim = val
        elif oper == "<":
            upper_lim = val - 1

        return (lower_lim, upper_lim)

    # Check for lower limit
    match_obj = re.match(regex_string_lower_limit, condition)
    if match_obj:
        val = int(match_obj.group(1))
        oper = match_obj.group(2)

        if oper == "==":
            return (val, val)
        elif oper == "<=":
            lower_lim = val
        elif oper == "<":
            lower_lim = val + 1

        return (lower_lim, upper_lim)

# Quick and dirty code to get names of control loops


def get_control_loops(lines):

    # Regex for finding CL names
    cl_regex = (".* ([A-z0-9]+)\.c==([A-z0-9]+)\.c.*")

    for line in lines:
        match_obj = re.match(cl_regex, line)
        if match_obj:
            return match_obj.groups()

    return []

# For each invariant, compute the clock limits for each control loop
# NOTE: Take the max_early and max_delay values from when the abstractions are generated


def populate_limit_dict_from_condition_tuple(condition_tuple,
                                             control_loops, limit_dict, max_early=2, max_delay=2):

    # get regions
    region_cl1 = int(condition_tuple[0])
    region_cl2 = int(condition_tuple[1])

    # if dictionaries haven't been created yet, create them first...
    if not region_cl1 in limit_dict['cl1']:
        limit_dict['cl1'][region_cl1] = set()
    if not region_cl2 in limit_dict['cl2']:
        limit_dict['cl2'][region_cl2] = set()

    cl1_name = control_loops[0]
    cl2_name = control_loops[1]

    # FIXME: could be made scalable by parametrizing indices based on number of loops
    # NOTE: If the abstractions are generated using isotropic partitioning the limits need
    # to be taken from the CL TGA models based on edge guards
    cl1_min_abs = int(condition_tuple[0]) - max_early
    cl1_max_abs = int(condition_tuple[0]) + max_delay

    cl2_min_abs = int(condition_tuple[1]) - max_early
    cl2_max_abs = int(condition_tuple[1]) + max_delay

    # loop to trigger if conditions are satisfied
    trigger_cl_name = condition_tuple[3]

    cl_to_trigger = 0

    if trigger_cl_name == cl1_name:
        cl_to_trigger = 1
    else:
        cl_to_trigger = 2

    # main part to decipher
    invariant_string = condition_tuple[2]

    # there could be multiple combinations for the same region
    invariants = invariant_string.replace('(', '').replace(')', '').split('||')

    # each invariant deals with precisely one upper and lower limit per CL
    for invariant in invariants:

        # this is for handling cases where the control loop clocks
        #  need to have the same valuation
        equality_requirement = 0

        # this is for handling cases where there is a relation between 
        # the clocks in the triggering condition

        # first loop in the condition eg. in cl1.c-cl2.c, first_loop = 1
        first_loop = 0
        
        # upper-bounded value
        diff_val = 0 

        # start with absolute limits
        cl1_low = cl1_min_abs
        cl1_high = cl1_max_abs

        cl2_low = cl2_min_abs
        cl2_high = cl2_max_abs

        # handle cases one by one
        clock_conditions = invariant.split('&&')
        clock_conditions = [condition.strip()
                            for condition in clock_conditions]

        for condition in clock_conditions:

            # 1. handle cases in which there is a relation between the two clocks
            if '-' in condition:

                # the case in which loop1.c - loop2.c == x needn't be handled. 
                # It is covered in other cases (where loop clocks have exact 
                # valuation matches, e.g. loop1.c == 9)
                if '==' in condition:
                    continue

                reg_exp_first_loop = '([A-z0-9]+)\.c.*'
                match_obj = re.match(reg_exp_first_loop, condition)

                # assuming a match is found
                if not match_obj:
                    print("Strange error. Check regex for clock difference part")
                    continue

                cl_name = match_obj.group(1)
                first_loop = 1 if cl_name == cl1_name else 2

                reg_exp_diff_value = '.*(<|<=)([+-]?[0-9]+).*'
                match_obj = re.match(reg_exp_diff_value, condition)

                # assuming a match is found
                if not match_obj:
                    print("Strange error. Check regex for clock difference part")
                    continue

                oper = match_obj.group(1)
                lim = int(match_obj.group(2))

                if oper == "<=":
                    diff_val = lim

                elif oper == "<":
                    diff_val = lim - 1

                continue

            # 2. handle cases in which the clocks need to be equal
            if cl1_name in condition and cl2_name in condition:
                equality_requirement = 1
                continue

            # 3. handle cases for the individual clock limits
            if cl1_name in condition:
                condition = condition.replace(cl1_name, '')
                lims1 = extract_limits(condition, cl1_low, cl1_high)
                (cl1_low, cl1_high) = lims1

            elif cl2_name in condition:
                condition = condition.replace(cl2_name, '')
                lims2 = extract_limits(condition, cl2_low, cl2_high)
                (cl2_low, cl2_high) = lims2

        limit_set_cl1 = (cl1_low, cl1_high, equality_requirement,
                         first_loop, diff_val, cl_to_trigger)
        limit_set_cl2 = (cl2_low, cl2_high, equality_requirement,
                         first_loop, diff_val, cl_to_trigger)

        # add limit sets to appropriate dictionaries
        limit_dict['cl1'][region_cl1].update({limit_set_cl1})
        limit_dict['cl2'][region_cl2].update({limit_set_cl2})


#################################################################################
# MAIN PROGRAM BELOW

def parse_strategy(strat_path, output_dir):

    # This regex string represents all states from which 
    # an ack! signal is sent, i.e. when triggering must occur
    # Groups: (Region, CL1), (Region, CL2), (clock conditions), (CL to trigger)
    REGEX_STRING = (r'\nState: \( .*\).*cl[A-z0-9]+\.from_region=([0-9]+).*cl[A-z0-9]+\.from_region=([0-9]+).*\n.*'
                    '[\n]*When you are in \((.*)\).*[\n]+([A-z0-9]+)\.Trans.*, from_region := to_region.*}\n')

    with open(strat_path, 'r') as strat:

        # assuming the control loop names show up in the first 5 lines somewhere
        first_5_lines = list(islice(strat, 5))
        raw_strat = strat.read()
        raw_condition_tuples = re.findall(
            REGEX_STRING, raw_strat)  # , re.MULTILINE)

    control_loops = get_control_loops(first_5_lines)


    # dictionary consisting of CLs and their dictionaries for each region
    limit_dict = dict()
    limit_dict['cl1'] = dict()
    limit_dict['cl2'] = dict()

    for condition in raw_condition_tuples:
        populate_limit_dict_from_condition_tuple(
            condition, control_loops, limit_dict)

    # export everthing as JSON files
    for cl in limit_dict:

        cl_dict = limit_dict[cl]

        info_json = dict()
        info_json["regions"] = list(cl_dict.keys())

        for region in cl_dict:
            size_regional_conditions = len(cl_dict[region])
            info_json[region] = size_regional_conditions

        # This is needed to overcome LabVIEW's limitations of not having dictionaries
        with open(os.path.join(output_dir, f"strat_{cl}_info_dict.json"), "w") as f:

            json.dump(info_json, f, default=set_default)
            print(f"Successfully saved {cl}'s strategy info to file")

        # Actual strategy file
        with open(os.path.join(output_dir, f"strat_{cl}_dict.json"), "w") as f:

            json.dump(cl_dict, f, default=set_default)
            print(f"Successfully saved {cl}'s actual strategy to file")

    # so now we have: (regions, clock limits, CL to trigger) 
    # in which if the values from the scheduler lie, trigger
