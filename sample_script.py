#!/usr/bin/env python3

import shortuuid
import os
import pickle

# Starting with a number sometimes causes Uppaal to crash, so we use letters only
shortuuid.set_alphabet("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")

run = shortuuid.uuid()[:4]

# Directory in which all data to be loaded in LabVIEW is stored
# 1. Region matrix files
# 2. Strategy files 
lv_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lv_dir')

"""
Abstractions
"""

from traffic_models import abstraction
import traffic_models.traffic_model_generator as tmg
import traffic_models.region_matrix_creator as rmc

# First create the traffic models - 
# here we specify the maximum number of time steps to 
# elapse before "naturally" triggering (see Gleizer et. al 2020) 
kmax1 = 40
kmax2 = 45

traffic_list = tmg.create_traffic_abstractions(kmax1, kmax2)

for i in range(0, len(traffic_list)):
    rmc.create_region_matrix_JSON_files(traffic_list[i], lv_data_path, f"k40_{i}")

# Limit early trigger to N steps before
N = 2
for t in traffic_list:
    t.transition = {(i[0], k): set(j[0] for j in J)
                    for (i, k), J in t.transition.items() if i[0] - k <= N}

from tga_models.ta import abstractions as ab
from tga_models import ControlLoop as cl
from tga_models import Network as nw
from tga_models.ta import timedautomata as ta

# Create TA models from traffic models
ETC_CL1 = ab.ETCTimeTA(traffic_list[0], clock_name='c')
ETC_CL2 = ab.ETCTimeTA(traffic_list[1], clock_name='c')

loops = []
loops.append(cl.ControlLoop(ETC_CL1, f'cl1'))
loops.append(cl.ControlLoop(ETC_CL2, f'cl2'))
net = nw.Network(1, 5, index=run)
loops.append(net)

for loop in loops:
    loop.template.layout(auto_nails=True)

# Synchronizing actions need to be defined here
sync = ["up", "down", "timeout", "ack", "nack"]
ntga = ta.NTA(*loops, synchronisation=sync)

ntga.template.declaration = f'{ntga.template.declaration}\nint EarNum;\nint EarMax = 4;'

output_path = os.path.join(lv_data_path, "ntga_model.xml")

# Write our model xml file
with open(output_path, 'w') as file:
    file.write(ntga.to_xml())
    print("NTGA XML written!")

from strategy import strategy_generator as sg
from strategy import strategy_parser as sp

strat_name = f"strat_{run}"
strat_filepath = os.path.join(lv_data_path, strat_name)

# generate a strategy file
sg.generate_strategy_from_ntga(output_path, net, lv_data_path, strat_name)

# parse the strategy file and create JSON files to be loaded in LabVIEW
strat_filepath = os.path.join(lv_data_path, strat_name)
sp.parse_strategy(strat_filepath, lv_data_path)