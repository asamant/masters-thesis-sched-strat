"""
This sample script generates an NTGA XML file that can be
parsed to obtain an NTGA model in UPPAAL.

author: @asamant

"""

import os
import sys
import numpy as np
import pickle

from ControlLoop import ControlLoop
from Network import Network
from ta import *


current_dir = os.path.dirname(os.path.realpath(__file__))

# Needed for defining traffic models
traffic_models_dir = os.path.join(
    os.path.dirname(current_dir), "traffic-models")

if traffic_models_dir not in sys.path:
    sys.path.append(traffic_models_dir)

traffic_file = os.path.join(current_dir, "traffics_k40.dat")
# traffic_file = os.path.join(current_dir, "traffics_k45.dat")

with open(traffic_file, "rb") as f:
    traffics = pickle.load(f)

# Limit early trigger to N steps before
N = 2
for t in traffics:
    t.transition = {(i[0], k): set(j[0] for j in J)
                    for (i, k), J in t.transition.items() if i[0] - k <= N}

# Create TA models from traffic models
ETC_CL1 = ETCTimeTA(traffics[0], clock_name='c')
ETC_CL2 = ETCTimeTA(traffics[1], clock_name='c')

loops = []
loops.append(ControlLoop(ETC_CL1, f'cl1'))
loops.append(ControlLoop(ETC_CL2, f'cl2'))
net = Network(1, 5)
loops.append(net)

for loop in loops:
    loop.template.layout(auto_nails=True)

# Synchronizing actions need to be defined here
sync = ["up", "down", "timeout", "ack", "nack"]
ntga = NTA(*loops, synchronisation=sync)

ntga.template.declaration = f'{ntga.template.declaration}\nint EarNum;\nint EarMax = 4;'

output_path = os.path.join(current_dir, "ntga_model.xml")

# Write our model to xml/{strategy_name}.xml
with open(output_path, 'w') as file:
    file.write(ntga.to_xml())
    print("NTGA XML written!")
