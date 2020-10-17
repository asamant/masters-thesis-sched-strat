#!/usr/bin/env python3

import subprocess
import os
import sys

# Package for TGA models
tga_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TGA model definitions needed
if tga_dir not in sys.path:
    sys.path.append(tga_dir)


# location of your uppaal installation
VERIFYTA = "/home/aniket/uppaal64-4.1.20-stratego-7/bin-Linux/verifyta"

file_dir = os.path.dirname(__file__)

"""
Create Strategy
"""


def generate_strategy_from_ntga(ntga_xml_filepath,
                                net, strat_dir=file_dir, strat_name='demo_strat'):

    # Create a path for the strategy
    strat_path = os.path.join(strat_dir, strat_name)

    # create the query is our scheduler objective:
    # control such that we don't reach network.Bad state
    with open(f'{strat_path}.q', 'w') as file:
        file.write(
            f"strategy {strat_name} = control: A[] not ({net.name}{net.index}.Bad)")

    arg_list = [VERIFYTA, '-u', '-s', '--generate-strategy', '2', '--print-strategies',
                strat_dir, ntga_xml_filepath, f'{strat_path}.q']
    print(' '.join(arg_list))

    # Run the query in UPPAAL to generate the strategy file
    verify_ta = subprocess.run(arg_list, stdout=subprocess.PIPE)
    result = verify_ta.stdout.decode('utf-8')

    with open(f'{strat_name}.txt', 'w+') as file:
        file.writelines([f'{strat_name}\n'])
        file.write(result)

    print(result)
    print(f"strategy written to {strat_name}")
    print(f"results written to {strat_name}.txt")
