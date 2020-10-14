TGA Models README
===

This directory consists of code for generating TGA models of the control loops and the network, and generating a single NTGA composed of these models. The files in the `ta` package are taken from [this repository](https://github.com/pschalkwijk/Python2Uppaal) with necessary changes. `ControlLoop.py` and `Network.py` contain the TGA model definitions from my thesis.

A sample script is provided to demonstrate the loading of a traffic model file and creation of the corresponding NTGA model thereafter. The final XML file can be verified by loading via UPPAAL.