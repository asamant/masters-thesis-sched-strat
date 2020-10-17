# README

This repository consists of code related to a part of my Master's thesis. In short, the output of this code is a scheduling strategy in the form of a JSON string that can be interpreted in LabVIEW using its JSON API. 

## Overview

The following parts are involved in generating the strategy:

- Traffic model generation: The state space of a control system is partitioned into regions depending on whether the control loops use an [ETC scheme](https://ieeexplore.ieee.org/document/6425820) or a [PETC scheme](https://ieeexplore.ieee.org/document/6310015), and also depending on the partitioning approach followed (see [this](https://ieeexplore.ieee.org/document/8526306) and [this](https://arxiv.org/abs/2003.07642), for instance). In this thesis, we consider the traffic model introduced in [this](https://arxiv.org/abs/2003.07642) paper (a PETC scheme with time-based partitioning). Code from the authors is used with slight modifications to generate traffic models.
- Timed Game Automata model generation: Based on the traffic models and their transition relations, TGA models for the control loops and the network are generated using Python code inspired from [this](https://github.com/pschalkwijk/Python2Uppaal) project and [this](https://launchpad.net/pyuppaal) project. The output is an XML file representing an NTGA model with a parallel composition of all control loop and network TGA models.
- Strategy generation and parsing: [UPPAAL Stratego](https://people.cs.aau.dk/~marius/stratego/) is used to create a non-losing strategy from the UPPAAL model generated in the previous step; this strategy is a text file with a known structure. Only certain snippets in the text are relevant for performing actions and regular expressions are used for parsing those snippets to extract relevant information. This information (a set of tuples) is stored as a JSON file to be loaded in LabVIEW.

Theoretical (and some implementation) details can be found in my [thesis](http://resolver.tudelft.nl/uuid:2dccaa3b-dbff-428e-a5d3-d46ada57504d).

## Project structure

The code associated with each of the stages is housed in the appropriate directories, and a directory `lv_dir` consists of the output generated from `sample_script.py` provided for reference.

```
root
|
|--- traffic-models/
|
|--- tga-models/
|
|--- strategy/
|
|--- lv_dir/
```


All the Python libraries that are required can be loaded via [miniconda](https://docs.conda.io/en/latest/miniconda.html) by using the `conda-environment.yml` file. [This](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) can be used for understanding how to use conda environments.