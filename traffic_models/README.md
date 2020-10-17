Traffic models README
===

This directory consists of code to generate traffic models and save relevant information (traffic model regions and transition relations).

## Main code

`traffic_model_generator` is used for defining:
- the LTI system (plant + fb controller)
- the triggering criterion
- the number of regions in which to partition
- the type of traffic model to generate

`region_matrix_creator` is used for creating JSON files consisting of information related to the systems' traffic models. In this thesis, the implementation of the NCS is on hardware from National Instruments, and hence LabVIEW is used for programming. But the necessary information (traffic models) being generated using Python and LabVIEW (2016) not supporting dictionary datatypes, some workarounds are needed - and those involve saving traffic model information in JSON files, and additional JSON files for storing metadata. These files are loaded in LabVIEW on the physical setup and used for obtaining the information about the traffic models.

The remaining files are from the ETCTime library by @ggleizer, used for mathematically defining event-triggered control systems. Those files have been appropriately modified to include only relevant parts of the original code.

## Acknowledgements

Sincere thanks to @ggleizer for providing access to the ETCTime library.