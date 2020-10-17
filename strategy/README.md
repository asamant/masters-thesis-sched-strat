# Strategy generation and parsing README

This directory consists of code related to generating UPPAAL strategies and parsing them. The output of the strategy generation process is a verbose text file which consists of a set of "if-then" conditions that tell what a "game player" (i.e. the scheduler) should do (i.e. take a controllable edge or wait) based on the current clock conditions. This file is provided as input to the strategy parser for having valuable information extracted.

The output of the parsing process is a JSON string consisting of all the clock conditions and the output of the strategy (i.e. plant index to trigger).

The parser generates a JSON consisting of info like:

- State space regions of the plants,
- Clock valuations, and,
- plant to trigger

What information should be added to the JSON file depends on the generated strategy. Most "if-then" blocks are just for *waiting* and hence are redundant; we include only those blocks which involve an *ack* edge - i.e. one that involves trasmitting control action data over the network.

The JSON file describing the strategy is loaded in LabVIEW on the machine running the scheduler, to which the strategy's information about regions and clock valuations is passed - based on this, the scheduler decides which plant to trigger, and thus, when to send control actions over the physical CAN network.