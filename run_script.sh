#!/bin/bash

# Install required Python libraries
pip install transformers[torch]
pip install datasets

# Run the Python script
python clusterRun.py