#!/bin/bash

/usr/local/cuda/bin/nvcc -g -G reforce.cu interaction_map.cu cell_list.cu utilities.cu log.cu events.cu  -o ops_descriptors.o 
