#!/bin/bash

# download and extract the CrossDocked2020 molecular structures
wget http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz && \
wget http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3_types.tgz && \
mkdir CrossDocked2020 && \
tar -xzvf CrossDocked2020_v1.3_types.tgz && \
tar -C CrossDocked2020 -xzf CrossDocked2020_v1.3.tgz

