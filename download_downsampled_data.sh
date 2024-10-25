#!/bin/bash

# download and extract the CrossDocked2020 molecular structures without saving a copy of the archive
mkdir -p CrossDocked2020 && \
echo "Downloading and extracting downsampled CrossDocked2020 7.3G..." && \
wget -qO- http://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3.tgz | pv | tar -C CrossDocked2020 -xzf - && \
echo "Downloading and extracting downsampled CrossDocked2020 types 21M..." && \
wget -qO- http://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3_types.tgz | pv | tar -xzvf - && \
echo "Extraction complete."



