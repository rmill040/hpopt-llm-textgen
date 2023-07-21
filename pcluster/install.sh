#!/usr/bin/env bash

cd /opt/shared
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p anaconda
rm https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh

echo 'export PATH="/opt/shared/anaconda/bin:$PATH"' >> ~/.bashrc 
source ~/.bashrc
conda update conda
conda init bash