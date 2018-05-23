#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Author: Aaron Penne
Created: 2018-05-22
Platform: Python 3.6 on macOSX
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



###############################################################################
# Main
    
# Set up directories  # FIXME make this selectable
code_dir = os.path.dirname(__file__)  # Returns full path of this script
data_dir = os.path.join(code_dir, 'data')
output_dir = os.path.join(code_dir, 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    
res = pd.read_csv(os.path.join(data_dir, 'resources.csv'))
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# Get sum of each application request
res['total'] = res['quantity'] * res['price']
res_unique = res.groupby(by='id').sum()
# Map to the training set
df = pd.merge(train, res_unique, how='left', left_on='id', right_on='id')
