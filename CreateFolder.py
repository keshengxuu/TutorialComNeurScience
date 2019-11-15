#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:46:12 2019

@author: ksxu
"""

import os
file_name = ['/file1','/file2','/file3']
path = r'/home/ksxu/Documents'
for name in file_name:
    os.mkdir(path+name)

    
    
