#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Matthew Levine, CMS, Caltech
"""
import numpy as np

def lds(t0, u0, A):
    dudt = A @ u0
    return dudt
