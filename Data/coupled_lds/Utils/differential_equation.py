#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Matthew Levine, CMS, Caltech
"""
import numpy as np

def f_ode(t0, u0, eps=0.1, A=np.array([[0, 5],[-5,0]])):
    # small oscillations around a bigger oscillation!
    x = u0[:A.shape[0]]
    y = u0[A.shape[0]:]
    dx = eps*A @ x + (1/eps)*y
    dy = (1/eps) * A @ y
    du = np.hstack((dx, dy))
    return du

# def f_ode(t0, u0, eps=0.5, A=np.array([[0, 5],[-5,0]])):
#     # looks like a nice non-linear oscillator
#     x = u0[:A.shape[0]]
#     y = u0[A.shape[0]:]
#     dx = eps*A @ x + (1/eps)*y
#     dy = (1/eps) * A @ y
#     du = np.hstack((dx, dy))
#     return du

# def f_ode(t0, u0, eps=2.5, A=np.array([[0, 1],[-1,0]])):
#     x = u0[:A.shape[0]]
#     y = u0[A.shape[0]:]
#     dx = eps*A @ x + (5/eps)*y
#     dy = (1/eps) * A @ y
#     du = np.hstack((dx, dy))
#     return du
