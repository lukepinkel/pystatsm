#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 03:11:26 2020

@author: lukepinkel
"""

import matplotlib.pyplot as plt

def maxplot(fig=None, tight=False):
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    if fig is None and tight:
        fig = plt.gcf()
        fig.tight_layout()
    return fig, mng
