#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 21:48:03 2021

@author: ksxuphy
"""
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
    
def zoom_outside(srcax, roi, dstax, color="red", linewidth=2, roiKwargs={}, arrowKwargs={}):
    '''Create a zoomed subplot outside the original subplot
    
    srcax: matplotlib.axes
        Source axis where locates the original chart
    dstax: matplotlib.axes
        Destination axis in which the zoomed chart will be plotted
    roi: list
        Region Of Interest is a rectangle defined by [xmin, ymin, xmax, ymax],
        all coordinates are expressed in the coordinate system of data
    roiKwargs: dict (optional)
        Properties for matplotlib.patches.Rectangle given by keywords
    arrowKwargs: dict (optional)
        Properties used to draw a FancyArrowPatch arrow in annotation
    '''
    roiKwargs = dict([("fill", False), ("linestyle", "dashed"),
                      ("color", color), ("linewidth", linewidth)]
                     + list(roiKwargs.items()))
    arrowKwargs = dict([("arrowstyle", "-"),("linestyle", "dashed"), ("color", color),
                        ("linewidth", linewidth)]
                       + list(arrowKwargs.items()))
    
    # draw a rectangle on original chart
    srcax.add_patch(Rectangle([roi[0], roi[1]], roi[2]-roi[0], roi[3]-roi[1], 
                            **roiKwargs))
    #the setting of destination axis
    dstax.tick_params(axis = 'both',
                      colors= color,
                      length=6, 
                      width=2, 
                      bottom=False,  # ticks along the bottom edge are off
                      left=False, # ticks along the left edge are off
                      labelbottom=False,# labels along the bottom edge are off)
                      labelleft=False# labels along the bottom edge are off
                      )
    
    spines_position = ['left','right','top','bottom'] 
    for Sp_po in spines_position:
        dstax.spines[Sp_po].set_color(color) 
        dstax.spines[Sp_po].set_capstyle('butt')
        dstax.spines[Sp_po].set_linewidth(2)
        dstax.spines[Sp_po].set_linestyle('dashed')
    # get coordinates of corners
    srcCorners = [[roi[0], roi[1]], [roi[0], roi[3]],
                  [roi[2], roi[1]], [roi[2], roi[3]]]
    dstCorners = dstax.get_position().corners()
    srcBB = srcax.get_position()
    dstBB = dstax.get_position()
    # find corners to be linked
    if srcBB.max[0] <= dstBB.min[0]: # right side
        if srcBB.min[1] < dstBB.min[1]: # upper
            corners = [1, 2]
        elif srcBB.min[1] == dstBB.min[1]: # middle
            corners = [0, 1]
        else:
            corners = [0, 3] # lower
    elif srcBB.min[0] >= dstBB.max[0]: # left side
        if srcBB.min[1] < dstBB.min[1]:  # upper
           corners = [0, 3]
        elif srcBB.min[1] == dstBB.min[1]: # middle
            corners = [2, 3]
        else:
            corners = [1, 2]  # lower
    elif srcBB.min[0] == dstBB.min[0]: # top side or bottom side
        if srcBB.min[1] < dstBB.min[1]:  # upper
            corners = [0, 2]
        else:
            corners = [1, 3] # lower
    else:
        RuntimeWarning("Cannot find a proper way to link the original chart to "
                       "the zoomed chart! The lines between the region of "
                       "interest and the zoomed chart wiil not be plotted.")
        return
    # plot 2 lines to link the region of interest and the zoomed chart
    for k in range(2):
        srcax.annotate('', xy=srcCorners[corners[k]], xycoords="data",
            xytext=dstCorners[corners[k]], textcoords="figure fraction",
            arrowprops=arrowKwargs)
        
        


# prepare something to plot
x = range(100)
y = [-100, -50, 0, 50, 100] * int(len(x)/5)

#fig=plt.figure(1,figsize=(8,6))
# create a figure
#plt.clf()
_,axes = plt.subplots(3, 3)
plt.subplots_adjust(wspace=0.2, hspace=0.2)

# plot the main chart
axes[1, 1].plot(x, y)

# plot zoomed charts
zoom_outside(srcax=axes[1, 1], roi=[0, 80, 20, 100], dstax=axes[0, 0], color="C1")
zoom_outside(axes[1, 1], [20, -150, 60, 120], axes[1, 0], "C2")
zoom_outside(axes[1, 1], [80, 80, 100, 100], axes[0, 1], "C3")
zoom_outside(axes[1, 1], [0, -20, 20, 20], axes[2, 2], "C4")
zoom_outside(axes[1, 1], [80, -20, 100, 20], axes[1, 2], "C5")
zoom_outside(axes[1, 1], [0, -100, 20, -80], axes[0, 2], "C6")
zoom_outside(axes[1, 1], [40, -100, 60, -80], axes[2, 0], "C7")
zoom_outside(axes[1, 1], [80, -100, 100, -80], axes[2, 1], "C8")    