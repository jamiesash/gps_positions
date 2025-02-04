#!/usr/bin/env python

# --------------------------------------------------------------------------------
# Libraries. 

import argparse
import os
import toml
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from pycurrents.file.binfile_n import BinfileSet
from pycurrents.data import navcalc
from pycurrents.num import interp1 
from pycurrents.adcp.uhdas_cfg_api import get_cfg, find_cfg_files

# --------------------------------------------------------------------------------
# input arguments.

parser = argparse.ArgumentParser()
parser.add_argument("--uhdas_dir", 
                   default = str(Path().resolve()),
                   help="uhdas directory you want to access.",
                   type = str)

args = parser.parse_args()
path_cruise = args.uhdas_dir

# --------------------------------------------------------------------------------
# functions and subroutines

# give a list of rbin files aoutput catinated array.
def arrayrbins(files):
    mat = list()
    for i in files:
        tmp = BinfileSet(str(i))
        mat.append(tmp.array)
    mat = np.vstack(mat)
    return(mat) 

# generates list of rbin files and calls arrayrbins.
def readrbins(pth, sensor, tag):
    tag = "*" + tag + "*.rbin"
    files = sorted(Path(pth+sensor+"/").glob(tag))
    mat = arrayrbins(files)
    #cols = BinfileSet(str(files[0])).columns
    mat = np.array(mat) #, dtype=cols)
    return(mat)

def carvect(primary, sensor, hdg):
    # interpolate onto the primary pos sensor
    lon = interp1(sensor[:,0], sensor[:,2], primary[:,0])
    lat = interp1(sensor[:,0], sensor[:,3], primary[:,0])
    # find the distance in u and v from the primary gps. 
    dlon = lon - primary[:,2]
    dlat = lat - primary[:,3]
    dxdy = navcalc.diffxy_from_difflonlat(dlon , dlat, primary[:,3])
    dxdy = np.array(dxdy)
    # interpolate dxdy onto heading device resolution. 
    dx = interp1(primary[:,0], dxdy[0,:], hdg[:,0])
    dy = interp1(primary[:,0], dxdy[1,:], hdg[:,0])
    dxdy = np.array([dx, dy])
    return(dxdy)

# find the location on the ship
def shiplocal(vects, hdg):
    # Rotates the distance vector by the heading. 
    # Done via a change of basis. 
    location = []
    for i in range(1, len(hdg)):
        tht = hdg[i,1]*(np.pi/180) # to radians
        A = np.array([[np.cos(tht), np.sin(tht)],
                [np.sin(tht)*-1, np.cos(tht)]]).T
        location.append(np.matmul(A, vects[:,i]))
    location = np.vstack(location)
    return(location)

# input is matrix of dimensions [x,2] where x is any number.
# columns are the dx dy.   
def xymean(mat):
    ux = np.nanmean(mat[:, 0])
    uy = np.nanmean(mat[:, 1])
    cent = np.array([ux, uy])
    return(cent)

# --------------------------------------------------------------------------------
# loading data

files = find_cfg_files(path_cruise + "/raw/config/")
if files['sensor'] is None: 
    sys.exit("Error: No /raw/config directory found. Are you in a UHDAS cruise directory?")
proc_file = get_cfg(files['proc'])
sens_file = get_cfg(files['sensor'])
hdg_inst = proc_file["hdg_inst"]
pos_inst = proc_file["pos_inst"]

# remove the adcps, primary pos, and hdg instrument from the sensor list.
gps = sens_file['sensor_keys']
adcps = sens_file['adcp_keys']
adcps.append(pos_inst)
adcps.append(hdg_inst)
for i in adcps:
    try:
        gps.remove(i)
    except ValueError:
        pass

# catinate gps rbin arrays, and list directories that do not have .gps.rbin files. 
badies = list()
sensors = list()
for i in gps:
    if any(File.endswith(".gps.rbin") for File in os.listdir(path_cruise + '/rbin/' + i)):
        sensors.append(readrbins(pth = path_cruise + '/rbin/', sensor = i, tag = 'gps'))
    else:
        badies.append(i)
        print("No gps files in the " + i + " rbins directory.")

# remove gps devises without gps.rbin files. 
for i in badies:
    gps.remove(i)

# reading in the gps coordinates from the rbin. 
prime = readrbins(pth = path_cruise + '/rbin/', sensor = pos_inst, tag = 'gps')
head  = readrbins(pth = path_cruise + '/rbin/', sensor = hdg_inst, tag = 'hdg')

# --------------------------------------------------------------------------------
# calculations
distvec = list()
dxdy = list()
centers = list()
for device in sensors:
    tmp = carvect(primary = prime, sensor = device, hdg = head)
    pos = shiplocal(vects = tmp, hdg = head)
    
    centers.append(xymean(mat = pos))
    distvec.append(pos)
    dxdy.append(tmp)
centers = np.array(centers)

# --------------------------------------------------------------------------------
# plot the centerpoints.

evens = range(2, len(sensors)*2+2, 2)
odds = range(1, len(sensors)*2, 2)
    
fig = plt.figure(figsize=(12, 4*len(sensors)))
for k in range(0, len(sensors)):
    
    diff = dxdy[k]
    local = distvec[k]

    ux = np.nanmean(diff[0,:])
    ox = np.nanstd(diff[0,:])
    uy = np.nanmean(diff[1,:])
    oy = np.nanstd(diff[1,:])
    
    sup = np.nanmax([ux - ox*3.5, ux + ox*3.5, uy - oy*3.5, uy + oy*3.5])
    inf = np.nanmin([ux - ox*3.5, ux + ox*3.5, uy - oy*3.5, uy + oy*3.5])

    ax = fig.add_subplot(len(sensors), 2, odds[k])
    ax.plot(diff[0,:], diff[1,:], "k.", markersize=1, alpha = 0.01)
    ax.set_title("device: " + gps[k] + "pre-rotation", size = 11)
    ax.set_ylabel("dist from " + pos_inst, size = 11)
    ax.set_xlim(inf, sup)
    ax.set_ylim(inf, sup)
    plt.gca().set_aspect(1)
    plt.setp(ax.spines.values(), linewidth = 1.2)

    ux = np.nanmean(local[:,0])
    ox = np.nanstd(local[:,0])
    uy = np.nanmean(local[:, 1])
    oy = np.nanstd(local[:, 1])
    sup = np.nanmax([ox, oy])

    ax = fig.add_subplot(len(sensors), 2, evens[k])
    ax.plot(local[:,0], local[:, 1], "k.", markersize=1, alpha = 0.01)
    ax.set_title("device: " + gps[k] + "post-rotation", size = 11)
    ax.set_xlim(ux - sup*3.5, ux + sup*3.5)
    ax.set_ylim(uy - sup*3.5, uy + sup*3.5)
    plt.gca().set_aspect(1)
    plt.setp(ax.spines.values(), linewidth = 1.2)
plt.show()
# --------------------------------------------------------------------------------

