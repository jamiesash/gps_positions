#!/usr/bin/env python

# --------------------------------------------------------------------------------
# Calculates the relitive location of the gps devices to the primary device. The 
# gps possitions are taken from the gbin directory. Currently every 120 gps 
# postiion is read to speed up the calculation. Ulitmatly the diffeences between
# the gps readings and the primary gps device readings are rotated by the ships
# heading. 
# --------------------------------------------------------------------------------
# Libraries. 

import argparse
import os
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
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
parser.add_argument("--sdate", 
                   default = False,
                   help="Start time for subset. String format as '%Y-%m-%d %H:%M:%S'.",
                   type = str)
parser.add_argument("--edate", 
                   default = False,
                   help="End time for subset. String format as '%Y-%m-%d %H:%M:%S'.",
                   type = str)
args = parser.parse_args()
path_cruise = args.uhdas_dir
sdate = args.sdate
edate = args.edate

# --------------------------------------------------------------------------------
# functions and subroutines

# generates list of rbin files and calls arrayrbins.
def list_files(path, sensor, tag):
    tag = "*" + tag + "*.rbin"
    files = sorted(Path(path+sensor+"/").glob(tag))
    return(files)
    
def read_nav(files, cruise_year, sdate = True, edate = True):
    # read them matfiles. 
    tmp = BinfileSet(files, step = 120)
    # get into numpy's world. 
    mat = np.array(tmp.array)
    days = mat[:,2] 

    # convert to an unambgous datetime format. 
    cruise_year = datetime.strptime(cruise_year, '%Y').year
    since_start = datetime(cruise_year, 1, 1)      # This is the "days since" part
    tmp = list()
    for i in range(0, days.shape[0]):
        # from file_id.variables: origin 2024-01-01 00:00:00
        delta = timedelta(days[i])     # Create a time delta object from the number of days
        offset = since_start + delta    # Add the specified number of days to the start of the year
        tmp.append(offset.strftime('%Y-%m-%d %H:%M:%S'))
    time = np.array(tmp)

    # cut the cruise data by the given string start and stop times. 
    if isinstance(sdate, str):
        time_idx = np.where((time > sdate) & (time < edate))
        mat = mat[time_idx[0],:]

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
    ux = np.nanmedian(mat[:, 0])
    uy = np.nanmedian(mat[:, 1])
    cent = np.array([ux, uy])
    return(cent)

# --------------------------------------------------------------------------------
# upload the data

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
for i in adcps:
    try:
        gps.remove(i)
    except ValueError:
        pass
gps.remove(hdg_inst)

# catinate gps rbin arrays, and list directories that do not have .gps.rbin files. 
badies = list()
sensors = list()
for i in gps:
    if any(File.endswith(".gps.rbin") for File in os.listdir(path_cruise + '/rbin/' + i)):
        files = list_files(path = path_cruise + '/rbin/', sensor = i, tag = 'gps')
        tmp = read_nav(files, cruise_year = '2024', sdate = sdate, edate = edate)
        sensors.append(tmp)
    else:
        badies.append(i)
        print("No gps files in the " + i + " rbins directory.")

# remove gps devises without gps.rbin files. 
for i in badies:
    gps.remove(i) 

# reading in the gps coordinates for the position device. 
files = list_files(path = path_cruise + '/rbin/', sensor = pos_inst, tag = 'gps')
prime = read_nav(files, cruise_year = '2024', sdate = sdate, edate = edate)

# reading in the gps coordinates for heading device. 
files = list_files(path = path_cruise + '/rbin/', sensor = hdg_inst, tag = 'hdg')
head  = read_nav(files, cruise_year='2024', sdate = sdate, edate = edate)

# --------------------------------------------------------------------------------
# calculations

gps_locations = list()
for device in sensors:
    tmp = carvect(primary = prime, sensor = device, hdg = head)
    pos = shiplocal(vects = tmp, hdg = head)
    gps_locations.append(xymean(mat = pos))
gps_locations = np.array(gps_locations)

# --------------------------------------------------------------------------------
# print output

# print a table of the gps lacations.
print("GPS locations relative to the primary gps device.")
print("    x |      y |           gps name")
print("-----------------------------------")
print(f"   hdg |    hdg | {hdg_inst:>18s}") 
for ((x, y), name) in zip(gps_locations, gps):
    print(f"{x:>6.2f} | {y:6.2f} | {name:>18s}")
for sonar in adcps:
    x = proc_file['xducer_dx'][sonar]
    y = proc_file['xducer_dy'][sonar]
    print(f"{x:>6.2f} | {y:6.2f} | {sonar:>18s}")
print("-----------------------------------")
