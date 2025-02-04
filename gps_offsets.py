"""
develop a tool to calculate alongship and athwartship offsets between GPS(s)
"""

# to read the rbins
from pycurrents.file.binfile_n import BinfileSet  # concatenated rbins

# to grid onto a common time base (in Python, so very fast)
from pycurrents.num import interp1           # interpolation

# to calculate meters difference in a lot or lat difference
# (it's in there somewhere - get familiar with all the pieces in here.
# they're very useful)
from pycurrents.data import navcalc # uv_from_txy, unwrap_lon, unwrap ...

# to do mean, stddev (also in Cython, I think)
from pycurrents.num import Stats            # mean,std,med (masked)

import os
import toml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [6,6]
plt.rcParams['figure.dpi'] = 300

# define where the folder and cruise
# -----only need to change these two -----
path_main = '/home/longren/Desktop/projects/alongship-athwartship-offsets-GPS'
cruise_name = 'SE_24_02'
#cruise_name = 'HLY22TD'

# define the paths to the desired cruise and data
path_cruise = path_main + '/{}/'.format(cruise_name)
path_to_rbindir = path_cruise + 'rbin/'

# here is every gps for that cruise ignore the gyro
gps_devices = [x for x in os.listdir(path_to_rbindir) if x[:4] != 'gyro']
gps_tups = [(x, 'gps') for x in gps_devices]

# from the proc file, get the origin, or main, gps device & the heading device
if os.path.isfile(path_cruise + 'raw/config/{}_proc.toml'.format(cruise_name)):  # .toml
    proc_cfg = toml.load(path_cruise + 'raw/config/{}_proc.toml'.format(cruise_name))
    pos_origin = (proc_cfg['pos_inst'], proc_cfg['pos_msg'])
    heading_tup = (proc_cfg['hdg_inst'], proc_cfg['hdg_msg'])
elif os.path.isfile(path_cruise + 'raw/config/{}_proc.py'.format(cruise_name)):  # .py
    proc_cfg = open(path_cruise + 'raw/config/{}_proc.py'.format(cruise_name), 'r')
    for line in proc_cfg:
        if line[:9] == 'pos_inst ':
            pos_inst = line.split('\'')[1]
        if line[:8] == 'pos_msg ':
            pos_msg = line.split('\'')[1]
        if line[:9] == 'hdg_inst ':
            hdg_inst = line.split('\'')[1]
        if line[:8] == 'hdg_msg ':
            hdg_msg = line.split('\'')[1]
    pos_origin = (pos_inst, pos_msg)
    heading_tup = (hdg_inst, hdg_msg)
else:
    raise NotImplementedError

# define which parts of the cruise to use
filebase_list = [x.name.split('.')[0] for x in list(Path(path_to_rbindir + heading_tup[0]).glob('*.rbin'))]

### ----- example rotation into heading -----
dx = np.random.randint(-9, 9)
dy = np.random.randint(-9, 9)
heading = np.random.randint(0, 360)  # deg

def rotateOffsets(dx, dy, heading):
    heading_rad = (heading/360)*2*np.pi  # rad
    gps_direction = np.arctan2(dx, dy)  # rad
    magnitude = (dx**2 + dy**2)**0.5
    gps_direction_rotated = gps_direction+heading_rad
    dx_rotated = np.sin(gps_direction_rotated)*magnitude
    dy_rotated = np.cos(gps_direction_rotated)*magnitude
    return dx_rotated, dy_rotated

dx_rotated, dy_rotated = rotateOffsets(dx, dy, heading)
## copied outside of function
heading_rad = (heading/360)*2*np.pi  # rad
gps_direction = np.arctan2(dx, dy)  # rad
magnitude = (dx**2 + dy**2)**0.5
gps_direction_rotated = gps_direction+abs(heading_rad)
##
plt.figure()
plt.scatter(dx, dy, c='r')
plt.annotate('hdg', xy=(0, 0), ha='center', va='center',
             xytext=(np.sin(heading_rad)*magnitude*0.7, 
                     np.cos(heading_rad)*magnitude*0.7), 
             arrowprops=dict(arrowstyle='<-', lw=2))
plt.plot([0,np.sin(gps_direction)*magnitude], 
         [0,np.cos(gps_direction)*magnitude], ls='-', c='r', label='before')
plt.plot([0,dx_rotated], 
         [0,dy_rotated], ls='-', c='g', label='after')
plt.scatter(dx_rotated, dy_rotated, c='g')
circle = np.linspace(0, 2*np.pi, 100)
plt.plot(np.sin(circle)*magnitude,
         np.cos(circle)*magnitude,
         ls='--', c='k')
rotation_path = np.linspace(gps_direction, gps_direction_rotated, 100)
plt.plot(np.sin(rotation_path)*magnitude,
         np.cos(rotation_path)*magnitude,
         ls='-', lw=3, c='k')
plt.annotate('north', xy=(0, 0), ha='center', 
             xytext=(0, 0.3*magnitude), arrowprops=dict(arrowstyle='<-'))
lim = magnitude * 1.25
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
plt.legend(bbox_to_anchor=(0.8, 1.0))
plt.axis('square')
plt.grid()
plt.title('known variables: gdx={}, gdy={}, hdg={}deg'.format(dx, dy, heading))
plt.savefig(path_main + '/' + cruise_name + '-figures/gps_offset_example_rotation.png')
plt.close()
### ----------

def find_offsets(origin_tup, compare_tup, heading_tup, path_to_rbindir, filebase_list):
    '''
    origin_tup is (pos_inst, pos_msg) from proc_cfg.toml
    compare_tup is an (instrument, message) tuple for some other gps
    heading_tup is an (instrument, message) tuple for the heading device to use
    path_to_rbin is the path to the rbin directory (then use origin_tup and compare_tup to build the paths)
    filebase_list is like ['se2024_068_00000',
                           'se2024_068_07200',
                           'se2024_068_14333',]
    '''
    
    filebases = filebase_list[:]
    getPaths = lambda tup: np.sort([p for p in list(Path(path_to_rbindir+tup[0]).glob('*'+tup[1]+'.rbin')) \
                                    if p.name.split('.')[0] in filebases])
    origin_paths = getPaths(origin_tup)
    compare_paths = getPaths(compare_tup)
    heading_paths = getPaths(heading_tup)

    origin_rbins = BinfileSet(origin_paths)
    compare_rbins = BinfileSet(compare_paths)
    heading_rbins = BinfileSet(heading_paths)

    dday_origin = origin_rbins.u_dday
    compare_lat = interp1(x_old=compare_rbins.u_dday, y_old=compare_rbins.lat, 
                          x_new=dday_origin)
    compare_lon = interp1(x_old=compare_rbins.u_dday, y_old=compare_rbins.lon, 
                          x_new=dday_origin)
    heading = interp1(x_old=heading_rbins.u_dday, y_old=heading_rbins.heading, 
                      x_new=dday_origin)
    heading *= -1  # THIS FIXES WEIRDNESS

    gdx, gdy = navcalc.diffxy_from_difflonlat(dlon=compare_lon - origin_rbins.lon,
                                              dlat=compare_lat - origin_rbins.lat,
                                              alat=origin_rbins.lat)

    gdx_rotated, gdy_rotated = rotateOffsets(gdx, gdy, heading)
    #gdx_rotated, gdy_rotated = gdx, gdy  # TEST: NO ROTATION

    Sgdx, Sgdy = Stats(gdx_rotated), Stats(gdy_rotated)
    port_offset, fwd_offset, stats = Sgdx.mean, Sgdy.mean, None

    """     fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(dday_origin, gdx_rotated, label='gdx')
    ax1.plot(dday_origin, gdy_rotated, label='gdy')
    ax2.plot(dday_origin, heading, label='hdg', c='k')
    ax1.legend(bbox_to_anchor=(0.2, 1.15))
    ax2.legend(bbox_to_anchor=(1.0, 1.1))
    ax1.set_ylabel('diffgxy [m]')
    ax2.set_ylabel('heading (deg)')
    ax1.set_xlabel('dday')
    ax1.set_title('origin: {}\ncompare: {}'.format(origin_tup[0], compare_tup[0]))
    plt.savefig(path_main + '/' cruise_name + '-/figures/gps_offsets_temporal-{}.png'.format(compare_tup[0]))
    fig.clear() """

    plt.figure()
    plt.scatter(gdx_rotated, gdy_rotated, s=0.0001, zorder=1)
    plt.scatter(0, 0, s=25, c='k', label=origin_tup[0], zorder=2)
    plt.scatter(port_offset, fwd_offset, s=25, c='w', edgecolors='k', label=compare_tup[0], zorder=3)
    lim = np.nanmax([np.nanmax(gdx_rotated), np.nanmax(gdy_rotated)]) * 1.5
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    plt.legend()
    plt.grid()
    plt.xlabel('gdx [m]')
    plt.ylabel('gdy [m]')
    plt.title('gps offset')
    plt.savefig(path_main + '/' + cruise_name + '-figures/gps_offsets_spatial-{}.png'.format(compare_tup[0]))
    
    return port_offset, fwd_offset, stats, gdy_rotated, gdx_rotated


# loop over each gps
gps_offsets = {}
for gps_tup in gps_tups:
    if gps_tup == pos_origin:
        continue
    port_offset, fwd_offset, stats, gdy_rotated, gdx_rotated = find_offsets(origin_tup=pos_origin,
                                                  compare_tup=gps_tup,
                                                  heading_tup=heading_tup,
                                                  path_to_rbindir=path_to_rbindir,
                                                  filebase_list=filebase_list)
    print(gps_tup, 'port_offset: ', format(port_offset, '.2f'), 'm')
    print(gps_tup, 'fwd_offset: ', format(fwd_offset, '.2f'), 'm')
    print(gps_tup, 'stats: ', stats)
    gps_offsets[gps_tup[0]] = {}
    gps_offsets[gps_tup[0]]['port_offset'] = port_offset
    gps_offsets[gps_tup[0]]['fwd_offset'] = fwd_offset
    gps_offsets[gps_tup[0]]['stats'] = stats
    gps_offsets[gps_tup[0]]['gdy_rotated'] = gdy_rotated
    gps_offsets[gps_tup[0]]['gdx_rotated'] = gdx_rotated
    

# plot the gps offsets
plt.figure()
plt.scatter(0, 0, s=5, c='k', label=pos_origin[0], zorder=3)
for gps_tup in gps_tups:
    if gps_tup == pos_origin:
        continue
    port_offset = gps_offsets[gps_tup[0]]['port_offset']
    fwd_offset = gps_offsets[gps_tup[0]]['fwd_offset']
    plt.scatter(port_offset, fwd_offset, s=25, edgecolors='k', label=gps_tup[0], zorder=2)
    gdy_rotated = gps_offsets[gps_tup[0]]['gdy_rotated']
    gdx_rotated = gps_offsets[gps_tup[0]]['gdx_rotated']
    #plt.scatter(gdx_rotated, gdy_rotated, s=0.00005, c='k', zorder=1)
port_lim = np.max([abs(gps_offsets[x]['port_offset']) for x in gps_offsets.keys()])*1.1
fwd_lim = np.max([abs(gps_offsets[x]['fwd_offset']) for x in gps_offsets.keys()])*1.1
#port_lim = np.nanmax([abs(gps_offsets[x]['gdx_rotated']) for x in gps_offsets.keys()])*1.1
#fwd_lim = np.nanmax([abs(gps_offsets[x]['gdy_rotated']) for x in gps_offsets.keys()])*1.1
plt.xlim([-port_lim,port_lim])
plt.ylim([-fwd_lim,fwd_lim])
plt.legend()
plt.grid()
plt.xlabel('gdx [m]')
plt.ylabel('gdy [m]')
plt.title('gps offset')
plt.savefig(path_main + '/' + cruise_name + '-figures/gps_offsets-combined.png')
