import json
import sys
import pyrtklib as prl
from rtk_util import *
import pickle
import pymap3d as p3d
import matplotlib.pyplot as plt
import cv2

center_pos = [22.33051516,114.18075434,0]

def obs2utc(obstime):
    return obstime.time+obstime.sec-18

def filter_obs(obss,start,end):
    new_obss = []
    for o in obss:
        ut = obs2utc(o.data[0].time)
        if ut < start or ut > end:
            continue
        new_obss.append(o)
    return new_obss

def satno2name(sats):
    name = prl.Arr1Dchar(4)
    if not isinstance(sats,list):
        prl.satno2id(sats+1,name)
        return name.ptr
    names = []
    for i in sats:
        prl.satno2id(i+1,name)
        names.append(name.ptr)
    return names

try:
    fname = sys.argv[1]
except IndexError:
    # print('Usage: python visualize.py <filename>')
    # sys.exit(1)
    fname = "config/1109_klt1_421.json"

with open(fname) as f:
    config = json.load(f)

with open(config['label'], 'rb') as f:
    labels = pickle.load(f)

gt = np.loadtxt(config['gt'], delimiter=',')

obs = prl.obs_t()
nav = prl.nav_t()
sta = prl.sta_t()

prl.readrnx(config['files'][0],1,"",obs,nav,sta)
prl.readrnx(config['files'][1],2,"",obs,nav,sta)

prl.sortobs(obs)
obss = split_obs(obs)
obss = filter_obs(obss, config['start_utc'], config['end_utc'])
assert len(obss) == len(labels) and len(obss) == len(gt), f"Length mismatch: {len(obss)} vs {len(labels)} vs {len(gt)}, the config file and the label file may not match"



#visualize

fig, (ax,ax_img) = plt.subplots(2,1,figsize=(8, 12),gridspec_kw={'height_ratios': [2, 1]})
ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_title('Position Visualization (ENU Coordinates)')

ax.set_xlim(-500, 500)  
ax.set_ylim(-500, 500)  

ax.set_aspect('equal', adjustable='datalim')

sol_points = ax.scatter([], [], c='red', label='SPP Solution')    # WLS blue
nlos_sol_points = ax.scatter([], [], c='blue', label='NLOS Down-weight Solution')  # NLOS-down-weight red
gt_points = ax.scatter([], [], c='green', label='Ground Truth')  # Ground truth green

ax.legend()


sol_positions = []
nlos_sol_positions = []
gt_positions = []

def update_plot(sol_enu, nlos_sol_enu,gt_enu,img):
    sol_positions.append(sol_enu[:2]) 
    nlos_sol_positions.append(nlos_sol_enu[:2])
    gt_positions.append(gt_enu[:2])

    sol_points.set_offsets(sol_positions)
    nlos_sol_points.set_offsets(nlos_sol_positions)
    gt_points.set_offsets(gt_positions)

    ax_img.clear()
    ax_img.imshow(img)
    ax_img.axis('off')

    plt.draw()
    plt.pause(0.05)

sols_enu = []
nlos_sols_enu = []
gts_enu = []

for o,l,gtp in zip(obss,labels,gt):
    sats = satno2name(l[1])
    los = satno2name(l[2])
    nlos = set(sats)-set(los)
    sol = get_wls_pnt_pos(o,nav,SYS=['G','C'])
    nlos_sol = get_nlos_wls_pnt_pos(o,nav,nlos=nlos,SYS=['G','C'])
    sol_enu = p3d.ecef2enu(sol['pos'][0],sol['pos'][1],sol['pos'][2],*center_pos)
    nlos_sol_enu = p3d.ecef2enu(nlos_sol['pos'][0],nlos_sol['pos'][1],nlos_sol['pos'][2],*center_pos)
    gt_enu = p3d.geodetic2enu(gtp[1],gtp[2],gtp[3],*center_pos)
    
    sols_enu.append(sol_enu)
    nlos_sols_enu.append(nlos_sol_enu)
    gts_enu.append(gt_enu)
    image = l[3][:,:,::-1].copy()
    for s in l[4]:
        image_part = l[4][s]
        sname = satno2name(s)
        x,y = int(image_part[1]/3.46875),int(image_part[2]/3.46875)
        if sname in los:
            image = cv2.circle(image,(x,y),5,(0,255,0),-1)
        elif sname in nlos:
            image = cv2.circle(image,(x,y),5,(255,0,0),-1)
    update_plot(sol_enu,nlos_sol_enu,gt_enu,image)

sols_enus = np.array(sols_enu)
nlos_sols_enus = np.array(nlos_sols_enu)
gts_enus = np.array(gts_enu)
print("error of sols:",np.linalg.norm(sols_enus-gts_enus,axis=1).mean())
print("error of nlos sols:",np.linalg.norm(nlos_sols_enus-gts_enus,axis=1).mean())

plt.show()