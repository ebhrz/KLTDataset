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
        if ut < int(start) or ut > int(end):
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
    fname = "config/0610_klt3_404.json"
    fname = "config/1109_klt2_294.json"

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

obss_rtk = split_obs(obs,True)
obss_rtk = filter_obs(obss_rtk, config['start_utc'], config['end_utc'])



#visualize

fig, (ax,ax_img) = plt.subplots(2,1,figsize=(8, 12),gridspec_kw={'height_ratios': [2, 1]})
ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_title('Position Visualization (ENU Coordinates)')

ax.set_xlim(-500, 500)  
ax.set_ylim(-500, 500)  

ax.set_aspect('equal', adjustable='datalim')

spp_sol_points = ax.scatter([], [], c='red', label='RTKLIB SPP Solution')   
rtk_sol_points = ax.scatter([], [], c='blue', label='RTKLIB RTK Solution') 
gt_points = ax.scatter([], [], c='green', label='Ground Truth')  # Ground truth green

ax.legend()


spp_sol_positions = []
rtk_sol_positions = []
gt_positions = []

def update_plot(spp_sol_enu, rtk_sol_enu,gt_enu,img):
    spp_sol_positions.append(spp_sol_enu[:2]) 
    rtk_sol_positions.append(rtk_sol_enu[:2])
    gt_positions.append(gt_enu[:2])

    spp_sol_points.set_offsets(spp_sol_positions)
    rtk_sol_points.set_offsets(rtk_sol_positions)
    gt_points.set_offsets(gt_positions)

    ax_img.clear()
    ax_img.imshow(img)
    ax_img.axis('off')

    plt.draw()
    plt.pause(0.05)

rtk_sols_enus = []
spp_sols_enus = []
gts_enu = []

prcopt = prl.prcopt_default
prcopt.mode = prl.PMODE_KINEMA
prcopt.navsys = prl.SYS_CMP|prl.SYS_GPS
prcopt.soltype = 0
prcopt.elmin = 0#15.0*prl.D2R
prcopt.tidecorr = 0
prcopt.posopt[4] = 0
prcopt.tropopt = prl.TROPOPT_SAAS
prcopt.ionoopt = prl.IONOOPT_BRDC
prcopt.sateph = prl.EPHOPT_BRDC
prcopt.rb[0] = sta.pos[0]
prcopt.rb[1] = sta.pos[1]
prcopt.rb[2] = sta.pos[2]
prcopt.modear = 2


for o,o_rtk,l,gtp in zip(obss,obss_rtk,labels,gt):
    rtksol = get_rtklib_pnt(o_rtk,nav,prcopt,"DGNSS")
    sppsol = get_rtklib_pnt(o,nav,prcopt,"SPP")
    sats = satno2name(l[1])
    los = satno2name(l[2])
    nlos = set(sats)-set(los)
    rtk_sol = {'pos':list(rtksol[0].rr)}
    spp_sol = {'pos':list(sppsol[0].rr)}
    rtk_sol_enu = p3d.ecef2enu(rtk_sol['pos'][0],rtk_sol['pos'][1],rtk_sol['pos'][2],*center_pos)
    spp_sol_enu = p3d.ecef2enu(spp_sol['pos'][0],spp_sol['pos'][1],spp_sol['pos'][2],*center_pos)

    gt_enu = p3d.geodetic2enu(gtp[1],gtp[2],gtp[3],*center_pos)
    
    spp_sols_enus.append(spp_sol_enu)
    rtk_sols_enus.append(rtk_sol_enu)
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
    update_plot(spp_sol_enu,rtk_sol_enu,gt_enu,image)
    input()

spp_sols_enus = np.array(spp_sols_enus)
rtk_sols_enus = np.array(rtk_sols_enus)
gts_enus = np.array(gts_enu)
print("2D error of RTKLIB SPP:",np.linalg.norm((spp_sols_enus-gts_enus)[:,:2],axis=1).mean())
print("2D error of RTKLIB RTK:",np.linalg.norm((rtk_sols_enus-gts_enus)[:,:2],axis=1).mean())
print("3D error of RTKLIB SPP:",np.linalg.norm(spp_sols_enus-gts_enus,axis=1).mean())
print("3D error of RTKLIB RTK:",np.linalg.norm(rtk_sols_enus-gts_enus,axis=1).mean())
plt.show()