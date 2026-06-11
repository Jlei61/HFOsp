"""DIAGNOSTIC: single-end kick + endpoint_centroid_axis (sparse-friendly, Increment-1) on
the 16-contact montage where onset_front_axis failed. Option B validation."""
import sys, os, numpy as np
ENG = os.path.join("results","topic4_sef_hfo","lif_snn","engine"); sys.path.insert(0, ENG)
from params import Params, compute_nu_theta
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,
    attach_geometry, endpoint_centroid_axis, axis_angle_error_deg, direction_readability)
from src.sef_hfo_snn_adapter import snn_event_envelope, event_window_for_run

L, DENSITY, T, DT, DRIVE = 3.0, 1000.0, 220.0, 0.1, 0.6
CENTER = np.array([L/2, L/2]); PITCH = 0.45
def montage():
    return merge_montages([build_shaft(np.deg2rad(10.0),PITCH,8,tuple(CENTER),"A"),
                           build_shaft(np.deg2rad(100.0),PITCH,8,tuple(CENTER),"B")])
def end_at(theta_deg, frac=0.6):
    return CENTER + frac*(L/2)*np.array([np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))])
def build(theta_deg, AR, seed=1):
    p = Params(g=3.6,L=L,density=DENSITY,T=T,dt=DT,nu_ext_ratio=DRIVE,seed=seed)
    rng=np.random.default_rng(seed); pos,labels,NE,NI=place_neurons(p,rng)
    net=build_connectivity_rot(p,pos,labels,NE,NI,rng,theta_EE=np.deg2rad(theta_deg),AR=AR,verbose=False)
    return p, net, net["pos"][:NE], compute_nu_theta(p)[0]
def read_axis(p, net, posE, nut, kick_xy, m):
    net["rng"]=np.random.default_rng(p.seed); on =simulate_kick(p,net,KICK_BOOST=2*nut,kick_center=list(kick_xy))
    net["rng"]=np.random.default_rng(p.seed); off=simulate_kick(p,net,KICK_BOOST=0.0,kick_center=list(kick_xy))
    env,fdt,agg=snn_event_envelope(on["E_spk_bool"],posE,m,DT); _,_,aggr=snn_event_envelope(off["E_spk_bool"],posE,m,DT)
    win=event_window_for_run(agg,aggr,fdt)
    if win is None: return None,None,0
    art=extract_lagpat(env,fdt,[win],float(env.min()),0.5*(float(env.max())-float(env.min())),0.5,fdt)
    art=attach_geometry(art,m); r0,b0=art.ranks[:,0],art.bools[:,0]
    ax=endpoint_centroid_axis(r0,b0,art.contact_coords,k_dir=3,eps_deg=0.5*PITCH)
    rd=direction_readability(r0,b0,art.contact_coords)
    return ax, rd, int(b0.sum())
m = montage()
for th in (45.0, 90.0):
    p,net,posE,nut = build(th,2.0)
    ax,rd,npart = read_axis(p,net,posE,nut, end_at(th), m)
    err = None if ax is None else round(axis_angle_error_deg(ax, np.deg2rad(th)),1)
    print(f"AR=2 theta={th} single-end: endpoint-axis err_vs_theta={err} | readability={None if rd is None or rd!=rd else round(rd,2)} | n_part={npart}", flush=True)
p,net,posE,nut = build(45.0,2.0)
ax2,_,np2 = read_axis(p,net,posE,nut, end_at(90.0), m)
e2 = None if ax2 is None else round(axis_angle_error_deg(ax2, np.deg2rad(45.0)),1)
print(f"kick-track (theta=45 fixed, kick@90-end): endpoint-axis err_vs_45={e2} | n_part={np2}", flush=True)
p,net,posE,nut = build(45.0,1.0)
axi,rdi,npi = read_axis(p,net,posE,nut, end_at(45.0), m)
print(f"AR=1 iso single-end: readability={None if rdi is None or rdi!=rdi else round(rdi,2)} | endpoint-axis={'None' if axi is None else 'set'} | n_part={npi}", flush=True)
