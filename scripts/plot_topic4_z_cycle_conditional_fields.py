#!/usr/bin/env python3
"""Four conditional fields along the exact native-Z entry/refill/return path."""
from analyze_topic4_prescribed_z_phase import *
from plot_topic4_prescribed_z_phase import save
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    a=np.load(OUT/'exact_phase_snapshots.npz');times=(8800,10500,11100,12500)
    assert all(f't{tm}_r' in a.files for tm in times)
    s=MixedTimescaleSystem(quadrature=33);m=s.m;n=s.n;rows=[]
    saved=np.load(PREVIOUS/'mixed_timescale/native_replay.npz');native=np.load(REFERENCE/'trajectory.npz')
    fig=plt.figure(figsize=(13,11),layout='constrained');g=fig.add_gridspec(3,2,height_ratios=[.5,1,1]);top=fig.add_subplot(g[0,:])
    top.fill_between(native['z_time_ms']/1000,native['z_stats'][:,2],native['z_stats'][:,4],color='#cdb8db',alpha=.4)
    top.plot(native['z_time_ms']/1000,native['z_stats'][:,0],c='#70388d',lw=1.5)
    top.axvspan(10.68,11.68,color='#238662',alpha=.12)
    top.set(xlim=(0,13.68),ylim=(.5,1.03),xlabel='Reference time (s)',ylabel='Native E-target Z')
    for idx,tm in enumerate(times):
        ax=fig.add_subplot(g[1+idx//2,idx%2]);key=f't{tm}';r=a[key+'_r'];c=a[key+'_current'];z=a[key+'_z'];ne=a[key+'_expected_e'];ni=a[key+'_expected_i']
        def dr(rr):
            e,i=rr[:n],rr[n:];te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms
            ex=np.r_[te*(m.v_ee@e+m.j_ext_e_mv**2*ne),ti*(m.v_ie@e+m.j_ext_i_mv**2*ni)]
            inh=np.r_[te*z*z*(m.v_ei@i),ti*(m.v_ii@i)]
            mu=np.r_[c[0]-z*c[1]+c[4],c[2]-c[3]+c[5]];d=(s.phi(mu,ex,inh)-rr)/s.tr
            return np.array([np.average(d[:n],weights=m.count_e),np.average(d[n:],weights=m.count_i)])*1e6
        def f(xy):return dr(np.r_[bounded_mean(r[:n],m.count_e,xy[0]/1000,.5),bounded_mean(r[n:],m.count_i,xy[1]/1000,1.)])
        center=np.array([np.average(r[:n],weights=m.count_e),np.average(r[n:],weights=m.count_i)])*1000
        trajectory=saved['fields_hz'][tm-51:tm+50];tr=np.c_[np.average(trajectory[:,0],axis=1,weights=m.count_e),np.average(trajectory[:,1],axis=1,weights=m.count_i)]
        endpoint=center+dr(r)*np.array([s.tr[0],s.tr[n]])*.001
        xmax=min(500,max(10.,max(tr[:,0].max(),center[0],endpoint[0])*1.3+2))
        ymax=min(1000,max(10.,max(tr[:,1].max(),center[1],endpoint[1])*1.3+2))
        xs=np.linspace(0,xmax,41);ys=np.linspace(0,ymax,41);X,Y=np.meshgrid(xs,ys);U=np.zeros_like(X);V=U.copy()
        ers=[bounded_mean(r[:n],m.count_e,x/1000,.5) for x in xs];irs=[bounded_mean(r[n:],m.count_i,y/1000,1.) for y in ys]
        for j in range(41):
            for k in range(41):U[j,k],V[j,k]=dr(np.r_[ers[k],irs[j]])
        denom=np.maximum(np.hypot(U/xmax,V/ymax),1e-12)
        ax.quiver(X[::4,::4],Y[::4,::4],(U/denom)[::4,::4],(V/denom)[::4,::4],color='0.7',angles='xy',scale_units='xy',scale=18,width=.003)
        for val,color in ((U,'#b13c76'),(V,'#24818b')):
            if val.min()<0<val.max():ax.contour(X,Y,val,levels=[0],colors=[color],linewidths=2)
        ax.plot(tr[:,0],tr[:,1],c='k',lw=1.2)
        for j in (10,35,65,85):ax.annotate('',xy=tr[j+3],xytext=tr[j],arrowprops={'arrowstyle':'->','color':'k','lw':1})
        ax.scatter(*center,s=48,c='#e3a731',edgecolor='black',zorder=5)
        if idx==3:ax.text(.5,.8,'Snapshot between events after refill',transform=ax.transAxes,ha='center',fontsize=11,color='#596066')
        ax.set(xlim=(-.015*xmax,xmax),ylim=(-.015*ymax,ymax),xlabel='Mean E rate (Hz)',ylabel='Mean I rate (Hz)',title=f'{idx+1}: {tm/1000:g} s | mean Z = {np.average(z,weights=m.count_e):.3f}')
        top.axvline(tm/1000,c='#b69245',ls='--',lw=.8);top.text(tm/1000,.52,str(idx+1),ha='center',fontsize=11,fontweight='bold')
        sol=root(f,center,tol=1e-10);error=float(abs(f(sol.x)).max());assert error<1e-5
        hh=.001;J=np.column_stack([(f(sol.x+np.eye(2)[j]*hh)-f(sol.x-np.eye(2)[j]*hh))/(2*hh) for j in range(2)]) if min(sol.x)>.001 else None
        actual=(a[key+'_next_r']-r)/s.dt;actual=np.array([np.average(actual[:n],weights=m.count_e),np.average(actual[n:],weights=m.count_i)])*1e6
        identity=float(abs(actual-dr(r)).max());assert identity<1e-6
        row={'time_ms':tm,'center_hz':center.tolist(),'intersection_hz':sol.x.tolist(),'root_residual_hz_per_s':error,'exact_center_drift_error':identity,
            'conditional_eigenvalues_per_s':[[float(v.real),float(v.imag)] for v in np.linalg.eigvals(J)] if J is not None else None}
        rows.append(row);np.savez_compressed(OUT/f'cycle_plane_{tm}.npz',X=X,Y=Y,U=U,V=V,trajectory_hz=tr,center_hz=center)
        write(OUT/'cycle_plane_plot_status.json',{'status':'RUNNING','rows':rows})
    handles=[Line2D([],[],c='#b13c76',lw=2,label='Conditional dE/dt = 0'),Line2D([],[],c='#24818b',lw=2,label='Conditional dI/dt = 0'),
        Line2D([],[],c='k',lw=1.3,label='Rate replay trajectory: +/-50 ms'),Line2D([],[],marker='o',ls='',mfc='#e3a731',mec='k',label='Exact rate snapshot')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=11)
    fig.suptitle('Conditional E–I fields along the native Z evolution and manual restoration\nCorrected rate replay; synaptic states and input fixed within each field',fontsize=14)
    save(fig,'native_z_cycle_conditional_fields',
        '上排沿用用户原SNN图中的完整空间Z轨迹，四个条件相平面对应8.8、10.5、11.1、12.5秒，即间隔活动、高活动、补回中和补回后。相平面的nullcline及方向场来自给定该空间Z及原OU输入的修正rate model，黑线为同一rate回放的实际前后50毫秒轨迹。',
        'Z来自原SNN，但这四张方向场不是直接从spike点拟合的原SNN向量场。每张固定隐藏状态，黑线沿途的隐藏状态却会变化；坐标范围分别适配所处状态，交点不证明整个延迟网络稳定，返回仍为外部干预。')
    write(OUT/'cycle_plane_plot_status.json',{'status':'COMPLETE','rows':rows,'scope':'Conditional projected fields of the corrected rate replay, anchored to the same native Z and manual-refill reference.'})


if __name__=='__main__':main()
