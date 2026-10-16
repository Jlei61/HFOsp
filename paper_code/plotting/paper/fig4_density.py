"""Copied from fig4/source/fig4_parameter_response_assets.py."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb,LightSource,LinearSegmentedColormap,Normalize
from mpl_toolkits.mplot3d import proj3d

def completed_density_layer(arrays,geometry):
    """Reuse the approved relief, contours and fitted shafts at plate resolution."""
    xx,yy=np.meshgrid(arrays['x_mm'],arrays['y_mm']);z=arrays['density_per_mm2']
    deep=np.array(to_rgb('#65509A'))
    cmap=LinearSegmentedColormap.from_list('position_purple',[np.ones(3)*(1-t)+deep*t for t in np.linspace(.015,1,256)])
    norm=Normalize(0,float(z.max()));ground=.00005
    temp=plt.figure(figsize=(4,4),dpi=300)
    ax=temp.add_axes([0,0,1,1],projection='3d',computed_zorder=False)
    gx,gy=np.meshgrid([1,19],[1,19])
    ax.plot_surface(gx,gy,np.zeros_like(gx),color=cmap(0.),shade=False,linewidth=0,antialiased=False,zorder=.5)
    background=np.unique(arrays['all_positions_mm'].reshape(-1,2),axis=0)
    ax.scatter(*background.T,zs=ground,s=3.2,c='#B6B2BF',alpha=.36,linewidths=0,depthshade=False,zorder=1)
    rgba=cmap(norm(z));spacing=float(np.diff(arrays['x_mm'])[0])
    rgba[...,:3]=LightSource(azdeg=315,altdeg=50).shade_rgb(rgba[...,:3],z*10.5/(z.max()*1.2),
        dx=spacing,dy=spacing,blend_mode='soft',fraction=.65)
    view=np.ix_((arrays['y_mm']>=1)&(arrays['y_mm']<=19),(arrays['x_mm']>=1)&(arrays['x_mm']<=19))
    ax.plot_surface(xx[view],yy[view],np.where(z[view]>=z.max()*1e-4,z[view],np.nan),
        rcount=321,ccount=321,facecolors=rgba[view],linewidth=0,edgecolor='none',antialiased=False,shade=False,zorder=2)
    isolated=[]
    for core in range(2):
        contour=geometry['density_contours']['AB'[core]]['0.9']
        for points in contour['paths_mm']:
            points=np.asarray(points)
            ax.plot(*points.T,zs=ground,color='#9784B6',lw=.6,alpha=.8,zorder=5)
        points=arrays['selected_positions_mm'][:,core]
        squared=((points[:,None,:]-points[None,:,:])**2).sum(2)
        values=np.exp(-squared/(2*.35**2)).mean(1)/(2*np.pi*.35**2)
        isolated.extend(points[values<contour['threshold_per_mm2']])
    isolated=np.unique(np.asarray(isolated).reshape(-1,2),axis=0)
    ax.scatter(*isolated.T,zs=ground,s=12,facecolors='none',edgecolors='#65509A',linewidths=.65,depthshade=False,zorder=6)
    names=np.array(geometry['contact_names']);xy=np.array(geometry['straight_contact_xy_mm'])
    for shaft in ('SCL','ICL'):
        ix=sorted(np.flatnonzero(np.char.startswith(names,shaft)),key=lambda i:int(names[i][len(shaft):]))
        points=xy[ix];direction=points[-1]-points[0];direction/=np.linalg.norm(direction)
        np.testing.assert_allclose(points-points[0],((points-points[0])@direction)[:,None]*direction,atol=1e-10,rtol=0)
        ax.plot(*points[[0,-1]].T,zs=ground,color='#948F9D',lw=.6,zorder=7)
        ax.plot(*points.T,zs=ground,ls='none',marker='o',ms=2.6,mfc='white',mec='#948F9D',mew=.65,zorder=7.1)
    ax.set(xlim=(1,19),ylim=(1,19),zlim=(0,float(z.max())*1.2))
    ax.view_init(elev=64,azim=-90.000001);ax.set_proj_type('ortho');ax.set_box_aspect((18,18,10.5));ax.set_axis_off()
    temp.canvas.draw()
    def projected(x,y,h):
        px,py,_=proj3d.proj_transform(x,y,h,ax.get_proj())
        return ax.transData.transform(np.column_stack([np.atleast_1d(px),np.atleast_1d(py)]))
    plane=projected([1,19,19,1],[1,1,19,19],np.zeros(4));lo,hi=plane.min(0),plane.max(0)
    plane_hi=hi.copy()
    # The paper's fixed ground-plane crop can cut a new peak near the far edge.
    # Extend only the canvas; preserve the ground-plane mapping and density values.
    hi[1]=max(hi[1],projected(xx.ravel(),yy.ravel(),z.ravel())[:,1].max()+8)
    rgba=np.asarray(temp.canvas.buffer_rgba()).copy();height=rgba.shape[0]
    crop=rgba[int(np.floor(height-hi[1])):int(np.ceil(height-lo[1])),int(np.floor(lo[0])):int(np.ceil(hi[0]))]
    labels=[]
    for component in arrays['core_components_per_mm2']:
        iy,ix=np.unravel_index(np.argmax(component),component.shape)
        pixel=projected(xx[iy,ix],yy[iy,ix],z[iy,ix]+z.max()*.05)[0]
        labels.append(1+(pixel-lo)/(plane_hi-lo)*18)
    plt.close(temp)
    extent=(1,19,1,1+(hi[1]-lo[1])/(plane_hi[1]-lo[1])*18)
    return crop,cmap,norm,labels,extent
