"""Render every predeclared +15-degree half-turn candidate, without a new gate."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,__version__
from scripts import analyze_topic4_shape_output_response as an
from scripts.render_topic4_shape_output_gifs import rotation_clip

def main():
    base=Path('/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913')
    out=base/'analysis/plus15_rotation_review';out.mkdir(exist_ok=True)
    path=base/'response/units/global_axis_+15/2511_847101/workers/trajectory.json'
    r,a,ids=an.an.load_unit(path,1500.)
    candidate=an.rt.read(base/'candidates/global_axis_+15.json');candidate['topology']=2511
    physics=an.rt.read(path.parent.parent/'applied_physics.json')
    source=base/'rotation'/hashlib.sha256(str(path).encode()).hexdigest()[:20]/'result.json'
    rotation=an.rt.read(source);tracks=[t for t in rotation['tracks'] if t['half_turn_candidate']]
    rows=[]
    for track in tracks:
        folder=out/f'track_{track["id"]}';folder.mkdir(exist_ok=True)
        movie=rotation_clip(candidate,847101,r,a,physics,folder,dict(tracks=[track]))
        movie['selection']='All predeclared half-turn candidates from this completed unit, one clip each; no additional selection.'
        gif=folder/movie['file'];lo=movie['window_ms'][0]
        with Image.open(gif) as im:
            n=im.n_frames
            for i in range(n):im.seek(i);im.load()
            # Dense frames only within each already detected track, not chosen by appearance.
            frame_ids=np.linspace(round((track['start_ms']-lo)/2),round((track['end_ms']-lo)/2),9).astype(int)
            sheet=Image.new('RGB',(1200,1260),'white');draw=ImageDraw.Draw(sheet)
            for k,i in enumerate(frame_ids):
                im.seek(int(i));tile=im.convert('RGB');tile.thumbnail((400,400));x=k%3*400;y=k//3*420;sheet.paste(tile,(x,y+20));draw.text((x+5,y+3),f't={lo+2*i:.1f}ms',fill='black')
            sheet.save(folder/'dense_track_frames.png')
        movie.update(path=str(gif),frames=n,sha256=an.rt.sha(gif),preview=str(folder/'dense_track_frames.png'))
        an.rt.write(folder/'manifest.json',movie);rows.append(movie)
    an.rt.write(out/'manifest.json',dict(created_unix=time.time(),source=str(source),source_sha256=an.rt.sha(source),trajectory=str(path),arrays_sha256=r['arrays_sha256'],decoder='Pillow '+__version__,candidate_count=len(tracks),movies=rows,physical_runs_added=0,producer=str(Path(__file__)),producer_sha256=an.rt.sha(Path(__file__))))
    (out/'README.md').write_text('# +15度首噪声的全部旋转筛查候选\n\n'+''.join(f'### track_{t["id"]}\n本完整20秒运行内既定筛查命中的候选，逐例显示原生2ms帧，不作再次挑选。白圆为core、青环为固定诊断环；仅约半圈，是否为持续重入需与原生活动一起判断。\n**关注点**：固定环内活动方向转动、传播前沿弯曲与稳定螺旋并非同一结论。\n\n' for t in tracks))
    print(out,len(rows),flush=True)

if __name__=='__main__':main()
