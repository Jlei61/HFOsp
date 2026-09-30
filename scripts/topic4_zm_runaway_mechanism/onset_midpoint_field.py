"""Bisect the displayed D coordinate along the actual full native Z path."""
from common import OUT,np,model,read,write,log
from onset_state_continuation import DEST
from transient_equal_D_fields import at_first_crossing
from datetime import datetime
import argparse


def prepare(left,right,label):
    fields={k:v for k,v in np.load(DEST/'fields.npz').items()};assert label not in fields
    s=model(40);D=[float(1-fields[k][s.E]@s.mean_weights) for k in [left,right]]
    assert 0<=D[0]<D[1]<=1
    target=.5*sum(D)
    native=np.load(OUT/'transient_native_Z_path_20260923/native_Z_path.npz')
    field,info=at_first_crossing(native['Z'],native['time_ms'],s,target)
    fields[label]=field
    temporary=DEST/'fields.next.npz';np.savez_compressed(temporary,**fields)
    temporary.replace(DEST/'fields.npz')
    write(DEST/f'{label}_field_construction.json',dict(
        created_local=datetime.now().astimezone().isoformat(),label=label,D=target,
        parameter_bracket_fields=[left,right],parameter_bracket_D=D,
        construction='First upward crossing of the midpointD in the original native full spatial Z path; interpolate only the two adjacent actual temporal samples. NOT a uniformZ field or interpolation directly between the distant endpoint shapes.',
        native_path_interpolation=info,
        regional_Z={name:float(np.average(field[s.E&(s.geo['group_region']==k)],weights=s.sizes[s.E&(s.geo['group_region']==k)]))
                    for k,name in enumerate(['Core A','Core B','Surround'])}))
    log('ONSET MIDPOINT FIELD',label,target,info)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left');p.add_argument('right');p.add_argument('label');a=p.parse_args();prepare(a.left,a.right,a.label)
