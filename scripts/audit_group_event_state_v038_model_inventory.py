#!/usr/bin/env python3
"""Inventory actual layer tensors separately from frozen geometry and masks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors(values):
    return {name: {'shape': list(value.shape), 'stored_values': value.numel()}
            for name, value in values.items() if isinstance(value, torch.Tensor)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); args = parser.parse_args()
    summary = json.loads((args.root/'final_reports/summary_main.json').read_text())
    lineage_path = Path(summary['state_lineage']['path']); lineage = json.loads(lineage_path.read_text())
    adapters = []; decoders = {}
    for source in summary['source_card_manifest']:
        path = Path(source.get('_loaded_overlay_card') or source['_loaded_source_card']); card = json.loads(path.read_text())
        if not card.get('adapter_checkpoint'): continue
        checkpoint = Path(card['adapter_checkpoint']); saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        blocks = {name: tensors(saved[name]) for name in ('static_adapter','dynamic_adapter','bmark_dynamic_adapter')}
        adapters.append({'subject':card['subject'],'seed':card['state_seed'],'family':card['state_provenance']['family'],
                         'card':str(path),'card_sha256':sha(path),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                         'parameter_tensors':blocks,'parameter_counts':{name:sum(t['stored_values'] for t in block.values()) for name,block in blocks.items()},
                         'dynamic_activation':'GELU after down projection; bounded modulation at decoder steps',
                         'upstream_frozen':card['state_provenance']['frozen']})
        decoder = card['decoder_provenance']['checkpoint']
        if decoder not in decoders:
            state = torch.load(decoder,map_location='cpu',weights_only=False)
            # Names are the nn.Parameter fields in WEModel/LBSSModel; H,
            # distances and connection masks are registered buffers.
            names = {'contact_bias','recurrent','input_gain','bias','kappa_logit','readout_gain'}
            parameters = {k:v for k,v in state.items() if k in names or k.startswith('stop_head.')}
            buffers = {k:v for k,v in state.items() if k not in parameters}
            decoders[decoder] = {'checkpoint_sha256':sha(decoder),'parameter_tensors':tensors(parameters),
                                 'stored_parameter_count':sum(v.numel() for v in parameters.values()),
                                 'trainable_during_H2a':0,'buffers':tensors(buffers),
                                 'active_recurrent_mask_entries':int(state['node_mask'].sum()),
                                 'masked_recurrent_tensor_count':state['recurrent'].numel(),
                                 'interpretation':'stored recurrent matrix includes masked entries; geometry/masks are not learned weights'}
    result = {'status':'COMPLETE','h1_lineage':str(lineage_path),'h1_lineage_sha256':sha(lineage_path),
              'h1_model_count':len(lineage['h1_cards']),'h2a_adapter_count':len(adapters),'h2a_adapters':adapters,
              'frozen_decoder_count':len(decoders),'frozen_decoders':decoders,
              'interpretation':'layer dimensions and parameter inventory; no outcome reads or model fitting; not a capacity adequacy proof'}
    output = args.root/'model_inventory.json'; output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','h1_model_count','h2a_adapter_count','frozen_decoder_count')}))


if __name__ == '__main__': main()
