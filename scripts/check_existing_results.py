#!/usr/bin/env python3
"""Verify locally generated STELLAR Encoder ablation outputs and summarize them.
Usage: python scripts/check_existing_results.py --results ~/projects/stellar/results
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--out', type=Path, default=Path('tables'))
args = parser.parse_args()
args.out.mkdir(exist_ok=True, parents=True)
seeds = [42, 7, 123, 2026]
frames = []
paired = []
for seed in seeds:
    root = args.results / f'masked_transductive_seed{seed}_ep250'
    comp = pd.read_csv(root/'comparison.csv')
    required = {'model','test_accuracy','test_macro_f1','test_balanced_accuracy'}
    assert required.issubset(comp.columns), f'Incomplete comparison: {root}'
    assert set(comp['model']) == {'spatial','no_edges'}, f'Unexpected model names: {root}'
    assert len(comp)==2, f'Duplicate model records: {root}'
    comp['seed'] = seed
    frames.append(comp)
    a=pd.read_csv(root/'spatial_predictions.csv').sort_values('cell_index').reset_index(drop=True)
    b=pd.read_csv(root/'no_edges_predictions.csv').sort_values('cell_index').reset_index(drop=True)
    for d in [a,b]:
        assert {'cell_index','true','predicted'}.issubset(d.columns)
        assert d.cell_index.is_unique
    assert a.cell_index.equals(b.cell_index), f'Mismatched test cells in seed {seed}'
    assert a['true'].equals(b['true']), f'Mismatched test labels in seed {seed}'
    assert len(a)==17397, f'Unexpected test size seed {seed}: {len(a)}'
    a_corr=(a.predicted==a.true).astype(int)
    b_corr=(b.predicted==b.true).astype(int)
    paired.append({'seed':seed,'n_test':len(a),
                   'spatial_only_correct':int(((a_corr==1)&(b_corr==0)).sum()),
                   'no_edges_only_correct':int(((a_corr==0)&(b_corr==1)).sum()),
                   'both_correct':int(((a_corr==1)&(b_corr==1)).sum()),
                   'both_wrong':int(((a_corr==0)&(b_corr==0)).sum()),
                   'delta_accuracy_pp':float((a_corr-b_corr).mean()*100)})
all_df=pd.concat(frames,ignore_index=True)
all_df.to_csv(args.out/'all_seed_results.csv',index=False)
wide=all_df.pivot(index='seed',columns='model',values=['test_accuracy','test_macro_f1','test_balanced_accuracy'])
metrics = ['test_accuracy','test_macro_f1','test_balanced_accuracy']
summary=[]
for metric in metrics:
    spatial=wide[metric]['spatial']
    noedges=wide[metric]['no_edges']
    diff=spatial-noedges
    summary.append({'metric':metric,'spatial_mean':spatial.mean(),
                    'spatial_sd':spatial.std(ddof=1),
                    'no_edges_mean':noedges.mean(),'no_edges_sd':noedges.std(ddof=1),
                    'paired_difference_mean':diff.mean(),
                    'paired_difference_sd':diff.std(ddof=1),
                    'spatial_win_seeds':int((diff>0).sum()),'n_seeds':len(diff)})
sumdf=pd.DataFrame(summary)
sumdf.to_csv(args.out/'summary_metrics.csv',index=False)
pairdf=pd.DataFrame(paired)
pairdf.to_csv(args.out/'paired_test_predictions.csv',index=False)
print('=== VALIDATED FOUR-SEED SUMMARY ===')
print(sumdf.round(5).to_string(index=False))
print('\n=== PAIRED TEST PREDICTION COMPARISON ===')
print(pairdf.to_string(index=False))
hfile=args.results/'neighborhood_analysis/class_stratified_all_seeds.csv'
if hfile.exists():
    h=pd.read_csv(hfile)
    h=h[(h.n_low>=30)&(h.n_high>=30)]
    if not h.empty:
        s=h.groupby('class_id').agg(n_seeds=('seed','nunique'),
                   mean_difference_pp=('high_minus_low_pp','mean'),
                   sd_difference_pp=('high_minus_low_pp','std'),
                   positive_seeds=('high_minus_low_pp',lambda x:int((x>0).sum())))
        s.to_csv(args.out/'within_class_homophily_summary.csv')
        print('\n=== EXPLORATORY WITHIN-CLASS HOMOPHILY ===')
        print(s.round(4).to_string())
print('\nSaved aggregate tables in',args.out.resolve())
print('NOTE: Technical seed repeats on one graph are NOT independent biological replicates.')
