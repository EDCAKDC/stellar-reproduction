"""Generate publication-style exploratory plots from locally produced STELLAR CSVs.

Example (run from the unpacked project directory):
    python make_figures.py --results ~/projects/stellar/results
Requires: numpy, pandas, matplotlib.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--results', type=Path, default=Path.home()/'projects/stellar/results')
p.add_argument('--output', type=Path, default=Path('figures'))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)

# Paired seed-level global macro F1
parts=[]
for seed in [42,7,123,2026]:
    file = a.results / f'masked_transductive_seed{seed}_ep250/comparison.csv'
    if not file.exists():
        raise FileNotFoundError(file)
    df=pd.read_csv(file)
    parts.append(df.assign(seed=seed))
combined=pd.concat(parts,ignore_index=True)
wide=combined.pivot(index='seed',columns='model',values='test_macro_f1')
for col in ['spatial','no_edges']:
    if col not in wide:
        raise ValueError(f'Missing model: {col}')
fig,ax=plt.subplots(figsize=(8,4.8))
arr=[wide['spatial'],wide['no_edges']]
means=[float(v.mean()) for v in arr]; sds=[float(v.std(ddof=1)) for v in arr]
x=np.arange(2)
ax.bar(x,means,yerr=sds,capsize=5,width=.52,alpha=.5)
for j,seed in enumerate(wide.index):
    pts=wide.loc[seed,['spatial','no_edges']].to_numpy(dtype=float)
    ax.plot(x,pts,marker='o',linewidth=.8,alpha=.7,label=f'seed {seed}')
ax.set_xticks(x,['Spatial edges','No neighbor edges'])
ax.set_ylabel('Held-out test macro-F1')
ax.set_ylim(0,1)
ax.set_title('STELLAR Encoder ablation — exploratory transductive evaluation')
ax.legend(loc='lower right',fontsize=8)
fig.tight_layout()
fig.savefig(a.output/'macro_f1_from_local_results.png',dpi=300)
plt.close(fig)

# Within-class contrast in high-vs-low homophily; scatter every available seed
file=a.results/'neighborhood_analysis/class_stratified_all_seeds.csv'
if not file.exists():
    raise FileNotFoundError(file)
df=pd.read_csv(file)
df=df[(df.n_low >= 30) & (df.n_high >= 30)]
order=[(7,'Squamous epithelial'),(0,'B cells'),(2,'Innate'),(9,'T cells')]
fig,ax=plt.subplots(figsize=(9,5.2))
for k,(cid,name) in enumerate(order):
    d=df[df.class_id==cid].sort_values('seed')
    if d.empty:
        continue
    values=d.high_minus_low_pp.to_numpy(dtype=float)
    jitter=np.linspace(-.10,.10,len(values))
    ax.scatter(k+jitter,values,s=38,alpha=.7,zorder=2)
    ax.errorbar(k,values.mean(),yerr=values.std(ddof=1) if len(values)>1 else 0,
                fmt='D',color='black',capsize=5,zorder=3)
ax.axhline(0,linestyle='--',color='gray',linewidth=1)
ax.set_xticks(np.arange(len(order)),[x[1] for x in order],rotation=12)
ax.set_ylabel('High minus low homophily effect (percentage points)')
ax.set_title('Within-class post hoc analysis (seed-level points; mean ± SD)')
fig.tight_layout()
fig.savefig(a.output/'class_homophily_effect_from_local_results.png',dpi=300)
plt.close(fig)
print('FIGURES GENERATED:')
print(a.output/'macro_f1_from_local_results.png')
print(a.output/'class_homophily_effect_from_local_results.png')
