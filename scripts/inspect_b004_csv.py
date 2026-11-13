#!/usr/bin/env python3
"""Inspect public B004 CODEX CSV *before* selecting a section-held-out experiment.
Does not train a model or write any individual-cell records.
Usage: python scripts/inspect_b004_csv.py ~/Downloads/B004_training_dryad.csv
"""
import argparse
import json
from pathlib import Path
import re
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('csv',type=Path)
p.add_argument('--out',type=Path,default=Path('tables/b004_schema_audit.json'))
a=p.parse_args()
if not a.csv.is_file():
    p.error(f'CSV file not found: {a.csv}')
df=pd.read_csv(a.csv,low_memory=False)
print(f'Rows: {len(df):,} | Columns: {len(df.columns)}')
print('\n=== ALL COLUMN NAMES ===')
for col in df.columns: print(' -',col, '|',df[col].dtype)
pattern=re.compile(r'(region|section|tissue|donor|sample|organ|\broi\b|\bfov\b|image|cell[_ ]?type|annotation|label|(^x$)|(^y$)|x.?coord|y.?coord)',re.I)
metadata=[c for c in df if pattern.search(str(c))]
print('\n=== CANDIDATE METADATA COLUMNS (REVIEW MANUALLY) ===')
audit={'n_rows':len(df),'n_columns':len(df.columns),'columns':list(map(str,df.columns)),
       'candidate_metadata':{},'numeric_cols':list(map(str,df.select_dtypes('number').columns))}
for c in metadata:
    s=df[c]
    count=int(s.nunique(dropna=True))
    record={'dtype':str(s.dtype),'unique_values':count,'missing':int(s.isna().sum())}
    if count<=40:
        counts=s.astype('string').value_counts(dropna=False).head(40)
        record['top_counts']={str(k):int(v) for k,v in counts.items()}
    else:
        record['preview_values']=[str(v) for v in s.dropna().head(5).tolist()]
    audit['candidate_metadata'][str(c)] = record
    print('\n',c,': unique=',count,' missing=',record['missing'])
    if 'top_counts' in record:
        for k,v in record['top_counts'].items():print('   ',k,':',v)
print('\nIMPORTANT: Do not choose feature columns automatically. Review metadata and marker lists first.')
a.out.parent.mkdir(exist_ok=True,parents=True)
a.out.write_text(json.dumps(audit,indent=2,ensure_ascii=False))
print('Saved schema-only report:',a.out)
