#!/usr/bin/env python3
"""Estimate proof-carrying expansion capacity without generating tasks."""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from worksurface.convert_tables import connect_registry
PROFILES=ROOT/'data/worksurface_lite/profiles'
NUMERIC=re.compile(r'[$€£¥]?\b\d[\d,]*(?:\.\d+)?%?\b')

def main():
    table=Counter(); per_task=defaultdict(lambda:Counter())
    for pdir in sorted(PROFILES.iterdir()):
        tdir=pdir/'tables'
        if not (tdir/'registry.json').exists(): continue
        con,active=connect_registry(str(tdir))
        for view,meta in active.items():
            tid=str(meta['task']); named=[c['name'] for c in meta.get('columns',[]) if not c['name'].startswith('unnamed')]
            table['views']+=1; per_task[tid]['row_count']+=1
            for col in named:
                q=f'''SELECT COUNT(*) AS n, COUNT("{col}") AS present, COUNT(DISTINCT "{col}") AS distinct_n, COUNT(TRY_CAST(REPLACE(REPLACE(REPLACE("{col}",'$',''),',',''),'%','') AS DOUBLE)) AS numeric_n FROM "{view}"'''
                try:n,p,d,nn=con.execute(q).fetchone()
                except Exception:continue
                if d and d>1: table['distinct_candidates']+=1;per_task[tid]['distinct']+=1
                if nn and nn/max(n,1)>=0.6:
                    table['numeric_columns']+=1;per_task[tid]['numeric_aggregates']+=4
        con.close()
    spans=Counter()
    for pdir in sorted(PROFILES.iterdir()):
        kb=pdir/'kb_docs'
        if not kb.exists():continue
        grouped=defaultdict(list)
        for f in kb.glob('t*__*.md'):
            m=re.match(r't(\d+)__',f.name)
            if m:grouped[m.group(1)].append(f)
        for tid,files in grouped.items():
            loc=defaultdict(set)
            for f in files:
                text=f.read_text(errors='ignore')
                for s in set(NUMERIC.findall(text)):
                    if len(s)>=4:loc[s].add(f.name)
            unique=sum(len(v)==1 for v in loc.values())
            spans['tasks_with_docs']+=1;spans['unique_spans']+=unique;per_task[tid]['unique_doc_spans']+=unique
    report={'table_capacity':dict(table),'rag_graph_capacity':dict(spans),'per_task':per_task}
    out=ROOT/'results/expansion_capacity.json';out.write_text(json.dumps(report,indent=2,default=dict))
    print(json.dumps({'table_capacity':dict(table),'rag_graph_capacity':dict(spans)},indent=2))
if __name__=='__main__':main()
