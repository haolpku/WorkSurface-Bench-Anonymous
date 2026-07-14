#!/usr/bin/env python3
import json,glob,os,statistics
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];runs=ROOT/'runs_final1151';tasks=list(map(json.loads,open(ROOT/'data/worksurface_lite/tasks/tasks_final_1151.jsonl')))
reports={}
for p in glob.glob(str(runs/'*'/'S*.scored.json')):
 n=Path(p).name.replace('.scored.json','');setting,model=n.split('_',1);reports[(setting,model)]=json.load(open(p))
models=['gpt-4o-mini','deepseek-v4-pro','gemini-3.1-pro-preview','gpt-5.5'];settings=['S2','S3','S4','S5','S6'];types=['rag_only','table_only','graph_only','cross_surface']
x=[];y=[];labels=[]
for s in settings:
 for m in models:
  for typ in types:
   r=reports[(s,m)]['by_task_type'][typ];x.append(r['route_f1']);y.append(r['answer']);labels.append(f'{s}:{m}:{typ}')
def ranks(a):
 order=sorted(range(len(a)),key=lambda i:a[i]);r=[0]*len(a);i=0
 while i<len(order):
  j=i
  while j+1<len(order) and a[order[j+1]]==a[order[i]]:j+=1
  v=(i+j+2)/2
  for k in range(i,j+1):r[order[k]]=v
  i=j+1
 return r
rx,ry=ranks(x),ranks(y);mx,my=statistics.mean(rx),statistics.mean(ry);rho=sum((a-mx)*(b-my) for a,b in zip(rx,ry))/(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**.5
g={'models':models}
for s,k in [('S3','S3_naive'),('S4','S4_react'),('S6','S6_hint'),('S5','S5_constrained')]:g[k]=[reports[(s,m)]['overall']['answer'] for m in models]
lines={s:[statistics.mean(reports[(s,m)]['by_task_type'][t]['answer'] for m in models) for t in types] for s in settings}
data={'scatter_3a':{'x_route_f1':x,'y_answer':y,'labels':labels,'spearman_rho':rho},'guidance_3b':g,'per_surface_3c':{'lines':lines}}
(ROOT/'results/figure3_data.json').write_text(json.dumps(data,indent=2))
# Figure 4 inputs (the plotting script expects four legacy path labels).
d={'task_type':dict(Counter(t['task_type'] for t in tasks)),'persona':dict(Counter(t['source']['persona'] for t in tasks)),'answer_type':dict(Counter(t['answer_type'] for t in tasks))}
paths=Counter('graph_table_cross' if t['required_surfaces']==['graph','table'] else 'rag_graph_cross' if t['required_surfaces']==['rag','graph'] else 'llm_augmented' if t['id'].startswith('pool_') else 'deterministic' for t in tasks)
(ROOT/'results/quality_report.json').write_text(json.dumps({'distribution':d,'path_counts':{k:paths.get(k,0) for k in ['deterministic','llm_augmented','graph_table_cross','rag_graph_cross']}},indent=2))
print(json.dumps({'rho':rho,'n':len(x),'paths':paths},indent=2))
