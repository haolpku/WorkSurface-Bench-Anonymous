#!/usr/bin/env python3
"""Hard validation for every evidence item in the balanced 1,000 candidate."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from worksurface.convert_tables import connect_registry
from worksurface.common import persona_slug
TASKS=ROOT/'data/worksurface_lite/tasks/tasks_balanced_1000_candidate.jsonl'
def norm(v):
 if isinstance(v,float) and v.is_integer():return int(v)
 if isinstance(v,float):return round(v,2)
 return v
def equivalent(got,want):
 if norm(got)==norm(want):return True
 if isinstance(got,(int,float)) and isinstance(want,(int,float)):
  return abs(float(got)-float(want))<=0.011
 if isinstance(want,list) and isinstance(got,list):
  first=[norm(r[0]) if isinstance(r,(list,tuple)) and r else norm(r) for r in got]
  return first==[norm(x) for x in want]
 return False
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tasks',default=str(TASKS));ap.add_argument('--report',default=str(ROOT/'results/balanced_1000_validation.json'));args=ap.parse_args()
 task_path=Path(args.tasks);tasks=[json.loads(x) for x in task_path.read_text().splitlines()];cons={};active={};graphs={};stats=Counter();errors=[]
 for t in tasks:
  slug=persona_slug(t['source']['persona']);pdir=ROOT/'data/worksurface_lite/profiles'/slug
  if slug not in cons:
   cons[slug],active[slug]=connect_registry(str(pdir/'tables'))
   gp=pdir/'graph/surface_graph.json';graphs[slug]=json.load(open(gp)) if gp.exists() else {'nodes':[],'edges':[]}
  node_ids={n['id'] for n in graphs[slug]['nodes']};edge_triples={(e.get('from'),e.get('rel'),e.get('to')) for e in graphs[slug]['edges']}
  for e in t.get('gold_evidence',[]):
   s=e.get('surface');stats[s]+=1
   if s=='table' and e.get('query'):
    try:r=cons[slug].execute(e['query']).fetchall()
    except Exception as ex:errors.append((t['id'],'sql_error',repr(ex)));continue
    if not r:errors.append((t['id'],'sql_empty',''));continue
    got=r[0][0] if len(r)==1 and len(r[0])==1 else r
    if 'verified_result' in e and not equivalent(got,e['verified_result']):errors.append((t['id'],'sql_mismatch',f"{got!r}!={e['verified_result']!r}"))
   elif s=='rag' and e.get('file') and e.get('span'):
    f=pdir/'kb_docs'/e['file']
    if not f.exists() or e['span'] not in f.read_text(errors='ignore'):errors.append((t['id'],'rag_span_missing',str(f)))
   elif s=='graph' and e.get('graph_path'):
    p=e['graph_path']
    if p[0] not in node_ids or p[-1] not in node_ids:errors.append((t['id'],'graph_node_missing',str(p)))
    elif len(p)>=3 and (p[0],p[1],p[2]) not in edge_triples:errors.append((t['id'],'graph_edge_missing',str(p)))
   elif s=='graph' and e.get('graph_query'):
    q=e['graph_query'];node=q.get('node');rel=q.get('relation')
    if node not in node_ids:errors.append((t['id'],'graph_query_node_missing',str(q)))
    elif rel and not any(src==node and relation==rel for src,relation,_ in edge_triples):errors.append((t['id'],'graph_query_edge_missing',str(q)))
  stats['tasks_checked']+=1
 for con in cons.values():con.close()
 new_errors=[e for e in errors if any(tag in e[0] for tag in ('_xgt_','_xrt_','_xtri_'))]
 report={'n_tasks':len(tasks),'evidence_counts':dict(stats),'errors':len(errors),'error_counts':dict(Counter(e[1] for e in errors)),'new_cross_surface_errors':len(new_errors),'new_cross_surface_error_examples':new_errors,'all_errors':errors}
 Path(args.report).write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
