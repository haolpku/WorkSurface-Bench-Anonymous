#!/usr/bin/env python3
"""Audit verifiable same-task joins for two- and three-surface expansion."""
from __future__ import annotations
import json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from worksurface.convert_tables import connect_registry
PROFILES=ROOT/'data/worksurface_lite/profiles'
COMMON={'true','false','yes','none','null','china','other','normal','completed','approved'}
def usable(v):
 s=str(v).strip()
 if not 3<=len(s)<=60 or s.lower() in COMMON:return False
 if s.isdigit() and len(s)<4:return False
 return bool(re.search(r'[A-Za-z\u4e00-\u9fff0-9]',s))
def main():
 links=[];gt=[]
 for pdir in sorted(PROFILES.iterdir()):
  kb=pdir/'kb_docs';tdir=pdir/'tables'
  if not kb.exists() or not (tdir/'registry.json').exists():continue
  docs=defaultdict(list)
  for f in kb.glob('t*__*.md'):
   m=re.match(r't(\d+)__',f.name)
   if m:docs[m.group(1)].append((f,f.read_text(errors='ignore')))
  con,active=connect_registry(str(tdir));bytask=defaultdict(list)
  for view,meta in active.items():bytask[str(meta['task'])].append((view,meta))
  for tid,views in bytask.items():
   column_owners=defaultdict(list)
   schema_inventory=[]
   for view,meta in views:
    schema_inventory.append({'view':view,'source_file':meta['source_file'],'source_sheet':meta['sheet'],'columns':[{'normalized':c['name'],'original':c.get('orig',c['name'])} for c in meta.get('columns',[]) if not c['name'].startswith('unnamed')]})
    for c in meta.get('columns',[]):
     if not c['name'].startswith('unnamed'):column_owners[c['name']].append((view,meta,c))
   for col,owners in column_owners.items():
    if len(owners)==1:
     view,meta,c=owners[0];gt.append({'task_id':tid,'view':view,'file':meta['source_file'],'sheet':meta['sheet'],'column':col,'column_orig':c.get('orig',col),'rows':meta['rows'],'verified_task_table_schema':schema_inventory,'persona':pdir.name})
   if tid not in docs:continue
   for view,meta in views:
    for c in meta.get('columns',[]):
     col=c['name']
     if col.startswith('unnamed'):continue
     try:vals=[r[0] for r in con.execute(f'SELECT DISTINCT "{col}" FROM "{view}" WHERE "{col}" IS NOT NULL LIMIT 300').fetchall()]
     except Exception:continue
     for value in vals:
      if not usable(value):continue
      s=str(value).strip();hits=[];actual={}
      for f,text in docs[tid]:
       m=re.search(re.escape(s),text,re.I)
       if m:hits.append(f);actual[f]=m.group(0)
      if len(hits)==1:
       try:count=con.execute(f'SELECT COUNT(*) FROM "{view}" WHERE CAST("{col}" AS VARCHAR)=?',[s]).fetchone()[0]
       except Exception:continue
       links.append({'task_id':tid,'doc':hits[0].name,'doc_span':actual[hits[0]],'value':s,'view':view,'file':meta['source_file'],'sheet':meta['sheet'],'column':col,'column_orig':c.get('orig',col),'match_count':count,'persona':pdir.name})
  con.close()
 # deduplicate same logical join
 key=lambda x:(x['task_id'],x['doc'],x['value'],x['view'],x['column'])
 links=list({key(x):x for x in links}.values());gt=list({(x['task_id'],x['view'],x['column']):x for x in gt}.values())
 report={'graph_table_unique_column_links':len(gt),'rag_table_shared_value_links':len(links),'tri_surface_links':len(links),'by_persona':dict(Counter(x['persona'] for x in links)),'by_task_count':len({x['task_id'] for x in links}),'graph_table_candidates':gt,'shared_value_candidates':links}
 (ROOT/'results/cross_surface_capacity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in report.items() if not k.endswith('candidates')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
