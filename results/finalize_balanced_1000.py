#!/usr/bin/env python3
"""Repair legacy evidence representations and freeze the balanced 1,000 set."""
import json,re,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from worksurface.common import persona_slug
from worksurface.convert_tables import connect_registry
TASKS=ROOT/'data/worksurface_lite/tasks/tasks_balanced_1000_candidate.jsonl'

def compact(s):
 return re.sub(r'[^0-9a-z]+','',str(s).lower())

def main():
 tasks=[json.loads(x) for x in TASKS.read_text().splitlines()]
 invalid={'ws_lite_137_q005','ws_lite_386_q003','ws_lite_87_q002',
          'ws_lite_300_aq003','ws_lite_354_aq004','ws_lite_386_aq003'}
 tasks=[t for t in tasks if t['id'] not in invalid]
 graphs={};repairs={'graph_gold':0,'rag_span':0,'replaced_invalid':len(invalid)}
 for t in tasks:
  slug=persona_slug(t['source']['persona'])
  if slug not in graphs:
   graphs[slug]=json.load(open(ROOT/'data/worksurface_lite/profiles'/slug/'graph/surface_graph.json'))
  graph=graphs[slug];nodes={n['id'] for n in graph['nodes']}
  edges={(e.get('from'),e.get('rel'),e.get('to')) for e in graph['edges']}
  # Legacy list-all Graph tasks sometimes retained source dependencies that
  # were intentionally absent from the projected graph. Freeze their gold to
  # the complete set actually exposed by graph_neighbors.
  if t.get('required_surfaces')==['graph'] and t['id'].endswith('_q001'):
   tid=f"task_{t['source']['task_id']}"
   actual=sorted(next((n.get('filename') for n in graph['nodes'] if n['id']==dst),dst.split('::',1)[-1])
                 for src,rel,dst in edges if src==tid and rel=='task_requires_file' and dst in nodes)
   expected=t.get('gold_answer') if isinstance(t.get('gold_answer'),list) else []
   if actual and actual!=expected:
    t['gold_answer']=actual
    t['gold_evidence']=[{'surface':'graph','graph_query':{'node':tid,'relation':'task_requires_file'},
      'verified_complete_set':actual,'verified_result':actual,
      'claim':'Enumerating all exposed task_requires_file neighbors returns exactly this complete file set.'}]
    t['notes']='Frozen against the executable projected graph; non-projected source dependencies are excluded.'
    repairs['graph_gold']+=1
  # Preserve exact-span auditing while accepting formatting differences only
  # by replacing the recorded span with the exact source substring.
  pdir=ROOT/'data/worksurface_lite/profiles'/slug/'kb_docs'
  for e in t.get('gold_evidence',[]):
   if e.get('surface')!='rag' or not e.get('file') or not e.get('span'):continue
   f=pdir/e['file']
   if not f.exists():continue
   text=f.read_text(errors='ignore')
   if e['span'] in text:continue
   target=compact(e['span'])
   candidates=re.findall(r'[$€£¥]?\s*\d[\d,]*(?:\.\d+)?%?',text)
   matches=[x for x in candidates if compact(x)==target]
   if len(matches)==1:
    e['span']=matches[0]
    repairs['rag_span']+=1
 # Replace three invalid legacy table tasks with unused executable row counts.
 used_views={e.get('table') for t in tasks for e in t.get('gold_evidence',[]) if e.get('surface')=='table'}
 table_added=[]
 for pdir in sorted((ROOT/'data/worksurface_lite/profiles').iterdir()):
  if len(table_added)>=3:break
  con,active=connect_registry(str(pdir/'tables'))
  for view,meta in sorted(active.items()):
   cols=[c for c in meta.get('columns',[]) if not c['name'].startswith(('_source_','unnamed'))]
   if not cols:continue
   col=cols[0]['name'];orig=str(cols[0].get('orig') or col)
   query=f'''SELECT COUNT(*) FROM "{view}" WHERE NULLIF(TRIM(CAST("{col}" AS VARCHAR)), '') IS NOT NULL'''
   rows=int(con.execute(query).fetchone()[0]);sid=str(meta['task'])
   table_added.append({'id':f'ws_lite_{sid}_freeze_table_{len(table_added)+1:02d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':pdir.name.replace('_',' ').title(),'rubric_refs':['freeze_replacement_v1']},'question':f"How many rows in the {meta.get('sheet','sheet')} sheet of {meta['source_file']} contain a non-empty '{orig}' value?",'difficulty':'easy','task_type':'table_only','required_surfaces':['table'],'gold_tools':['table_query'],'applicable_skills':[],'gold_answer':rows,'answer_type':'number','gold_evidence':[{'surface':'table','table':view,'source_file':meta['source_file'],'source_sheet':meta.get('sheet','sheet'),'columns':[col],'query':query,'verified_result':rows,'claim':'The executable non-empty-value count returns the answer.'}],'notes':'QC replacement for an invalid legacy item.'})
   used_views.add(view)
   if len(table_added)>=3:break
  con.close()
 # Replace three invalid legacy RAG tasks with exact single-number statements.
 rag_added=[]
 for pdir in sorted((ROOT/'data/worksurface_lite/profiles').iterdir()):
  if len(rag_added)>=3:break
  for f in sorted((pdir/'kb_docs').glob('t*__*.md')):
   sidm=re.match(r't(\d+)__',f.name)
   if not sidm:continue
   for line in f.read_text(errors='ignore').splitlines():
    vals=re.findall(r'[$€£¥]?\s*\d[\d,]*(?:\.\d+)?%',line)
    if len(vals)!=1 or len(line)<18 or len(line)>180:continue
    span=vals[0].strip();blank=line.replace(vals[0],'___',1).strip();sid=sidm.group(1)
    rag_added.append({'id':f'ws_lite_{sid}_freeze_rag_{len(rag_added)+1:02d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':pdir.name.replace('_',' ').title(),'rubric_refs':['freeze_replacement_v1']},'question':f"In {f.name.split('__',1)[1]}, complete this statement: '{blank}' What exact percentage fills the blank?",'difficulty':'easy','task_type':'rag_only','required_surfaces':['rag'],'gold_tools':['kb_search'],'applicable_skills':[],'gold_answer':span,'answer_type':'string','gold_evidence':[{'surface':'rag','file':f.name,'span':span,'claim':'The exact percentage occurs verbatim in the cited statement.'}],'notes':'QC replacement for an invalid legacy item.'})
    break
   if len(rag_added)>=3:break
 tasks.extend(table_added+rag_added);repairs['table_replacements']=len(table_added);repairs['rag_replacements']=len(rag_added)
 if len(tasks)!=1000:raise RuntimeError(f'freeze size {len(tasks)}')
 TASKS.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks))
 (ROOT/'results/balanced_1000_freeze_report.json').write_text(json.dumps(repairs,indent=2))
 print(json.dumps(repairs,indent=2))

if __name__=='__main__':main()
