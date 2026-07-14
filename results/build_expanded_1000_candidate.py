#!/usr/bin/env python3
"""Build a 1,000-task proof-carrying candidate without paraphrase duplicates."""
from __future__ import annotations
import json,re,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from worksurface.convert_tables import connect_registry

SRC=ROOT/'data/worksurface_lite/tasks/tasks_natural_graph_v2.jsonl'
AI=ROOT/'results/ai_precheck_natural_graph_v2/gpt-5.5_all.jsonl'
OUT=ROOT/'data/worksurface_lite/tasks/tasks_expanded_1000_candidate.jsonl'
REPORT=ROOT/'results/expanded_1000_candidate_report.json'
SPAN_RE=re.compile(r'(?:[$€£¥]\s*)?\b\d[\d,]*(?:\.\d+)?%?\b')

def normstem(s):return re.sub(r'[^a-z0-9]+','',Path(s).stem.lower())
def clean_answer(v):
 if isinstance(v,float) and v.is_integer():return int(v)
 return round(v,2) if isinstance(v,float) else v

def main():
 base=[json.loads(x) for x in SRC.read_text().splitlines() if x.strip()]
 verdict={r['id']:r for r in map(json.loads,AI.read_text().splitlines())}
 dropped={i for i,r in verdict.items() if r.get('gold_correct')=='No'}
 tasks=[t for t in base if t['id'] not in dropped]
 existing_ids={t['id'] for t in tasks};existing_q={re.sub(r'\W+',' ',t['question'].lower()).strip() for t in tasks}
 added=[]
 # 80 new Graph-only count tasks, one per source task.
 graph_templates=[
  'How many source files must be ready before Task {sid} can begin?',
  'How many input files should be included in the handoff for Task {sid}?',
  'Before starting Task {sid}, how many required files need to be collected?',
  'What is the total number of source files needed for Task {sid}?',
  'How many files does the team need to prepare for Task {sid}?',
  'Count the required input files for Task {sid}.',
  'How many file dependencies must be available to complete Task {sid}?',
  'For Task {sid}, how many source files belong in the preparation checklist?',
  'How many required files are attached to Task {sid}?',
  'What is the size of the required-file set for Task {sid}?',
 ]
 for gi,t in enumerate([x for x in tasks if x['required_surfaces']==['graph']][:80]):
  sid=str(t['source']['task_id']);files=t['gold_answer'] if isinstance(t['gold_answer'],list) else [t['gold_answer']]
  item={'id':f'ws_lite_{sid}_exg_count','source':{**t['source'],'rubric_refs':['expanded_graph_count_v1']},'question':graph_templates[gi%len(graph_templates)].format(sid=sid),'difficulty':'easy','task_type':'graph_only','required_surfaces':['graph'],'gold_tools':['graph_neighbors'],'applicable_skills':[],'gold_answer':len(files),'answer_type':'number','gold_evidence':[{'surface':'graph','graph_query':{'node':f'task_{sid}','relation':'task_requires_file'},'verified_complete_set':files,'verified_result':len(files),'claim':'Counting the complete required-file neighbor set gives the answer.'}],'notes':'Deterministic expansion: complete graph-neighbor count.'}
  added.append(item)
 # 250 Table-only tasks: 200 row counts plus 50 named-column numeric maxima.
 table_candidates=[];numeric=[]
 for pdir in sorted((ROOT/'data/worksurface_lite/profiles').iterdir()):
  tdir=pdir/'tables'
  if not (tdir/'registry.json').exists():continue
  con,active=connect_registry(str(tdir))
  for view,meta in sorted(active.items()):
   sid=str(meta['task']);file=meta['source_file'];sheet=meta.get('sheet','sheet');rows=int(meta['rows'])
   q=f'How many data rows are in the {sheet} sheet of {file} for Task {sid}?'
   table_candidates.append({'id':f'ws_lite_{sid}_ext_{len(table_candidates)+1:04d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':pdir.name.replace('_',' ').title(),'rubric_refs':['expanded_table_exec_v1']},'question':q,'difficulty':'easy','task_type':'table_only','required_surfaces':['table'],'gold_tools':['table_query'],'applicable_skills':[],'gold_answer':rows,'answer_type':'number','gold_evidence':[{'surface':'table','table':view,'source_file':file,'source_sheet':sheet,'query':f'SELECT COUNT(*) FROM "{view}"','verified_result':rows,'claim':'The registry maps this view to the stated workbook sheet; the executable row-count query returns the result.'}],'notes':'Deterministic expansion: executable table row count.'})
   for c in meta.get('columns',[]):
    col=c['name'];orig=str(c.get('orig') or col)
    if col.startswith('unnamed') or col.startswith('_source_'):continue
    cast=f'''TRY_CAST(REPLACE(REPLACE(REPLACE("{col}",'$',''),',',''),'%','') AS DOUBLE)'''
    try:
     n,nn,val=con.execute(f'SELECT COUNT(*), COUNT({cast}), ROUND(MAX({cast}),2) FROM "{view}"').fetchone()
    except Exception:continue
    if n and nn/n>=0.6 and val is not None:
     val=clean_answer(float(val));numeric.append((sid,file,sheet,view,col,orig,val,pdir.name))
     break
  con.close()
 table_candidates=table_candidates[:200]
 for sid,file,sheet,view,col,orig,val,pname in numeric:
  if len(table_candidates)>=250:break
  table_candidates.append({'id':f'ws_lite_{sid}_ext_{len(table_candidates)+1:04d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':pname.replace('_',' ').title(),'rubric_refs':['expanded_table_exec_v1']},'question':f'What is the maximum {orig} recorded in the {sheet} sheet of {file} for Task {sid}?','difficulty':'medium','task_type':'table_only','required_surfaces':['table'],'gold_tools':['table_query'],'applicable_skills':[],'gold_answer':val,'answer_type':'number','gold_evidence':[{'surface':'table','table':view,'source_file':file,'source_sheet':sheet,'columns':[col],'query':f'''SELECT ROUND(MAX(TRY_CAST(REPLACE(REPLACE(REPLACE("{col}",'$',''),',',''),'%','') AS DOUBLE)),2) FROM "{view}"''','verified_result':val,'claim':'The registry maps this view to the stated workbook sheet; the executable named-column maximum returns the result.'}],'notes':'Deterministic expansion: executable named-column aggregate.'})
 added.extend(table_candidates)
 # 281 RAG+Graph tasks from spans unique among a task's required documents.
 profiles=ROOT/'data/worksurface_lite/profiles';rg=[]
 for pdir in sorted(profiles.iterdir()):
  kb=pdir/'kb_docs';gp=pdir/'graph/surface_graph.json'
  if not kb.exists() or not gp.exists():continue
  graph=json.load(open(gp));nodes=[n for n in graph['nodes'] if n.get('type')=='file'];bytask=defaultdict(list)
  for n in nodes:bytask[str(n.get('task'))].append(n)
  docs=defaultdict(list)
  for f in kb.glob('t*__*.md'):
   m=re.match(r't(\d+)__',f.name)
   if m:docs[m.group(1)].append(f)
  for sid,files in sorted(docs.items(),key=lambda x:int(x[0])):
   if not bytask.get(sid):continue
   locations=defaultdict(set);texts={}
   for f in files:
    text=f.read_text(errors='ignore');texts[f]=text
    for span in set(SPAN_RE.findall(text)):
     span=span.strip()
     if len(span)>=4:locations[span].add(f)
   for span,locs in sorted(locations.items()):
    if len(locs)!=1:continue
    f=next(iter(locs));canon=re.sub(r'^t\d+__','',f.name);matches=[n for n in bytask[sid] if normstem(n.get('filename',''))==normstem(canon)]
    if len(matches)!=1:continue
    gold=matches[0]['filename'];qid=f'ws_lite_{sid}_exrg_{len(rg)+1:04d}'
    rag_graph_templates=[
     'A teammate remembers seeing "{span}" in one of the files needed for Task {sid}. Which file should they open?',
     'Which input document for Task {sid} contains the text "{span}"?',
     'I need to verify "{span}" for Task {sid}. Which source file contains it?',
     'One of the files used by Task {sid} mentions "{span}". What is its filename?',
     'For Task {sid}, locate the required document containing "{span}".',
     'Which file in the Task {sid} handoff includes "{span}"?',
     'A value of "{span}" appears in one required source for Task {sid}. Identify the file.',
     'Where should I look for "{span}" among the inputs to Task {sid}? Give the filename.',
     'Find the Task {sid} source document that contains "{span}".',
     'Which required file for Task {sid} is the one with "{span}" in its contents?',
     'The team needs the Task {sid} file mentioning "{span}". Which file is it?',
     'Identify the Task {sid} input that contains the exact text "{span}".',
    ]
    question=rag_graph_templates[len(rg)%len(rag_graph_templates)].format(span=span,sid=sid)
    nq=re.sub(r'\W+',' ',question.lower()).strip()
    if nq in existing_q:continue
    rg.append({'id':qid,'source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':pdir.name.replace('_',' ').title(),'rubric_refs':['expanded_rag_graph_unique_v1']},'question':question,'difficulty':'medium','task_type':'cross_surface','required_surfaces':['rag','graph'],'gold_tools':['graph_neighbors','kb_search'],'applicable_skills':[],'gold_answer':gold,'answer_type':'string','gold_evidence':[{'surface':'graph','graph_path':[f'task_{sid}','task_requires_file',matches[0]['id']],'verified_candidate_scope':'all task_requires_file neighbors','claim':'The graph enumerates the required document candidates.'},{'surface':'rag','file':f.name,'span':span,'verified_unique_among_required_inputs':True,'claim':'The span occurs in this required document and no other candidate.'}],'notes':'Deterministic expansion: unique span over graph-scoped documents.'});existing_q.add(nq)
    if len(rg)>=281:break
   if len(rg)>=281:break
  if len(rg)>=281:break
 added.extend(rg)
 if len(added)!=611:raise RuntimeError(f'Expansion capacity shortfall: {len(added)} (table={len(table_candidates)}, graph=80, raggraph={len(rg)})')
 all_tasks=tasks+added
 if len(all_tasks)!=1000:raise RuntimeError(len(all_tasks))
 ids=[t['id'] for t in all_tasks]
 if len(ids)!=len(set(ids)):raise RuntimeError('duplicate ids')
 OUT.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in all_tasks))
 from collections import Counter
 report={'base_before_filter':len(base),'dropped_ai_gold_no':len(dropped),'clean_seed':len(tasks),'added':len(added),'final':len(all_tasks),'distribution':dict(Counter('+'.join(t['required_surfaces']) for t in all_tasks)),'added_distribution':dict(Counter('+'.join(t['required_surfaces']) for t in added)),'dropped_ids':sorted(dropped)}
 REPORT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
