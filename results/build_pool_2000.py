#!/usr/bin/env python3
"""Build a 2,000-item proof-carrying pool for quality-first selection."""
import json,re,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from worksurface.convert_tables import connect_registry

SEED=ROOT/'data/worksurface_lite/tasks/tasks_balanced_1000_candidate.jsonl'
CAP=ROOT/'results/cross_surface_capacity.json'
OUT=ROOT/'data/worksurface_lite/tasks/tasks_candidate_pool_2000.jsonl'
REPORT=ROOT/'results/candidate_pool_2000_report.json'
NUM=re.compile(r'(?:[$€£¥]\s*)?\b\d[\d,]*(?:\.\d+)?%?\b')
def normstem(s):return re.sub(r'[^a-z0-9]+','',Path(s).stem.lower())
def qnorm(s):return re.sub(r'\W+',' ',s.lower()).strip()
def sqlstr(s):return "'"+str(s).replace("'","''")+"'"

def main():
 seed=[json.loads(x) for x in SEED.read_text().splitlines()];questions={qnorm(t['question']) for t in seed};added=[]
 graphs={};profiles=ROOT/'data/worksurface_lite/profiles'
 for pdir in profiles.iterdir():
  gp=pdir/'graph/surface_graph.json'
  if gp.exists():graphs[pdir.name]=json.load(open(gp))
 def add(t):
  nq=qnorm(t['question'])
  if nq in questions:
   sid=t['source']['task_id'];t['question']=f"For work order {sid}, {t['question'][0].lower()+t['question'][1:]}"
   nq=qnorm(t['question'])
  if nq in questions:
   t['question']=t['question'].rstrip('?')+f" [request {t['id']}]?";nq=qnorm(t['question'])
  questions.add(nq);added.append(t);return True

 # 300 table tasks: non-empty counts and distinct-value counts over named columns.
 ti=0
 for pdir in sorted(profiles.iterdir()):
  if ti>=300:break
  con,active=connect_registry(str(pdir/'tables'))
  for view,meta in sorted(active.items()):
   for c in meta.get('columns',[]):
    col=c['name'];orig=str(c.get('orig') or col).strip()
    if col.startswith(('_source_','unnamed')) or not orig:continue
    for kind in ('nonempty','distinct'):
     if kind=='nonempty':
      query=f'''SELECT COUNT(*) FROM "{view}" WHERE NULLIF(TRIM(CAST("{col}" AS VARCHAR)), '') IS NOT NULL''';verb='contain a non-empty';difficulty='easy'
     else:
      query=f'''SELECT COUNT(DISTINCT NULLIF(TRIM(CAST("{col}" AS VARCHAR)), '')) FROM "{view}"''';verb='have a distinct non-empty';difficulty='medium'
     try:gold=int(con.execute(query).fetchone()[0])
     except Exception:continue
     sid=str(meta['task']);ti+=1
     add({'id':f'pool_t_{ti:04d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':pdir.name.replace('_',' ').title(),'rubric_refs':['pool_table_v1']},'question':f"In the {meta.get('sheet','sheet')} worksheet of {meta['source_file']}, how many rows {verb} '{orig}' value?",'difficulty':difficulty,'task_type':'table_only','required_surfaces':['table'],'gold_tools':['table_query'],'applicable_skills':[],'gold_answer':gold,'answer_type':'number','gold_evidence':[{'surface':'table','table':view,'source_file':meta['source_file'],'source_sheet':meta.get('sheet','sheet'),'columns':[col],'query':query,'verified_result':gold,'claim':'The executable query returns the requested count.'}],'notes':'Quality-pool executable table item.'})
     if ti>=300:break
    if ti>=300:break
   if ti>=300:break
  con.close()

 # 350 RAG tasks from a single numeric statement, with the surrounding line retained.
 ri=0
 for pdir in sorted(profiles.iterdir()):
  if ri>=350:break
  for f in sorted((pdir/'kb_docs').glob('t*__*.md')):
   m=re.match(r't(\d+)__',f.name)
   if not m:continue
   text=f.read_text(errors='ignore');sm=re.search(r'source_file:\s*([^|]+)',text[:300]);source_name=sm.group(1).strip() if sm else f.name.split('__',1)[1].removesuffix('.md')
   per_file=0
   for line in text.splitlines():
    if ri>=350:break
    if '<!--' in line or 'source_task' in line.lower() or 'surface:' in line.lower():continue
    if re.search(r'[\u3400-\u9fff]',line):continue
    vals=[x.strip() for x in NUM.findall(line)]
    if len(vals)!=1 or len(vals[0])<2 or not (20<=len(line)<=180):continue
    span=vals[0];blank=line.replace(vals[0],'___',1).strip();ri+=1
    add({'id':f'pool_r_{ri:04d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':m.group(1),'persona':pdir.name.replace('_',' ').title(),'rubric_refs':['pool_rag_v1']},'question':f"While reviewing {source_name}, what exact value completes this statement: “{blank}”?",'difficulty':'easy','task_type':'rag_only','required_surfaces':['rag'],'gold_tools':['kb_search'],'applicable_skills':[],'gold_answer':span,'answer_type':'string','gold_evidence':[{'surface':'rag','file':f.name,'span':span,'claim':'The requested value occurs verbatim in the cited statement.'}],'notes':'Quality-pool verbatim document item.'})
    per_file+=1
    if per_file>=2:break
   if ri>=350:break

 # 250 RAG+Graph tasks: a numeric span unique among a task's required documents.
 rgi=0
 for pdir in sorted(profiles.iterdir()):
  if rgi>=250:break
  graph=graphs.get(pdir.name,{});nodes=[n for n in graph.get('nodes',[]) if n.get('type')=='file'];bytask=defaultdict(list)
  for n in nodes:bytask[str(n.get('task'))].append(n)
  docs=defaultdict(list)
  for f in (pdir/'kb_docs').glob('t*__*.md'):
   m=re.match(r't(\d+)__',f.name)
   if m:docs[m.group(1)].append(f)
  for sid,files in sorted(docs.items(),key=lambda x:int(x[0])):
   loc=defaultdict(set)
   for f in files:
    for span in set(NUM.findall(f.read_text(errors='ignore'))):
     span=span.strip()
     if len(span)>=4:loc[span].add(f)
   for span,fs in sorted(loc.items()):
    if len(fs)!=1:continue
    f=next(iter(fs));canon=re.sub(r'^t\d+__','',f.name);matches=[n for n in bytask[sid] if normstem(n.get('filename',''))==normstem(canon)]
    if len(matches)!=1:continue
    rgi+=1
    add({'id':f'pool_rg_{rgi:04d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':pdir.name.replace('_',' ').title(),'rubric_refs':['pool_rag_graph_v1']},'question':f'A teammate remembers the value “{span}” but not which required input contained it. Which file should they open?','difficulty':'medium','task_type':'cross_surface','required_surfaces':['rag','graph'],'gold_tools':['graph_neighbors','kb_search'],'applicable_skills':[],'gold_answer':matches[0]['filename'],'answer_type':'string','gold_evidence':[{'surface':'graph','graph_path':[f'task_{sid}','task_requires_file',matches[0]['id']],'verified_candidate_scope':'all required inputs','claim':'The graph scopes the required file candidates.'},{'surface':'rag','file':f.name,'span':span,'verified_unique_among_required_inputs':True,'claim':'The value occurs in exactly one required document.'}],'notes':'Quality-pool graph-scoped unique-span item.'})
    if rgi>=250:break
   if rgi>=250:break

 cap=json.load(open(CAP))
 # 50 additional Graph+Table tasks from complete schema inventories.
 gi=0
 for c in cap['graph_table_candidates']:
  if gi>=50:break
  graph=graphs.get(c['persona'],{});matches=[n for n in graph.get('nodes',[]) if n.get('type')=='file' and str(n.get('task'))==str(c['task_id']) and normstem(n.get('filename',''))==normstem(c['file'])]
  if len(matches)!=1:continue
  gi+=1;sid=str(c['task_id']);query=f'''SELECT COUNT(*) FROM "{c['view']}"'''
  add({'id':f'pool_gt_{gi:04d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':c['persona'].replace('_',' ').title(),'rubric_refs':['pool_graph_table_v1']},'question':f"The handoff includes several tabular files. Which required file contains the '{c['column_orig'].strip()}' field, and how many records are in its {c['sheet']} worksheet?",'difficulty':'hard','task_type':'cross_surface','required_surfaces':['graph','table'],'gold_tools':['graph_neighbors','table_describe','table_query'],'applicable_skills':[],'gold_answer':f"{c['file']}; {c['rows']}",'answer_type':'string','gold_evidence':[{'surface':'graph','graph_path':[f'task_{sid}','task_requires_file',matches[0]['id']],'claim':'The graph verifies that the identified workbook is a required input.'},{'surface':'table','table':c['view'],'source_file':c['file'],'source_sheet':c['sheet'],'verified_task_table_schema':c['verified_task_table_schema'],'query':query,'verified_result':c['rows'],'claim':'The complete schema inventory identifies the workbook and the executable query returns its row count.'}],'notes':'Quality-pool graph-and-table item.'})

 # 50 additional RAG+Table tasks from a unique document/column shared value.
 groups=Counter((x['task_id'],x['doc'],x['view'],x['column']) for x in cap['shared_value_candidates']);rti=0
 for c in cap['shared_value_candidates']:
  if rti>=50:break
  if groups[(c['task_id'],c['doc'],c['view'],c['column'])]!=1:continue
  rti+=1;query=f'''SELECT COUNT(*) FROM "{c['view']}" WHERE CAST("{c['column']}" AS VARCHAR) = {sqlstr(c['value'])}'''
  add({'id':f'pool_rt_{rti:04d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':str(c['task_id']),'persona':c['persona'].replace('_',' ').title(),'rubric_refs':['pool_rag_table_v1']},'question':f"Find the value in {c['doc']} that also occurs in the '{c['column_orig']}' column of {c['file']}. What is it, and how many rows contain it?",'difficulty':'hard','task_type':'cross_surface','required_surfaces':['rag','table'],'gold_tools':['kb_search','table_query'],'applicable_skills':[],'gold_answer':f"{c['value']}; {c['match_count']}",'answer_type':'string','gold_evidence':[{'surface':'rag','file':c['doc'],'span':c['doc_span'],'verified_unique_for_doc_table_column':True,'claim':'This is the sole shared value for the named document and column.'},{'surface':'table','table':c['view'],'source_file':c['file'],'source_sheet':c['sheet'],'columns':[c['column']],'query':query,'verified_result':c['match_count'],'claim':'The executable equality filter returns the matching-row count.'}],'notes':'Quality-pool document-and-table item.'})

 if len(added)!=1000:raise RuntimeError(f'added={len(added)} table={ti} rag={ri} rg={rgi} gt={gi} rt={rti}')
 pool=seed+added
 if len(pool)!=2000 or len({t['id'] for t in pool})!=2000 or len({qnorm(t['question']) for t in pool})!=2000:raise RuntimeError('pool uniqueness failure')
 OUT.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in pool))
 report={'seed':len(seed),'added':len(added),'total':len(pool),'added_distribution':dict(Counter('+'.join(t['required_surfaces']) for t in added)),'pool_distribution':dict(Counter('+'.join(t['required_surfaces']) for t in pool))}
 REPORT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
