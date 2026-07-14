#!/usr/bin/env python3
"""Replace excess RAG+Graph with verified GT, RT, and tri-surface tasks."""
import json,re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/worksurface_lite/tasks/tasks_expanded_1000_candidate.jsonl'
CAP=ROOT/'results/cross_surface_capacity.json'
OUT=ROOT/'data/worksurface_lite/tasks/tasks_balanced_1000_candidate.jsonl'
def normstem(s):return re.sub(r'[^a-z0-9]+','',Path(s).stem.lower())
def sqlstr(s):return "'"+str(s).replace("'","''")+"'"
def main():
 tasks=[json.loads(x) for x in SRC.read_text().splitlines()];cap=json.load(open(CAP))
 # Preserve the naturally large RAG+Graph pool while correcting its earlier
 # dominance: replace 185 (not 235) items with 80 GT, 90 RT, and 15 tri tasks.
 remove=[t for t in tasks if '_exrg_' in t['id']][:185];remove_ids={t['id'] for t in remove};kept=[t for t in tasks if t['id'] not in remove_ids]
 graphs={}
 for pdir in (ROOT/'data/worksurface_lite/profiles').iterdir():
  gp=pdir/'graph/surface_graph.json'
  if gp.exists():graphs[pdir.name]=json.load(open(gp))
 def file_node(persona,tid,name):
  for n in graphs.get(persona,{}).get('nodes',[]):
   if n.get('type')=='file' and str(n.get('task'))==str(tid) and normstem(n.get('filename',''))==normstem(name):return n
  return None
 added=[]
 # 80 Graph+Table: graph proves required file; table schema identifies unique column and query computes rows.
 for c in cap['graph_table_candidates']:
  if len([x for x in added if x['required_surfaces']==['graph','table']])>=80:break
  node=file_node(c['persona'],c['task_id'],c['file'])
  if not node:continue
  sid=c['task_id'];gold=f"{c['file']}; {c['rows']}";required_tabular_files=sorted({x['source_file'] for x in c['verified_task_table_schema']})
  added.append({'id':f"ws_lite_{sid}_xgt_{len(added)+1:04d}",'source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':c['persona'].replace('_',' ').title(),'rubric_refs':['balanced_graph_table_v1']},'question':f"Among the files required for Task {sid}, which tabular input contains the column '{c['column_orig']}', and how many data rows does its {c['sheet']} sheet contain?",'difficulty':'hard','task_type':'cross_surface','required_surfaces':['graph','table'],'gold_tools':['graph_neighbors','table_describe','table_query'],'applicable_skills':[],'gold_answer':gold,'answer_type':'string','gold_evidence':[{'surface':'graph','graph_path':[f'task_{sid}','task_requires_file',node['id']],'verified_required_tabular_inputs':required_tabular_files,'claim':'Graph enumeration verifies this complete set of required tabular files and the identified workbook is in it.'},{'surface':'table','table':c['view'],'source_file':c['file'],'source_sheet':c['sheet'],'unique_column':{'normalized':c['column'],'original':c['column_orig']},'verified_task_table_schema':c['verified_task_table_schema'],'query':f'''SELECT COUNT(*) FROM "{c['view']}"''','verified_result':c['rows'],'claim':'The complete task-table schema inventory proves this is the only required task table with the named original/normalized column; the query returns its row count.'}],'notes':'Balanced expansion: required-file identification plus executable table count.'})
 links=cap['shared_value_candidates'];docgroups=Counter((x['task_id'],x['doc'],x['view'],x['column']) for x in links);taskgroups=Counter((x['task_id'],x['view'],x['column']) for x in links)
 # 90 RAG+Table: document named, and exactly one shared value for this doc/table/column tuple.
 rt=[x for x in links if docgroups[(x['task_id'],x['doc'],x['view'],x['column'])]==1]
 for i,c in enumerate(rt[:90]):
  sid=c['task_id'];gold=f"{c['value']}; {c['match_count']}"
  added.append({'id':f'ws_lite_{sid}_xrt_{i+1:04d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':c['persona'].replace('_',' ').title(),'rubric_refs':['balanced_rag_table_v1']},'question':f"In {c['doc']}, find the value that also appears under '{c['column_orig']}' in the {c['sheet']} sheet of {c['file']}. What is the value, and how many rows match it?",'difficulty':'hard','task_type':'cross_surface','required_surfaces':['rag','table'],'gold_tools':['kb_search','table_query'],'applicable_skills':[],'gold_answer':gold,'answer_type':'string','gold_evidence':[{'surface':'rag','file':c['doc'],'span':c.get('doc_span',c['value']),'verified_unique_for_doc_table_column':True,'claim':'This is the only value shared by the named document and table column.'},{'surface':'table','table':c['view'],'source_file':c['file'],'source_sheet':c['sheet'],'columns':[c['column']],'query':f'''SELECT COUNT(*) FROM "{c['view']}" WHERE CAST("{c['column']}" AS VARCHAR) = {sqlstr(c['value'])}''','verified_result':c['match_count'],'claim':'Executable equality filter returns the matching-row count.'}],'notes':'Balanced expansion: verbatim document value drives executable table filter.'})
 # 15 tri-surface: document is not named; graph scopes required docs, RAG finds the sole shared value, table counts it.
 tri=[x for x in links if taskgroups[(x['task_id'],x['view'],x['column'])]==1][:15]
 for i,c in enumerate(tri):
  sid=c['task_id'];node=file_node(c['persona'],sid,re.sub(r'^t\d+__','',c['doc']))
  if not node:continue
  gold=f"{node['filename']}; {c['value']}; {c['match_count']}"
  added.append({'id':f'ws_lite_{sid}_xtri_{i+1:03d}','source':{'benchmark':'Workspace-Bench-Lite','task_id':sid,'persona':c['persona'].replace('_',' ').title(),'rubric_refs':['balanced_tri_surface_v1']},'question':f"Among the documents required for Task {sid}, identify the file containing the value that also appears under '{c['column_orig']}' in {c['file']}. Report the document, the value, and how many table rows match it.",'difficulty':'hard','task_type':'cross_surface','required_surfaces':['rag','graph','table'],'gold_tools':['graph_neighbors','kb_search','table_query'],'applicable_skills':[],'gold_answer':gold,'answer_type':'string','gold_evidence':[{'surface':'graph','graph_path':[f'task_{sid}','task_requires_file',node['id']],'canonical_rag_file':c['doc'],'canonicalization':'The original required file is converted to this canonical Markdown document for RAG.','verified_candidate_scope':'all task_requires_file documents','claim':'Graph scopes the candidate documents, verifies the original file is required, and maps it to the canonical RAG document.'},{'surface':'rag','file':c['doc'],'span':c.get('doc_span',c['value']),'verified_unique_among_required_inputs':True,'claim':'RAG finds the sole task-document value shared with the target table column.'},{'surface':'table','table':c['view'],'source_file':c['file'],'source_sheet':c['sheet'],'columns':[c['column']],'query':f'''SELECT COUNT(*) FROM "{c['view']}" WHERE CAST("{c['column']}" AS VARCHAR) = {sqlstr(c['value'])}''','verified_result':c['match_count'],'claim':'Executable equality filter returns the matching-row count.'}],'notes':'Balanced expansion: all three surfaces are proof-carrying and necessary.'})
 counts=Counter('+'.join(t['required_surfaces']) for t in added)
 if counts['graph+table']!=80 or counts['rag+table']!=90 or counts['rag+graph+table']!=15:raise RuntimeError(counts)
 final=kept+added
 if len(final)!=1000:raise RuntimeError(len(final))
 seen_questions=set()
 for task in final:
  if task['question'] in seen_questions:
   task['question']=f"For Task {task['source']['task_id']}: {task['question']}"
  seen_questions.add(task['question'])
 OUT.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in final))
 report={'removed_rag_graph':len(remove),'added':dict(counts),'final':len(final),'distribution':dict(Counter('+'.join(t['required_surfaces']) for t in final)),'exact_duplicate_questions':len(final)-len({t['question'] for t in final})}
 (ROOT/'results/balanced_1000_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
