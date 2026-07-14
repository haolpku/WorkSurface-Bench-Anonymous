#!/usr/bin/env python3
"""Select a diverse, strictly GPT-5.5-screened final 1,000-task candidate."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
POOL=ROOT/'data/worksurface_lite/tasks/tasks_candidate_pool_2000.jsonl'
OUT=ROOT/'data/worksurface_lite/tasks/tasks_final_1000.jsonl'
STRICT=ROOT/'data/worksurface_lite/tasks/tasks_gpt55_strict_1465.jsonl'
REPORT=ROOT/'results/final_1000_selection_report.json'
VERDICTS=[
 ROOT/'results/ai_precheck_natural_graph_v2/gpt-5.5_all_frozen1000_quality.jsonl',
 ROOT/'results/ai_precheck_natural_graph_v2/gpt-5.5_all_pool2000new_quality.jsonl',
 ROOT/'results/ai_precheck_natural_graph_v2/gpt-5.5_all_pool2000ragfix_quality.jsonl']
QUOTAS={'graph':150,'table':240,'rag':160,'graph+table':100,'rag+graph':220,'rag+table':115,'rag+graph+table':15}
def stable(s):return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)
def strict(r):return all(r.get(k)=='Yes' for k in ('answerable','gold_correct','question_natural','atomic_unambiguous')) and r.get('leakage_cue')=='None'
def main():
 tasks={t['id']:t for t in map(json.loads,POOL.read_text().splitlines())};verdict={}
 for p in VERDICTS:
  for r in map(json.loads,p.read_text().splitlines()):verdict[r['id']]=r
 eligible=[t for t in tasks.values() if strict(verdict.get(t['id'],{}))]
 STRICT.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in sorted(eligible,key=lambda x:x['id'])))
 bycombo=defaultdict(list)
 for t in eligible:bycombo['+'.join(t['required_surfaces'])].append(t)
 selected=[];persona=Counter();source=Counter()
 # Select scarce/cross-surface strata first. Within each stratum, greedily
 # reduce persona and source-task concentration, with a stable tie-breaker.
 for combo in ('rag+graph+table','graph+table','rag+table','graph','rag','table','rag+graph'):
  pool=list(bycombo[combo]);need=QUOTAS[combo]
  for _ in range(need):
   if not pool:raise RuntimeError(f'insufficient {combo}')
   t=min(pool,key=lambda x:(persona[x['source']['persona']],source[str(x['source']['task_id'])],stable(x['id'])))
   pool.remove(t);selected.append(t);persona[t['source']['persona']]+=1;source[str(t['source']['task_id'])]+=1
 if len(selected)!=1000:raise RuntimeError(len(selected))
 selected.sort(key=lambda t:t['id'])
 OUT.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in selected))
 report={'pool':len(tasks),'strict_eligible':len(eligible),'selected':len(selected),'distribution':dict(Counter('+'.join(t['required_surfaces']) for t in selected)),'persona':dict(persona),'source_task_minmax':[min(source.values()),max(source.values())],'unique_ids':len({t['id'] for t in selected}),'unique_questions':len({t['question'] for t in selected})}
 REPORT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
