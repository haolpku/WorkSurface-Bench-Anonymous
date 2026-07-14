#!/usr/bin/env python3
"""Freeze tasks passing strict quality checks under at least two of three models."""
import json,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/worksurface_lite/tasks/tasks_gpt55_strict_1465.jsonl'
OUT=ROOT/'data/worksurface_lite/tasks/tasks_final_1151.jsonl'
CORE=ROOT/'data/worksurface_lite/tasks/tasks_core_429.jsonl'
REPORT=ROOT/'results/final_multimodel_release_report.json'
D=ROOT/'results/ai_precheck_natural_graph_v2/deepseek-v4-pro_strict1465_batched.jsonl'
G=ROOT/'results/ai_precheck_natural_graph_v2/gemini-3.1-pro-preview_strict1465_batched.jsonl'
def strict(r):return all(r.get(k)=='Yes' for k in ('answerable','gold_correct','question_natural','atomic_unambiguous')) and r.get('leakage_cue')=='None'
def stats(rows):return {'n':len(rows),'distribution':dict(Counter('+'.join(t['required_surfaces']) for t in rows)),'persona':dict(Counter(t['source']['persona'] for t in rows)),'source_tasks':len({str(t['source']['task_id']) for t in rows}),'unique_ids':len({t['id'] for t in rows}),'unique_questions':len({t['question'] for t in rows})}
def main():
 tasks=list(map(json.loads,SRC.read_text().splitlines()));d={r['id']:r for r in map(json.loads,D.read_text().splitlines())};g={r['id']:r for r in map(json.loads,G.read_text().splitlines())}
 release=[];core=[]
 for t in tasks:
  votes=1+strict(d[t['id']])+strict(g[t['id']])
  t['quality_screen']={'strict_pass_votes':votes,'models':['gpt-5.5']+(['deepseek-v4-pro'] if strict(d[t['id']]) else [])+(['gemini-3.1-pro-preview'] if strict(g[t['id']]) else [])}
  if votes>=2:release.append(t)
  if votes==3:core.append(t)
 OUT.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in release));CORE.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in core))
 raw=OUT.read_bytes();report={'release':stats(release),'core':stats(core),'sha256':hashlib.sha256(raw).hexdigest(),'selection_rule':'strict pass on Answerable, Gold correct, Question natural, Atomic/unambiguous, and Leakage=None by at least two of GPT-5.5, DeepSeek-v4-pro, and Gemini-3.1-Pro-Preview; required-surface necessity remains a separate human-audit field'}
 REPORT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
