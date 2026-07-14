#!/usr/bin/env python3
"""Merge the six repaired GPT-5.5 verdicts and summarize expansion screening."""
import json,re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'results/ai_precheck_natural_graph_v2'
def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
g4=rows(DIR/'gpt-4o-mini_all_expanded611.jsonl')
g5=rows(DIR/'gpt-5.5_all_expanded611.jsonl')
fix={r['id']:r for r in rows(DIR/'gpt-5.5_all_expanded6fix.jsonl')}
g5=[fix.get(r['id'],r) for r in g5]
(DIR/'gpt-5.5_all_expanded611_final.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in g5))
tasks=[json.loads(x) for x in (ROOT/'data/worksurface_lite/tasks/tasks_expanded_1000_candidate.jsonl').read_text().splitlines()]
added=[t for t in tasks if re.search(r'_(exg|ext|exrg)_',t['id'])]
def summary(rs):return {'n':len(rs),'answerable_yes':sum(r.get('answerable')=='Yes' for r in rs),'gold_yes':sum(r.get('gold_correct')=='Yes' for r in rs),'errors':sum(bool(r.get('error')) for r in rs)}
qnorm=[re.sub(r'\d+|"[^"]+"','<x>',t['question'].lower()) for t in added]
report={'models':{'gpt-4o-mini':summary(g4),'gpt-5.5':summary(g5)},'dual_pass_ids':sum(a.get('answerable')=='Yes' and a.get('gold_correct')=='Yes' and b.get('answerable')=='Yes' and b.get('gold_correct')=='Yes' for a,b in zip(g4,g5)),'added_distribution':dict(Counter('+'.join(t['required_surfaces']) for t in added)),'exact_duplicate_questions':len(added)-len({t['question'] for t in added}),'normalized_template_counts':Counter(qnorm).most_common(10)}
(ROOT/'results/expanded_1000_ai_screen.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
