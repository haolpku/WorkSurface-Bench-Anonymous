#!/usr/bin/env python3
"""Use GPT-5.5 to rewrite GPT-flagged questions without changing task semantics."""
import argparse,concurrent.futures,json,os,re,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS=ROOT/'data/worksurface_lite/tasks/tasks_balanced_1000_candidate.jsonl'
VERDICTS=ROOT/'results/ai_precheck_natural_graph_v2/gpt-5.5_all_frozen1000_quality.jsonl'
CACHE=ROOT/'results/quality_rewrites_gpt55.jsonl'
SUBSET=ROOT/'data/worksurface_lite/tasks/tasks_balanced_1000_quality_recheck.jsonl'

def parse(text):
 dec=json.JSONDecoder()
 for m in re.finditer(r'\{',text or ''):
  try:
   o,_=dec.raw_decode(text[m.start():])
   if isinstance(o,dict):return o
  except json.JSONDecodeError:pass
 raise ValueError((text or '')[:200])

def call(task,verdict,base,key):
 system=("Rewrite one benchmark question into fluent, plausible workplace English. Preserve exactly the same gold answer and evidence semantics. "
         "Make it atomic and unambiguous. A tightly coupled filename/value/count response is allowed. Do not mention benchmark, surface, RAG, graph, retrieval, DuckDB, tools, or scoring. "
         "Do not add facts unsupported by the evidence. Keep ordinary workplace words such as file, document, workbook, sheet, rows, source, and assignment. "
         "If the gold is INSUFFICIENT_EVIDENCE, retain an explicit evidence-bounded question. Return JSON only.")
 payload={'question':task['question'],'gold_answer':task['gold_answer'],'required_information':task['gold_evidence'],
          'audit_reason':verdict.get('reason'),'suggested_repair':verdict.get('repair')}
 body=json.dumps({'model':'gpt-5.5','messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)+'\nReturn {"question":"..."} only.'}],
                  'temperature':0,'max_tokens':1200}).encode()
 req=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=body,headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as resp:data=json.load(resp)
 out=parse(data['choices'][0]['message']['content']);q=str(out.get('question','')).strip()
 if len(q)<12:raise ValueError('short rewrite')
 return {'id':task['id'],'question':q}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--concurrency',type=int,default=10);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 key=os.getenv('WSB_API_KEY');base=os.getenv('WSB_API_BASE')
 if not key or not base:raise SystemExit('WSB_API_BASE and WSB_API_KEY required')
 tasks={t['id']:t for t in map(json.loads,TASKS.read_text().splitlines())}
 verdicts={r['id']:r for r in map(json.loads,VERDICTS.read_text().splitlines())}
 flagged={i:r for i,r in verdicts.items() if r.get('question_natural')!='Yes' or r.get('atomic_unambiguous')!='Yes'}
 done={}
 if args.resume and CACHE.exists():
  for r in map(json.loads,CACHE.read_text().splitlines()):
   if not r.get('error'):done[r['id']]=r
 todo=[i for i in flagged if i not in done]
 with CACHE.open('a') as h,concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
  fs={pool.submit(call,tasks[i],flagged[i],base,key):i for i in todo}
  for n,f in enumerate(concurrent.futures.as_completed(fs),1):
   i=fs[f]
   try:r=f.result()
   except Exception as e:r={'id':i,'error':repr(e)}
   h.write(json.dumps(r,ensure_ascii=False)+'\n');h.flush()
   if n%20==0:print(f'{n}/{len(todo)}',flush=True)
 latest={r['id']:r for r in map(json.loads,CACHE.read_text().splitlines())}
 changed=[]
 for i in flagged:
  r=latest.get(i,{})
  if r.get('question'):
   tasks[i]['question']=r['question'];changed.append(tasks[i])
 ordered=[]
 for line in TASKS.read_text().splitlines():ordered.append(tasks[json.loads(line)['id']])
 if len({t['question'] for t in ordered})!=len(ordered):raise RuntimeError('duplicate rewritten questions')
 TASKS.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in ordered))
 SUBSET.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in changed))
 print(json.dumps({'flagged':len(flagged),'rewritten':len(changed),'errors':len(flagged)-len(changed),'subset':str(SUBSET)},indent=2))

if __name__=='__main__':main()
