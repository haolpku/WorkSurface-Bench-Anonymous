#!/usr/bin/env python3
"""Batch-screen GPT-5.5 strict items with a second model."""
import argparse,concurrent.futures,json,os,re,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASKS=ROOT/'data/worksurface_lite/tasks/tasks_gpt55_strict_1465.jsonl'
OUTDIR=ROOT/'results/ai_precheck_natural_graph_v2'
SYSTEM="""Audit each benchmark item independently. Return valid JSON only.
For answerable, gold_correct, question_natural, and atomic_unambiguous use Yes, No, or Unsure.
For leakage_cue use None, Surface implied, Surface named, or Unsure.
Natural means fluent plausible workplace wording, not benchmark/meta wording or an awkward template. Atomic means a uniquely scoped deterministic request; tightly coupled fields are allowed. Ordinary words such as file, document, workbook, worksheet, column, and rows are not internal-surface leakage. Surface named applies only to graph, RAG, retrieval surface, DuckDB, internal tools, or equivalent mechanisms. Judge evidence sufficiency and gold correctness separately.
Return {"annotations":[{"id":"...","answerable":"Yes|No|Unsure","gold_correct":"Yes|No|Unsure","question_natural":"Yes|No|Unsure","atomic_unambiguous":"Yes|No|Unsure","leakage_cue":"None|Surface implied|Surface named|Unsure","reason":"brief"}]} only."""
def parse(s):
 d=json.JSONDecoder()
 for m in re.finditer(r'\{',s or ''):
  try:
   o,_=d.raw_decode(s[m.start():])
   if isinstance(o,dict):return o
  except json.JSONDecodeError:pass
 raise ValueError((s or '')[:200])
def call(batch,model,base,key):
 rows=[{'id':t['id'],'question':t['question'],'required_surfaces':t['required_surfaces'],'gold_answer':t['gold_answer'],'gold_evidence':t['gold_evidence']} for t in batch]
 body=json.dumps({'model':model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps({'items':rows},ensure_ascii=False)}],'temperature':0,'max_tokens':8192}).encode()
 req=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=body,headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as resp:data=json.load(resp)
 obj=parse(data['choices'][0]['message']['content']);ans=obj.get('annotations',[]);by={x.get('id'):x for x in ans}
 if any(t['id'] not in by for t in batch):raise ValueError(f'missing annotations {len(by)}/{len(batch)}')
 return [dict(by[t['id']],model=model) for t in batch]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--batch-size',type=int,default=5);ap.add_argument('--concurrency',type=int,default=5);args=ap.parse_args()
 key=os.getenv('WSB_API_KEY');base=os.getenv('WSB_API_BASE')
 if not key or not base:raise SystemExit('WSB_API_BASE and WSB_API_KEY required')
 tasks=list(map(json.loads,TASKS.read_text().splitlines()));safe=args.model.replace('/','-').replace(':','-');out=OUTDIR/f'{safe}_strict1465_batched.jsonl'
 done={}
 # Reuse valid per-item judgments from the earlier attempt.
 old=OUTDIR/f'{safe}_all_strict1465_quality.jsonl'
 for p in (old,out):
  if p.exists():
   for r in map(json.loads,p.read_text().splitlines()):
    if not r.get('error') and r.get('question_natural'):done[r['id']]=r
 todo=[t for t in tasks if t['id'] not in done];batches=[todo[i:i+args.batch_size] for i in range(0,len(todo),args.batch_size)]
 with out.open('a') as h,concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
  fs={pool.submit(call,b,args.model,base,key):b for b in batches}
  for n,f in enumerate(concurrent.futures.as_completed(fs),1):
   b=fs[f]
   try:rows=f.result()
   except Exception as e:rows=[{'id':t['id'],'model':args.model,'error':repr(e)} for t in b]
   for r in rows:h.write(json.dumps(r,ensure_ascii=False)+'\n')
   h.flush()
   if n%10==0:print(f'{n}/{len(batches)} batches',flush=True)
 latest=dict(done)
 for r in map(json.loads,out.read_text().splitlines()):
  if not r.get('error'):latest[r['id']]=r
 out.write_text(''.join(json.dumps(latest[t['id']],ensure_ascii=False)+'\n' for t in tasks if t['id'] in latest))
 print(json.dumps({'model':args.model,'complete':len(latest),'missing':len(tasks)-len(latest)},indent=2))
if __name__=='__main__':main()
