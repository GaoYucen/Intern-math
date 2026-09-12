"""9 targeted public regression questions; identical code replayed in both workers.
No official submission or hidden data. Local responses stored for public diagnostic only.
Failure-selected, NOT a holdout or overall score estimate.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib,json,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import r2_agent
import math_tools
import _math_tools_before as before
from llm_client import InternChatClient
from user_agent import ReasoningAgent
IDS=[47,177,192,200,214,241,250,263,272]
OUT=ROOT/'reports/r21_tool_diagnostic'
TLS=threading.local()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)

def compared_tool(code,timeout=12):
    new=math_tools.run_math(code,timeout)
    old=before.run_math(code,timeout)
    TLS.tools.append({'code':code,'sha256':hashlib.sha256(code.encode()).hexdigest(),'before':old,'after':new})
    return new

class RecordClient:
    def __init__(self):
        self.inner=InternChatClient();self.calls=[]
    def chat(self,*args,**kwargs):
        t=time.monotonic()
        try:
            result=self.inner.chat(*args,**kwargs)
            meta=self.inner.get_last_response_meta()
            self.calls.append({'status':'completed','response':result,'seconds':time.monotonic()-t,
                'thinking_mode':kwargs.get('thinking_mode'),'max_tokens':kwargs.get('max_tokens'),
                'finish_reason':meta.get('finish_reason'),'usage':meta.get('usage'),
                'http_attempt_count':meta.get('http_attempt_count')})
            return result
        except Exception as exc:
            self.calls.append({'status':'error','error_type':type(exc).__name__,'seconds':time.monotonic()-t})
            raise

def solve(row):
    TLS.tools=[]
    client=RecordClient();agent=ReasoningAgent(client);started=time.monotonic()
    try:
        result=agent.solve(row['problem'],{'idx':row['idx']})
        record={'idx':row['idx'],'status':'success',**result}
    except Exception as exc:
        record={'idx':row['idx'],'status':'error','error_type':type(exc).__name__,'final_response':''}
    record.update(problem=row['problem'],source_reference=row['answer'],calls=client.calls,
        tool_replay=TLS.tools,seconds=time.monotonic()-started,
        explicit_final=r2_agent.complete(record['final_response']))
    save(OUT/'items'/f"{row['idx']}.json",record)
    print('DONE',row['idx'],'calls',len(client.calls),'tool_after',[v['after']['ok'] for v in TLS.tools],flush=True)
    return record

def main():
    data=ROOT/'data/benchmark_v1/gold.jsonl'
    gold={r['idx']:r for r in map(json.loads,data.read_text().splitlines())}
    manifest={'indices':IDS,'n':len(IDS),'gold_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
        'selection':'All nine questions that used the tool in prior hardened run 34625550536',
        'config':asdict(r2_agent.R2Config()),'model':'intern-s2-preview-397b',
        'policy':'Only math_tools.py differs; prompt, tokenizer/model selection, temperature and token budgets unchanged',
        'comparison':'Identical newly generated code is executed once in each old/new worker; new worker result feeds solver',
        'limitations':'Failure-selected diagnostic; historical solver comparison is NOT a randomized paired trial; covariance question 192 is ambiguous',
        'acceptance':'Review exact code failures and final answers; execution success is not mathematical correctness; no automatic promotion',
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['r2_agent.py','user_agent.py','math_tools.py']}}
    save(OUT/'manifest.json',manifest)
    r2_agent.run_math=compared_tool
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(solve,gold[i]) for i in IDS]
        for task in as_completed(tasks):rows.append(task.result())
    replay=[v for r in rows for v in r['tool_replay']]
    calls=[v for r in rows for v in r['calls']]
    summary={'n':len(rows),'success':sum(r['status']=='success' for r in rows),
        'explicit_final':sum(r['explicit_final'] for r in rows),'model_calls':len(calls),
        'truncated_calls':sum(c.get('finish_reason')=='length' for c in calls),
        'completion_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) for c in calls),
        'tool_runs':len(replay),'old_success':sum(v['before']['ok'] for v in replay),
        'new_success':sum(v['after']['ok'] for v in replay),
        'new_only_success':sum(v['after']['ok'] and not v['before']['ok'] for v in replay),
        'old_only_success':sum(v['before']['ok'] and not v['after']['ok'] for v in replay),
        'mean_seconds':sum(r['seconds'] for r in rows)/len(rows),'max_seconds':max(r['seconds'] for r in rows),
        'mathematical_accuracy':'Pending offline review; not inferred from explicit_final','official_submissions':0}
    save(OUT/'summary.json',summary);print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()
