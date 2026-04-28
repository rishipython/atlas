"""Evaluator with baseline comparison (vendored local task_ref.py)."""
import concurrent.futures, importlib.util, time, traceback
from pathlib import Path
import numpy as np
from openevolve.evaluation_result import EvaluationResult

def _with_timeout(fn,args=(),timeout_seconds=120):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn,*args).result(timeout=timeout_seconds)

def _load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _measure(fn,problem,runs=3,warmup=1,timeout_seconds=120):
    for _ in range(warmup):
        try:_with_timeout(fn,(problem,),timeout_seconds)
        except Exception:pass
    times=[]; out=None
    for _ in range(runs):
        t0=time.perf_counter(); out=_with_timeout(fn,(problem,),timeout_seconds); t1=time.perf_counter()
        times.append((t1-t0)*1000.0)
    return float(np.min(times)), out

def evaluate(program_path, config=None):
    try:
        base=Path(__file__).parent
        ref=_load(base/'task_ref.py','task_ref')
        prog=_load(Path(program_path),'program')
        if not hasattr(prog,'run_solver'):
            return EvaluationResult(metrics={"correctness":0.0,"correctness_score":0.0,"performance_score":0.0,"combined_score":0.0,"speedup":0.0,"speedup_score":0.0},artifacts={"feedback":"Missing run_solver function"})
        task=getattr(ref,'SHA256Hashing')()
        num_trials,data_size,timeout_seconds,num_runs,warmup_runs=5,100,120,3,1
        corr=[]; speedups=[]; bt=[]; et=[]
        for trial in range(num_trials):
            problem=task.generate_problem(n=data_size, random_seed=trial)
            b,_=_measure(task.solve,problem,num_runs,warmup_runs,timeout_seconds)
            e,out=_measure(prog.run_solver,problem,num_runs,warmup_runs,timeout_seconds)
            ok=False
            try: ok=bool(task.is_solution(problem,out))
            except Exception: ok=False
            corr.append(1.0 if ok else 0.0); bt.append(b); et.append(e)
            if ok and e>0: speedups.append(b/e)
        avg=float(np.mean(corr)) if corr else 0.0
        ms=float(np.mean(speedups)) if speedups else 0.0
        combined=0.7*avg+0.3*min(ms,5.0)/5.0
        return EvaluationResult(metrics={"correctness":avg,"correctness_score":avg,"performance_score":float(np.mean([1.0/(1.0+t) for t in et])) if et else 0.0,"combined_score":float(combined),"speedup":ms,"speedup_score":ms},artifacts={"baseline_comparison":{"mean_speedup":ms,"baseline_times":bt,"evolved_times":et,"speedups":speedups}})
    except Exception as e:
        return EvaluationResult(metrics={"correctness":0.0,"correctness_score":0.0,"performance_score":0.0,"combined_score":0.0,"speedup":0.0,"speedup_score":0.0},artifacts={"feedback":f"{e}\n{traceback.format_exc()}"})
