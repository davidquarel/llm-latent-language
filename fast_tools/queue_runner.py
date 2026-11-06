# Runs fast_steer.py jobs from a list as GPUs become free (never touches foreign processes).
# 3 free GPUs -> pipelined 3-GPU config; only GPU0 free -> single-GPU low-memory config.
import subprocess, time, json, os, sys, shlex
M = "/workspace/hfcache/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
P = os.path.expanduser("~/david_quarel/ARENA_3.0/.venv/bin/python")
BASE = f"--model_name {M} --shuffle_path bench/orig_l2_7b.csv.shuffle.pt --out out_v2"
CFG3 = "--devices 0,1,2 --inflight 3 --cfg_chunk 32 --layer_split 1.4,1,0.75"
CFG2 = "--devices 1,2 --inflight 2 --cfg_chunk 32 --layer_split 1.15,1"
CFG1 = "--devices 0 --inflight 1 --cfg_chunk 1 --word_chunk 24"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
state_f = sys.argv[1] + ".state.json"
state = json.load(open(state_f)) if os.path.exists(state_f) else {}

def free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]

for i, j in enumerate(jobs):
    if state.get(j) == "done":
        continue
    while True:
        f = free_mib()
        if all(x > 15000 for x in f):
            cfg = CFG3; break
        if f[1] > 15000 and f[2] > 15000:
            cfg = CFG2; break
        if f[0] > 15500:
            cfg = CFG1; break
        time.sleep(60)
    cmd = f"{P} fast_steer.py {BASE} {cfg} {j}"
    print(time.strftime("%H:%M:%S"), "RUN", i, cfg.split()[1], j, flush=True)
    t = time.time()
    r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    log = r.stdout + r.stderr
    open(f"bench/q_{i}.log", "w").write(log)
    ok = r.returncode == 0 and "saved" in log
    state[j] = "done" if ok else "failed"
    json.dump(state, open(state_f, "w"), indent=1)
    print(time.strftime("%H:%M:%S"), "DONE" if ok else "FAILED", i, f"{time.time()-t:.0f}s", [l for l in log.splitlines() if l.startswith("saved") or "Error" in l][-2:], flush=True)
