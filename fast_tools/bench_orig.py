# Benchmark harness for the ORIGINAL run_steer2 (2025_translation_results.py).
# Execs the original source with minimal patches: #layer-pairs, 1 perm, timing, dump derangement.
import sys, time, os
NPAIRS = int(os.environ.get("NPAIRS", "20"))
OUT = os.environ.get("OUT", "orig_bench.csv")
full = open("2025_translation_results.py").read()
head, tail = full.split("def run_steer2(cfg):")
tail = "def run_steer2(cfg):" + tail
def rep(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b)
tail = rep(tail, "pairs = [(i, j) for i in range(n_layers) for j in range(i+1, n_layers)]",
           f"pairs = [(i, j) for i in range(n_layers) for j in range(i+1, n_layers)][:{NPAIRS}]\n        torch.cuda.synchronize(); _t0 = time.time()")
tail = rep(tail, "        count += 1\n", "        torch.cuda.synchronize(); print(f'BENCH perm_time={time.time()-_t0:.2f}s for {len(pairs)} pairs', flush=True); count += 1\n        break\n")
tail = rep(tail, "shuffle_idx = dearrange(torch.arange(len(suffix_df)))",
           f"shuffle_idx = dearrange(torch.arange(len(suffix_df))); torch.save(shuffle_idx, '{OUT}.shuffle.pt')")
tail = rep(tail, 'save_df(all_sweep, "steer_ALL", cfg)', f'all_sweep.to_csv("{OUT}", index=False)')
# TransformerLens only accepts official names: load the ungated mirror's weights under the official name.
import transformer_lens, transformers, torch as _torch
_orig_fp = transformer_lens.HookedTransformer.from_pretrained_no_processing
LOCAL = os.environ["LOCAL_MODEL"]; OFFICIAL = os.environ.get("OFFICIAL", "meta-llama/Llama-2-7b-hf")
def _fp(model_name, *a, **kw):
    hf = transformers.AutoModelForCausalLM.from_pretrained(LOCAL, dtype=kw.get("dtype", _torch.float32))
    tk = transformers.AutoTokenizer.from_pretrained(LOCAL)
    return _orig_fp(OFFICIAL, *a, hf_model=hf, tokenizer=tk, **kw)
transformer_lens.HookedTransformer.from_pretrained_no_processing = _fp
t = time.time()
exec(compile(head + tail, "2025_translation_results.py", "exec"), {"__name__": "__main__", "time": time, "__file__": os.path.abspath("2025_translation_results.py")})
print(f"BENCH total wallclock {time.time()-t:.1f}s")
