import sys, os, time, torch
sys.path.insert(0, os.getcwd()); import fast_steer as fs, pandas as pd
from src.prompt import gen_prompt, gen_common_suffixes
from src.constants import WORD_LIST
cfg = fs.Cfg(model_name=sys.argv[1], devices="0,1,2"); m, tok = fs.load_model(cfg); M = fs.LastTokModel(m, tok)
df = pd.read_csv(cfg.dataset_path); mask = df["en"].isin(WORD_LIST); pdf, sdf = df[mask], df[~mask].reset_index(drop=True)
pr, sf = gen_prompt(pdf, "fr", "de"), gen_common_suffixes(sdf["ru"], "ru", "zh")
t = time.time(); fs.tokenize_with_prompt(tok, pr, sf); print("tokenize", time.time() - t)
for mb in [16, 64, 133]:
    fs._sync_all(); t = time.time(); c = fs.prefill(M, pr, sf, mb=mb); fs._sync_all(); print("prefill mb", mb, time.time() - t)
ids = torch.randint(100, 1000, (64, 85), device=M.dev[0])
for kw in [dict(use_cache=True, logits_to_keep=1), dict(use_cache=False, logits_to_keep=1)]:
    fs._sync_all(); t = time.time(); m(input_ids=ids, **kw); fs._sync_all(); print("hf fwd 64x85", kw, time.time() - t)
