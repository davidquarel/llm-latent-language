import sys, os, time, torch
sys.path.insert(0, os.getcwd()); import fast_steer as fs
cfg = fs.Cfg(model_name=sys.argv[1], devices="0,1,2"); m, tok = fs.load_model(cfg); M = fs.LastTokModel(m, tok)
x = torch.randn(4000, 4096, device=M.dev[-1], dtype=torch.bfloat16)
def tm(f, n=5):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n
print("lm_head via module", tm(lambda: M.logits(x)))
print("lm_head raw matmul", tm(lambda: M.norm(x) @ M.lm_head.weight.T))
print("hooks on lm_head:", getattr(M.lm_head, "_hf_hook", None))
lg = M.logits(x)
print("argmax", tm(lambda: lg.argmax(-1).cpu()))
print("log_softmax", tm(lambda: torch.log_softmax(lg.float(), -1)))
L = M.layers[5]
print("layer mlp 4000 rows", tm(lambda: L.mlp(x.to(M.dev[5]))))
print("layer hook:", getattr(L.mlp, "_hf_hook", None))
