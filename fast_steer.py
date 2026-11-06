# %%
"""
Fast steering sweep (re-implementation of run_steer2 in 2025_translation_results.py).

What the original does, per (src,dest,src2,dest2) and for every layer window [start,end):
  * clean run of "src->dest" translation prompt for word i  -> c_new[l][i] = <resid_post_l, u_en(i)>
  * steered run on a query about word pi(i) (src2 -> dest2); at resid_post of layers start..end-1,
    at the last position only:  r <- r - <r,u_old> u_old + c_new u_new
    with u_old = u_en(pi(i)), u_new = u_en(i)  (normalised W_U rows of the 1st token of " "+english word)
  * score: does the model now output word i (steer_to) or still pi(i) (steer_from), in dest2 or English.

Why the original is slow: for each of the ~500 windows it re-runs the whole suffix (~10 tokens/word)
through all layers with TransformerLens hooks.

What this file does instead
  1. Prefill once: KV of the shared prompt (stored once, never broadcast) and of each word's suffix
     minus its last token. Those are identical for every window (only the last position is steered,
     and earlier positions cannot attend to it).
  2. Only the last (query) position is recomputed per window, with a custom last-token forward that
     reuses the HF modules' weights (Llama/Mistral/Qwen2/Yi architecture).
  3. Layer-wise de-duplication: at layer l, two windows (s,e),(s',e') produce identical states iff
     both are unsteered so far (s,s' > l), or s==s' and both still steering (e,e' > l). We only compute
     unique states ("trunk sharing"); ~3x fewer FLOPs than one pass per window. The same dedup is
     applied to greedy-generated tokens (rows that generated the same tokens share states).
  4. Scoring by exact greedy decoding, compared at the BYTE level against the candidate strings
     (word variants + closing quote). This handles words that are split into several tokens and
     byte-fallback tokens (<0xEC><0x95>...; Korean, rare CJK) with no first-token heuristics.
     Rows are pruned as soon as their decoded bytes stop being a prefix of any candidate.
  The original first-token metric (argmax in prefix/leading-byte answer set) is also computed from
  the step-0 logits so results can be compared one-to-one with the old CSVs.
"""
from __future__ import annotations
import os, sys, re, time, math, json, itertools
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.prompt import gen_prompt, gen_common_suffixes, get_answer_tensor2  # noqa: E402
from src.constants import WORD_LIST, LANG_TO_NAME, LANGS_NO_SPACE  # noqa: E402

torch.set_grad_enabled(False)
PROFILE = os.environ.get("PROFILE", "0") == "1"
PROF = {}
def _sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
class _T:
    def __init__(self, name): self.name = name
    def __enter__(self):
        if PROFILE: _sync_all()
        self.t = time.time()
    def __exit__(self, *a):
        if PROFILE: _sync_all()
        PROF[self.name] = PROF.get(self.name, 0.0) + time.time() - self.t


# --------------------------------------------------------------------------------------
# tokenizer helpers
# --------------------------------------------------------------------------------------
def token_bytes_table(tokenizer, vocab_size: int) -> List[bytes]:
    """bytes emitted by each token id when it follows other text (SentencePiece '▁' -> ' ')."""
    toks = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    special = set(tokenizer.all_special_ids)
    is_sp = any(t == "<0x0A>" for t in toks if t is not None)
    if not is_sp:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        u2b = {v: k for k, v in bytes_to_unicode().items()}
    out = []
    for i in range(vocab_size):
        t = toks[i] if i < len(toks) else None
        if t is None or i in special:
            out.append(b"\x00")  # never matches anything -> row dies
            continue
        if is_sp:
            m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", t)
            out.append(bytes([int(m.group(1), 16)]) if m else t.replace("▁", " ").encode("utf-8"))
        else:
            try:
                out.append(bytes(u2b[c] for c in t))
            except KeyError:  # added tokens that are not byte-level
                out.append(t.encode("utf-8"))
    return out


def tokenize_with_prompt(tokenizer, prompt: str, suffixes: List[str]):
    """Tokenise prompt (with BOS) and prompt+suffix jointly; return prompt ids and per-suffix ids.
    Checks the prefix property (joint tokenisation starts with the prompt tokenisation)."""
    p_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    out, bad = [], 0
    for s in suffixes:
        ids = tokenizer(prompt + s, add_special_tokens=True).input_ids
        if ids[: len(p_ids)] != p_ids:
            bad += 1
            ids = p_ids + tokenizer("🌍" + s, add_special_tokens=False).input_ids[2:]
        out.append(ids[len(p_ids):])
    return p_ids, out, bad


def candidates(word: str, lang: str) -> List[bytes]:
    """Accepted answer strings, as bytes, including the closing quote of the prompt format."""
    vs = {word, word.lower(), word.capitalize()}
    outs = set()
    for v in vs:
        for lead in (" ", ""):
            outs.add((lead + v + '"').encode("utf-8"))
    return sorted(outs)


# --------------------------------------------------------------------------------------
# model wrapper: last-token forward with shared prompt KV + per-word suffix KV
# --------------------------------------------------------------------------------------
class LastTokModel:
    def __init__(self, hf_model, tokenizer):
        self.m = hf_model
        self.tok = tokenizer
        c = hf_model.config
        self.n_layers = c.num_hidden_layers
        self.Hq = c.num_attention_heads
        self.Hkv = getattr(c, "num_key_value_heads", self.Hq)
        self.dh = getattr(c, "head_dim", None) or c.hidden_size // self.Hq
        self.g = self.Hq // self.Hkv
        self.layers = hf_model.model.layers
        self.dev = [next(l.parameters()).device for l in self.layers]
        self.emb = hf_model.model.embed_tokens
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head
        self.rot = hf_model.model.rotary_emb
        self.out_dev = self.lm_head.weight.device
        self.scale = 1.0 / math.sqrt(self.dh)
        self.V = self.lm_head.weight.shape[0]
        self._rope = {}
        self.max_pos = 4096
        act = getattr(c, "hidden_act", "silu")
        assert act in ("silu", "swish"), act
        self.dq, self.dkv = self.Hq * self.dh, self.Hkv * self.dh
        # fused weights; module weights become views into them (no extra memory, HF forward still works)
        self.W = []
        for L in self.layers:
            a, mlp = L.self_attn, L.mlp
            wqkv = torch.cat([a.q_proj.weight.data, a.k_proj.weight.data, a.v_proj.weight.data], 0)
            a.q_proj.weight.data, a.k_proj.weight.data, a.v_proj.weight.data = torch.split(wqkv, [self.dq, self.dkv, self.dkv], 0)
            bqkv = None
            if a.q_proj.bias is not None:
                bqkv = torch.cat([a.q_proj.bias.data, a.k_proj.bias.data, a.v_proj.bias.data], 0)
                a.q_proj.bias.data, a.k_proj.bias.data, a.v_proj.bias.data = torch.split(bqkv, [self.dq, self.dkv, self.dkv], 0)
            wgu = torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data], 0)
            ff = mlp.gate_proj.weight.shape[0]
            mlp.gate_proj.weight.data, mlp.up_proj.weight.data = torch.split(wgu, [ff, ff], 0)
            self.W.append(dict(ln1=L.input_layernorm.weight.data, eps1=L.input_layernorm.variance_epsilon,
                               ln2=L.post_attention_layernorm.weight.data, eps2=L.post_attention_layernorm.variance_epsilon,
                               wqkv=wqkv, bqkv=bqkv, wo=a.o_proj.weight.data, bo=a.o_proj.bias.data if a.o_proj.bias is not None else None,
                               wgu=wgu, ff=ff, wd=mlp.down_proj.weight.data))
        for dv in set(self.dev):
            self.rope(torch.zeros(1, dtype=torch.long), dv, self.lm_head.weight.dtype)
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        torch.cuda.empty_cache()

    @staticmethod
    def _rms(x, w, eps):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return w * xf.to(x.dtype)

    def W_U_rows(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head.weight[ids.to(self.out_dev)].float()

    def rope(self, pos: torch.Tensor, dev, dtype):
        key = (dev, dtype)
        if key not in self._rope:
            rdev = self.rot.inv_freq.device
            allpos = torch.arange(self.max_pos, device=rdev)[None]
            cos, sin = self.rot(torch.empty(1, device=rdev, dtype=dtype), allpos)
            self._rope[key] = (cos[0].to(dev), sin[0].to(dev))
        cos, sin = self._rope[key]
        pos = pos.to(dev)
        return cos[pos][:, None, :], sin[pos][:, None, :]  # (N,1,dh)

    @staticmethod
    def _rot_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def qkv(self, l, x, pos):
        """x (N,d) residual input of layer l -> q (N,Hq,dh), k,v (N,Hkv,dh) post-rope."""
        Wl = self.W[l]
        h = self._rms(x, Wl["ln1"], Wl["eps1"])
        qkv = F.linear(h, Wl["wqkv"], Wl["bqkv"])
        qk, v = qkv.split([self.dq + self.dkv, self.dkv], -1)
        qk = qk.view(-1, self.Hq + self.Hkv, self.dh)
        cos, sin = self.rope(pos, x.device, x.dtype)
        qk = qk * cos + self._rot_half(qk) * sin
        q, k = qk.split([self.Hq, self.Hkv], 1)
        return q, k, v.view(-1, self.Hkv, self.dh)

    def attend(self, l, q, k, v, prev_k, prev_v, ctx, wpos, maxcnt):
        """Attention of U query states over: shared prompt KV, their word's suffix KV, their own
        earlier-step KVs and themselves. bf16 matmuls with fp32 accumulation/outputs."""
        U, Hkv, g, dh = q.shape[0], self.Hkv, self.g, self.dh
        f32 = torch.float32
        qh = q.view(U, Hkv, g, dh)
        # prompt: (Hkv, U*g, P)
        qp = qh.permute(1, 0, 2, 3).reshape(Hkv, U * g, dh)
        sp = torch.bmm(qp, ctx.KpT[l], out_dtype=f32)  # KpT: (Hkv, dh, P)
        P_ = sp.shape[-1]
        sp = sp.view(Hkv, U, g, P_).permute(1, 0, 2, 3)  # (U,Hkv,g,P)
        # word suffix: scatter q into (Wc*maxcnt) slots, bmm per (word, head)
        Kg, Vg = ctx.Kg[l], ctx.Vg[l]
        Wc, _, S_, _ = Kg.shape
        qpad = q.new_zeros(Wc * maxcnt, Hkv, g, dh)
        qpad[wpos] = qh
        qpad = qpad.view(Wc, maxcnt, Hkv, g, dh).permute(0, 2, 1, 3, 4).reshape(Wc * Hkv, maxcnt * g, dh)
        sg = torch.bmm(qpad, Kg.reshape(Wc * Hkv, S_, dh).transpose(1, 2), out_dtype=f32)  # (Wc*Hkv, maxcnt*g, S)
        sg = sg.view(Wc, Hkv, maxcnt, g, S_).masked_fill(~ctx.Mg[l][:, None, None, None, :], float("-inf"))
        sg = sg.permute(0, 2, 1, 3, 4).reshape(Wc * maxcnt, Hkv, g, S_)[wpos]  # (U,Hkv,g,S)
        # own + earlier steps
        ks = torch.stack(prev_k + [k], dim=2)  # (U,Hkv,T,dh)
        vs = torch.stack(prev_v + [v], dim=2)
        T_ = ks.shape[2]
        st = torch.bmm(qh.reshape(U * Hkv, g, dh), ks.reshape(U * Hkv, T_, dh).transpose(1, 2), out_dtype=f32).view(U, Hkv, g, T_)
        pr = torch.softmax(torch.cat([sp, sg, st], dim=-1) * self.scale, dim=-1).to(q.dtype)
        pp, pg, pt = pr[..., :P_], pr[..., P_:P_ + S_], pr[..., P_ + S_:]
        o = torch.bmm(pp.permute(1, 0, 2, 3).reshape(Hkv, U * g, P_), ctx.Vp[l], out_dtype=f32)
        o = o.view(Hkv, U, g, dh).permute(1, 0, 2, 3)
        pgp = pg.new_zeros(Wc * maxcnt, Hkv, g, S_)
        pgp[wpos] = pg
        pgp = pgp.view(Wc, maxcnt, Hkv, g, S_).permute(0, 2, 1, 3, 4).reshape(Wc * Hkv, maxcnt * g, S_)
        og = torch.bmm(pgp, Vg.reshape(Wc * Hkv, S_, dh), out_dtype=f32)
        og = og.view(Wc, Hkv, maxcnt, g, dh).permute(0, 2, 1, 3, 4).reshape(Wc * maxcnt, Hkv, g, dh)[wpos]
        ot = torch.bmm(pt.reshape(U * Hkv, g, T_), vs.reshape(U * Hkv, T_, dh), out_dtype=f32).view(U, Hkv, g, dh)
        return (o + og + ot).reshape(U, self.Hq, dh).to(q.dtype)

    def finish(self, l, x, attn_out):
        Wl = self.W[l]
        x = x + F.linear(attn_out.reshape(x.shape[0], -1), Wl["wo"], Wl["bo"])
        gu = F.linear(self._rms(x, Wl["ln2"], Wl["eps2"]), Wl["wgu"])
        gt, up = gu.split([Wl["ff"], Wl["ff"]], -1)
        return x + F.linear(F.silu(gt) * up, Wl["wd"])

    def logits(self, x):
        x = x.to(self.out_dev, non_blocking=True)
        return F.linear(self._rms(x, self.norm.weight.data, self.norm.variance_epsilon), self.lm_head.weight.data)


@dataclass
class Context:
    """Shared prompt KV + per-word suffix KV (suffix minus the query token)."""
    Kp: list  # per layer (Hkv,P,dh)
    Vp: list
    Kg: list  # per layer (W,Hkv,S,dh)
    Vg: list
    Mg: list  # per layer-device (W,S) bool valid
    qtok: torch.Tensor  # (W,) query token id (last suffix token)
    qpos: torch.Tensor  # (W,) its position
    P: int
    n_bad_tok: int
    hf_logits: Optional[torch.Tensor] = None  # (W,V) HF logits at query pos (for validation)


def _cache_kv(cache, l):
    if hasattr(cache, "layers"):
        return cache.layers[l].keys, cache.layers[l].values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[l], cache.value_cache[l]
    return cache[l][0], cache[l][1]


@torch.no_grad()
def prefill(M: LastTokModel, prompt: str, suffixes: List[str], mb: int = 64, keep_hf_logits=False) -> Context:
    tok = M.tok
    p_ids, s_ids, bad = tokenize_with_prompt(tok, prompt, suffixes)
    P = len(p_ids)
    W = len(suffixes)
    S = max(len(s) for s in s_ids) - 1  # suffix minus the query token
    S = max(S, 1)
    nL = M.n_layers
    Kg = [torch.zeros(W, M.Hkv, S, M.dh, dtype=M.m.dtype, device=M.dev[l]) for l in range(nL)]
    Vg = [torch.zeros_like(k) for k in Kg]
    Mg_cpu = torch.zeros(W, S, dtype=torch.bool)
    qtok = torch.tensor([s[-1] for s in s_ids])
    qpos = torch.tensor([P + len(s) - 1 for s in s_ids])
    Kp = Vp = None
    hf_logits = torch.empty(W, M.V, dtype=torch.float32) if keep_hf_logits else None
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    from transformers import DynamicCache
    # 1) the shared prompt, once
    pout = M.m(input_ids=torch.tensor([p_ids], device=M.dev[0]), use_cache=True, logits_to_keep=1)
    Kp, Vp = [], []
    for l in range(nL):
        k, v = _cache_kv(pout.past_key_values, l)
        Kp.append(k[0].contiguous().to(M.dev[l])); Vp.append(v[0].contiguous().to(M.dev[l]))
    del pout
    # 2) per-word suffixes on top of the (expanded) prompt cache
    for b0 in range(0, W, mb):
        rows = list(range(b0, min(W, b0 + mb)))
        b = len(rows)
        Ls = max(len(s_ids[i]) for i in rows)
        ids = torch.full((b, Ls), pad, dtype=torch.long)
        am = torch.zeros((b, P + Ls), dtype=torch.long)
        am[:, :P] = 1
        for j, i in enumerate(rows):
            ids[j, : len(s_ids[i])] = torch.tensor(s_ids[i])
            am[j, P: P + len(s_ids[i])] = 1
        cache = DynamicCache()
        for l in range(nL):
            cache.update(Kp[l][None].expand(b, -1, -1, -1).contiguous(), Vp[l][None].expand(b, -1, -1, -1).contiguous(), l)
        pos = (P + torch.arange(Ls))[None].expand(b, -1)
        out = M.m(input_ids=ids.to(M.dev[0]), attention_mask=am.to(M.dev[0]), position_ids=pos.to(M.dev[0]),
                  past_key_values=cache, use_cache=True, logits_to_keep=0 if keep_hf_logits else 1)
        ns = torch.tensor([len(s_ids[i]) - 1 for i in rows])
        m_ = (torch.arange(S)[None, :] < ns[:, None])  # (b,S) valid suffix positions (query token excluded)
        for l in range(nL):
            k, v = _cache_kv(out.past_key_values, l)
            n_av = min(S, k.shape[2] - P)
            mm = m_[:, :n_av].to(k.device)[:, None, :, None]
            Kg[l][rows[0]: rows[-1] + 1, :, :n_av] = (k[:, :, P: P + n_av] * mm).to(M.dev[l])
            Vg[l][rows[0]: rows[-1] + 1, :, :n_av] = (v[:, :, P: P + n_av] * mm).to(M.dev[l])
        for j, i in enumerate(rows):
            n = len(s_ids[i]) - 1
            Mg_cpu[i, :n] = True
            if keep_hf_logits:
                hf_logits[i] = out.logits[j, n].float().cpu()
        del out, cache
    Mg = [Mg_cpu.to(M.dev[l]) for l in range(nL)]
    ctx = Context(Kp, Vp, Kg, Vg, Mg, qtok, qpos, P, bad, hf_logits)
    ctx.KpT = [k.transpose(1, 2).contiguous() for k in Kp]  # (Hkv, dh, P)
    return ctx


# --------------------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------------------
@dataclass
class SteerSpec:
    u_old: torch.Tensor  # (W,d) float, unit
    u_new: torch.Tensor  # (W,d) float, unit
    c_new: torch.Tensor  # (nL,W) float
    mode: str = "swap"  # swap | add | remove | patch | patch_en | patch_noen
    alpha: float = 1.0
    r_read: Optional[torch.Tensor] = None  # (nL,W,d) clean read-context resid_post (for patch modes)


def _proj2(v, uo, un):
    """projection of v onto span{uo, un} (rows), via Gram-Schmidt."""
    e1 = uo
    e2 = un - (un * e1).sum(-1, keepdim=True) * e1
    n2 = e2.norm(dim=-1, keepdim=True)
    e2 = torch.where(n2 > 1e-6, e2 / n2.clamp(min=1e-6), torch.zeros_like(e2))
    return (v * e1).sum(-1, keepdim=True) * e1 + (v * e2).sum(-1, keepdim=True) * e2


def apply_intervention(mode, alpha, x, uo, un, cn, rr):
    """x: (n,d) fp32 resid at the query position of the WRITE run (about word pi(i));
    uo/un: unit latent directions of the old (pi(i)) / new (i) word; cn: coefficient of the new word's
    direction in the clean READ run; rr: full READ-run resid (for patch modes)."""
    if mode == "swap":      # original: replace old word's coefficient by new word's
        return x - (x * uo).sum(-1, keepdim=True) * uo + alpha * cn[:, None] * un
    if mode == "add":       # only add the new word's direction
        return x + alpha * cn[:, None] * un
    if mode == "remove":    # only remove the old word's direction
        return x - (x * uo).sum(-1, keepdim=True) * uo
    if mode == "patch":     # full activation patch from the read run (upper bound)
        return rr.clone()
    if mode == "patch_en":  # patch only the 2-d latent subspace span{uo,un} from the read run
        return x - _proj2(x, uo, un) + alpha * _proj2(rr, uo, un)
    if mode == "patch_noen":  # patch everything EXCEPT span{uo,un}
        return rr - _proj2(rr, uo, un) + _proj2(x, uo, un)
    raise ValueError(mode)


def class_table(configs: List[Tuple[int, int]], nL: int) -> torch.Tensor:
    """CLS[l, c]: id of the equivalence class of config c's resid_post at layer l.
    0 = never steered so far; 1+s = still steering since s; 1+nL+c = finished steering (unique)."""
    C = len(configs)
    T = torch.empty(nL, C, dtype=torch.long)
    for ci, (s, e) in enumerate(configs):
        for l in range(nL):
            if l < s:
                T[l, ci] = 0
            elif l < e:
                T[l, ci] = 1 + s
            else:
                T[l, ci] = 1 + nL + ci
    return T


def steered_mask(configs, nL) -> torch.Tensor:
    """ST[l,c] = config c steers at resid_post of layer l."""
    T = torch.zeros(nL, len(configs), dtype=torch.bool)
    for ci, (s, e) in enumerate(configs):
        T[s:min(e, nL), ci] = True
    return T


class Scorer:
    """Byte-level greedy-decode scoring, deduplicated over (word, generated-prefix) keys."""

    def __init__(self, tok_bytes: List[bytes], cand: Dict[str, List[List[bytes]]]):
        self.tb = tok_bytes
        self.cand = cand  # class name -> per word list of candidate bytes
        self.classes = list(cand.keys())
        W = len(next(iter(cand.values())))
        self.all_c = [sorted(set(itertools.chain(*[cand[k][w] for k in self.classes]))) for w in range(W)]
        self.max_len = max(len(c) for cs in self.all_c for c in cs)

    def status(self, w: int, b: bytes):
        """returns (bitmask of classes completed, alive?)"""
        done = 0
        for ki, k in enumerate(self.classes):
            for c in self.cand[k][w]:
                if b.startswith(c):
                    done |= 1 << ki
                    break
        if done:
            return done, False
        if len(b) >= self.max_len + 8:
            return 0, False
        for c in self.all_c[w]:
            if c.startswith(b):
                return 0, True
        return 0, False


def run_engine(*a, **kw):
    """Synchronous wrapper around engine_gen."""
    g = engine_gen(*a, **kw)
    while True:
        try:
            with torch.no_grad():
                next(g)
        except StopIteration as e:
            return e.value


_STREAMS = {}
def get_streams(M, slot):
    """Fixed per-slot CUDA streams (one per device). Reused across chunks: the caching allocator
    only recycles memory within a stream, so fresh streams per chunk would leak cached blocks."""
    if slot not in _STREAMS:
        _STREAMS[slot] = {dv: torch.cuda.Stream(dv) for dv in set(M.dev) | {M.out_dev}}
    return _STREAMS[slot]


def run_pipelined(gens: list, max_inflight: int):
    """Round-robin several engine generators so that their GPU work overlaps across the layer-sharded
    devices (GPU0 runs chunk B's early layers while GPU2 runs chunk A's late layers)."""
    # gens: list of factories slot -> generator
    results = [None] * len(gens)
    pending = list(enumerate(gens))
    active = []  # (i, gen, slot)
    free = list(range(max_inflight))
    while pending or active:
        while pending and free:
            i, fac = pending.pop(0)
            slot = free.pop(0)
            active.append((i, fac(slot), slot))
        for item in list(active):
            i, g, slot = item
            try:
                with torch.no_grad():
                    next(g)
            except StopIteration as e:
                results[i] = e.value
                active.remove(item)
                free.append(slot)
    return results


def engine_gen(M: LastTokModel, ctx: Context, configs: List[Tuple[int, int]], words: torch.Tensor,
               steer: Optional[SteerSpec], scorer: Scorer, orig_ans: Optional[Dict[str, torch.Tensor]] = None,
               max_new: int = 24, attn_chunk: int = 1024, record_resid: bool = False, timers=None, slot: int = 0):
    """Greedy-decode every (config, word) row with steering at the query position.
    words: indices into ctx rows. Returns dict of per-row results (C, W)."""
    nL = M.n_layers
    C, Wn = len(configs), len(words)
    R = C * Wn
    CLS = class_table(configs, nL)  # (nL,C)
    ST = steered_mask(configs, nL)
    NCLS = 1 + nL + C
    row_cfg = torch.arange(C).repeat_interleave(Wn)  # (R,)
    row_w = words.repeat(C)  # (R,) ctx word index
    row_wl = torch.arange(Wn).repeat(C)  # local word idx (0..Wn) for scorer/answers
    # per-row outputs
    done_mask = torch.zeros(R, dtype=torch.long)
    n_steps = torch.zeros(R, dtype=torch.long)
    first_tok = torch.zeros(R, dtype=torch.long)
    gen_toks = torch.full((R, max_new + 1), -1, dtype=torch.long)
    orig_stats = {k: torch.zeros(R, 2) for k in (orig_ans or {})}  # (prob, correct)
    resid_rec = [None] * nL if record_resid else None

    # state of each row
    alive = torch.ones(R, dtype=torch.bool)
    key = row_wl.clone()  # key_t: id of (word, generated tokens)
    key_bytes = {int(k): b"" for k in key.unique()}  # key id -> bytes so far (key -> word via key_word)
    key_word = {int(k): int(k) for k in key.unique()}
    cur_tok = ctx.qtok[row_w]  # token fed at step t
    # storage of KV per step/layer: list over t of list over l of (K (U,Hkv,dh), V, S_rows (R,) long, -1 for dead)
    KV = []
    Sprev = []  # per step: list over layers of S_full (CPU)
    nsteps_done = 0
    streams = get_streams(M, slot)
    for dv, st_ in streams.items():
        st_.wait_stream(torch.cuda.current_stream(dv))  # prefill/ctx tensors are ready
    def on(dv):
        return torch.cuda.stream(streams[dv])
    steer_dev = {}
    if steer is not None:
        for dv in set(M.dev):
            steer_dev[dv] = {"u_old": steer.u_old.to(dv), "u_new": steer.u_new.to(dv), "c_new": steer.c_new.float().to(dv)}
            if steer.r_read is not None:
                steer_dev[dv]["r_read"] = steer.r_read.float().to(dv)
    for dv, st_ in streams.items():  # steer tensors were made on the default streams
        st_.wait_stream(torch.cuda.current_stream(dv))
    for t in range(max_new + 1):
        A = alive.nonzero().squeeze(1)
        if A.numel() == 0:
            break
        nsteps_done += 1
        kv_t = []
        _tp = _T('index'); _tp.__enter__()
        # state ids at each layer (computed upfront, CPU)
        keys_l = key[A][None, :] * NCLS + CLS[:, row_cfg[A]]  # (nL, |A|)
        S_l, reps_l = [], []
        for l in range(nL):
            u, inv = torch.unique(keys_l[l], return_inverse=True)
            rep = torch.empty(u.numel(), dtype=torch.long)
            rep.scatter_(0, inv, torch.arange(A.numel()))
            S_full = torch.full((R,), -1, dtype=torch.long)
            S_full[A] = inv
            S_l.append(S_full)
            reps_l.append(A[rep])  # global row ids of representatives
        # ---- precompute all index tensors for this step on CPU, ship once per device ----
        packs = {}
        plan = []
        plan_meta = []
        Wctx = ctx.Kg[0].shape[0]
        for l in range(nL):
            rep = reps_l[l]
            rw = row_w[rep]
            ent = {"rep_w": rw, "pos": ctx.qpos[rw] + t}
            # slot of each state in a (W_ctx, maxcnt) per-word padded layout (for the suffix-KV bmm)
            order = torch.argsort(rw, stable=True)
            cnt = torch.bincount(rw, minlength=Wctx)
            maxcnt = int(cnt.max())
            starts_ = torch.cumsum(cnt, 0) - cnt
            rank = torch.empty_like(rw); rank[order] = torch.arange(rw.numel()) - starts_[rw[order]]
            ent["wpos"] = rw * maxcnt + rank
            plan_meta.append(maxcnt)
            ent["parent"] = (S_l[l - 1][rep] if l > 0 else cur_tok[rep])
            for tp in range(t):
                ent[f"prev{tp}"] = Sprev[tp][l][rep]
            if t == 0 and steer is not None:
                stm = ST[l, row_cfg[rep]]
                si = stm.nonzero().squeeze(1)
                ent["st_idx"] = si
                ent["st_w"] = row_w[rep][si]
            if record_resid and t == 0:
                ent["rec"] = S_l[l][A]
            plan.append(ent)
        dev_ents = {}
        for l in range(nL):
            dev_ents.setdefault(M.dev[l], []).append(l)
        for dv, ls in dev_ents.items():
            flat, offs = [], []
            o = 0
            for l in ls:
                for kname, v in plan[l].items():
                    offs.append((l, kname, o, v.numel())); flat.append(v); o += v.numel()
            with on(dv):
                buf = torch.cat(flat).pin_memory().to(dv, non_blocking=True) if flat else None
            for (l, kname, o_, n_) in offs:
                packs[(l, kname)] = buf[o_: o_ + n_]
        _tp.__exit__(); _tl = _T('layers' if t == 0 else 'layers_gen'); _tl.__enter__()
        x = None
        for l in range(nL):
            dev = M.dev[l]
            _ctx = on(dev); _ctx.__enter__()
            U = reps_l[l].numel()
            PROF['rows_t0' if t == 0 else 'rows_gen'] = PROF.get('rows_t0' if t == 0 else 'rows_gen', 0) + U
            if l == 0:
                x = M.emb(packs[(0, "parent")])
            else:
                if x.device == dev:
                    x = x[packs[(l, "parent")]]
                else:
                    with on(x.device), on(dev):  # copy must see both devices' chunk streams
                        x = x.to(dev, non_blocking=True)
                    x = x[packs[(l, "parent")]]
            q, k, v = M.qkv(l, x, packs[(l, "pos")])
            prev_k, prev_v = [], []
            for tp in range(t):
                Kt, Vt = KV[tp][l]
                idx = packs[(l, f"prev{tp}")]
                prev_k.append(Kt[idx]); prev_v.append(Vt[idx])
            attn = M.attend(l, q, k, v, prev_k, prev_v, ctx, packs[(l, "wpos")], plan_meta[l])
            del prev_k, prev_v
            x = M.finish(l, x, attn)
            kv_t.append((k, v))
            if t == 0 and steer is not None and plan[l]["st_idx"].numel():
                sidx, w_rep = packs[(l, "st_idx")], packs[(l, "st_w")]
                sv = steer_dev[dev]
                xs = x[sidx].float()
                uo, un, cn = sv["u_old"][w_rep], sv["u_new"][w_rep], sv["c_new"][l][w_rep]
                xs = apply_intervention(steer.mode, steer.alpha, xs, uo, un, cn,
                                        sv["r_read"][l][w_rep] if "r_read" in sv else None)
                x[sidx] = xs.to(x.dtype)
            if record_resid and t == 0:
                resid_rec[l] = x[packs[(l, "rec")]].float().cpu()
            _ctx.__exit__(None, None, None)
        KV.append(kv_t)
        Sprev.append(S_l)
        # logits of unique final states
        with on(x.device), on(M.out_dev):
            lg = M.logits(x)  # (U_last, V)
            arg_dev = lg.argmax(-1)
        _tl.__exit__()
        yield  # GPU work for this step is queued; let other chunks launch theirs
        _tg = _T('logits'); _tg.__enter__()
        S_last = S_l[nL - 1]
        _co = on(M.out_dev); _co.__enter__()
        arg = arg_dev.cpu()
        if t == 0:
            if orig_ans:
                rep = reps_l[nL - 1]
                lse = torch.cat([torch.logsumexp(lg[i:i + 1024].float(), -1) for i in range(0, lg.shape[0], 1024)])
                arg_d = arg.to(lg.device, non_blocking=True)
                outs_ = []
                for kname, ans in orig_ans.items():
                    a = ans[row_wl[rep]].to(lg.device, non_blocking=True)  # (U, A) padded with -1
                    valid = a >= 0
                    pr = (lg.gather(1, a.clamp(min=0)).float() - lse[:, None]).exp() * valid
                    corr = ((a == arg_d[:, None]) & valid).any(-1)
                    outs_.append(torch.stack([pr.sum(-1), corr.float()], dim=1))
                outs_ = torch.stack(outs_).cpu()
                for ki, kname in enumerate(orig_ans):
                    orig_stats[kname][A] = outs_[ki][S_last[A]]
            first_tok[A] = arg[S_last[A]]
        del lg
        _co.__exit__(None, None, None)
        _tg.__exit__(); _ts = _T('score'); _ts.__enter__()
        g = arg[S_last[A]]  # next tokens for alive rows
        # update keys: (key, token) -> new key
        pair = key[A] * M.V + g
        upair, inv = torch.unique(pair, return_inverse=True)
        new_keys = {}
        new_alive = torch.zeros(A.numel(), dtype=torch.bool)
        status_cache = {}
        upair_l = upair.tolist()
        for j, pk in enumerate(upair_l):
            okey, gt = divmod(pk, M.V)
            w = key_word[okey]
            b = key_bytes[okey] + scorer.tb[gt]
            status_cache[j] = (w, b) + scorer.status(w, b)
        dm = torch.tensor([status_cache[j][2] for j in range(len(upair_l))], dtype=torch.long)
        al = torch.tensor([status_cache[j][3] for j in range(len(upair_l))], dtype=torch.bool)
        # new key ids: offset so they don't collide with old ones
        base = max(key_word.keys()) + 1
        for j in range(len(upair_l)):
            key_word[base + j] = status_cache[j][0]
            key_bytes[base + j] = status_cache[j][1]
        done_mask[A] = dm[inv]
        gen_toks[A, t] = g
        n_steps[A] = t + 1
        key[A] = base + inv
        cur_tok[A] = g
        alive[A] = al[inv]
        # compact stored KV: drop states no alive row refers to any more
        A2 = alive.nonzero().squeeze(1)
        if A2.numel() and A2.numel() < A.numel():
            for tp in range(len(KV)):
                for l in range(nL):
                    Sx = Sprev[tp][l]
                    keep, newid = torch.unique(Sx[A2], return_inverse=True)
                    S_new = torch.full_like(Sx, -1); S_new[A2] = newid
                    Sprev[tp][l] = S_new
                    dv = M.dev[l]
                    with on(dv):
                        kk = keep.pin_memory().to(dv, non_blocking=True)
                        Kt, Vt = KV[tp][l]
                        KV[tp][l] = (Kt[kk], Vt[kk])
        _ts.__exit__()
    out = dict(done=done_mask.view(C, Wn), first_tok=first_tok.view(C, Wn), n_steps=n_steps.view(C, Wn),
               steps=nsteps_done, final_bytes={})
    for kname, v in orig_stats.items():
        out[f"orig_prob_{kname}"] = v[:, 0].view(C, Wn)
        out[f"orig_acc_{kname}"] = v[:, 1].view(C, Wn)
    out["gen_toks"] = gen_toks.view(C, Wn, -1)
    out["key_final"] = key.view(C, Wn)
    out["key_bytes"] = key_bytes
    if record_resid:
        out["resid"] = torch.stack(resid_rec, 0)  # (nL, R, d)
    return out


# --------------------------------------------------------------------------------------
# experiment driver
# --------------------------------------------------------------------------------------
@dataclass
class Cfg:
    model_name: str = "NousResearch/Llama-2-7b-hf"
    dataset_path: str = "data_wendler/word_list.csv"
    langs: str = "fr,de,ru,zh"  # non-English languages for (src,dest,src2,dest2) permutations
    latent_lang: str = "en"  # language whose unembedding directions are swapped
    direction: str = "first"  # first | mean  (token(s) of the latent word used as direction)
    write_prompt: str = "orig"  # orig: few-shot prompt of (src,dest) as in run_steer2; own: (src2,dest2)
    grid: str = "orig"  # orig: end<=nL-1 as in run_steer2; full: end<=nL
    perms: str = ""  # e.g. "fr,de,ru,zh;de,fr,zh,ru" or None = all
    max_perms: int = 0
    cfg_chunk: int = 128  # configs per engine call (memory)
    inflight: int = 3  # engine chunks interleaved (pipelining over layer-sharded GPUs)
    word_chunk: int = 1000
    devices: str = "0"
    max_new: int = 24
    out: str = "out_fast"
    seed: int = 42
    shuffle_path: str = ""  # load derangement (to match an original run)
    tag: str = ""
    dtype: str = "bfloat16"
    max_mem_gb: float = 0.0
    layer_split: str = "1.25,1,0.8"  # relative #layers per device
    mode: str = "swap"
    alpha: float = 1.0
    start_step: int = 1  # >1: coarse grid over start layers (end layers stay dense)
    shots: int = 4  # few-shot examples in the prompt (4 = the original WORD_LIST)
    exclude_latent: bool = False  # drop the latent language from the permutation languages
    known: bool = True  # also measure whether the model can translate word i in the write context


def load_model(cfg: Cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    devs = [int(d) for d in cfg.devices.split(",")]
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if len(devs) == 1:
        m = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(f"cuda:{devs[0]}")
    else:
        from transformers import AutoConfig
        nL = AutoConfig.from_pretrained(cfg.model_name).num_hidden_layers
        # later layers hold more distinct steering states (more KV), and the last device holds lm_head:
        # put more layers on the first device(s). split fractions configurable via cfg.layer_split.
        fr = [float(x) for x in cfg.layer_split.split(",")] if cfg.layer_split else [1.0] * len(devs)
        cuts = np.round(np.cumsum(fr) / sum(fr) * nL).astype(int)
        dmap = {"model.embed_tokens": devs[0], "model.norm": devs[-1], "model.rotary_emb": devs[-1], "lm_head": devs[-1]}
        lo = 0
        for d, hi in zip(devs, cuts):
            for l in range(lo, hi):
                dmap[f"model.layers.{l}"] = d
            lo = hi
        m = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype, device_map=dmap)
    m.eval()
    return m, tok


def latent_dirs(M: LastTokModel, words: List[str], lang: str, mode: str) -> torch.Tensor:
    tok = M.tok
    space = "" if lang in LANGS_NO_SPACE else " "
    rows = []
    for w in words:
        ids = tok("🌍" + space + w, add_special_tokens=False).input_ids
        ids0 = tok("🌍", add_special_tokens=False).input_ids
        ids = ids[len(ids0):]
        if mode == "first":
            r = M.W_U_rows(torch.tensor(ids[:1]))[0]
        elif mode == "mean":
            r = M.W_U_rows(torch.tensor(ids)).mean(0)
        else:
            raise ValueError(mode)
        rows.append((r / r.norm()).cpu())
    return torch.stack(rows)


def orig_answer_sets(tok, words, cfgA):
    vocab = tok.get_vocab()
    pad = -1
    a = get_answer_tensor2(pd.Series(words), tok, vocab, cfgA, padding_value=tok.pad_token_id if tok.pad_token_id is not None else 0)
    pid = tok.pad_token_id if tok.pad_token_id is not None else 0
    a = a.clone()
    a[a == pid] = -1
    return a


def main(cfg: Cfg):
    t_all = time.time()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    m, tok = load_model(cfg)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # as TransformerLens does
    tok.padding_side = "right"  # as TransformerLens does (safe_tokenize relies on it)
    M = LastTokModel(m, tok)
    nL = M.n_layers
    print(f"loaded {cfg.model_name} in {time.time()-t_all:.1f}s  layers={nL} devs={sorted(set(map(str,M.dev)))}")
    tb = token_bytes_table(tok, M.V)

    df = pd.read_csv(cfg.dataset_path)
    mask = df["en"].isin(WORD_LIST)
    if cfg.shots > 4:  # add more few-shot examples (fixed random words, removed from the eval set)
        extra = df[~mask].sample(cfg.shots - 4, random_state=0).index
        mask = mask | df.index.isin(extra)
    prompt_df, suffix_df = df[mask], df[~mask].reset_index(drop=True)
    W = len(suffix_df)
    sh_ = torch.load(cfg.shuffle_path) if cfg.shuffle_path else None
    if sh_ is not None and len(sh_) == W:
        shuffle_idx = sh_
    else:
        from utils.misc import dearrange
        shuffle_idx = dearrange(torch.arange(W))
    sdf = suffix_df.iloc[shuffle_idx.numpy()].reset_index(drop=True)

    L = cfg.langs.split(",")
    if cfg.exclude_latent:
        L = [x for x in L if x != cfg.latent_lang]
    perms = list(itertools.permutations(L)) if not cfg.perms else [tuple(p.split(",")) for p in cfg.perms.split(";")]
    if cfg.max_perms:
        perms = perms[: cfg.max_perms]
    ends = nL if cfg.grid == "orig" else nL + 1
    configs = [(s, e) for s in range(0, nL, cfg.start_step) for e in range(s + 1, ends)]

    u_lat = latent_dirs(M, list(suffix_df[cfg.latent_lang]), cfg.latent_lang, cfg.direction)  # (W,d) word i
    @dataclass
    class _A:
        token_add_capitalization: bool = True
        token_add_prefixes: bool = True
        token_add_spaces: bool = True
        token_utf8_byte: bool = True
    ansA = _A()
    orig_cache = {}
    def oa(lang, df_):
        k = (lang, id(df_))
        if k not in orig_cache:
            orig_cache[k] = orig_answer_sets(tok, list(df_[lang]), ansA)
        return orig_cache[k]

    os.makedirs(cfg.out, exist_ok=True)
    mb = os.path.basename(cfg.model_name.rstrip("/"))
    if "snapshots" in cfg.model_name:
        mb = cfg.model_name.split("models--")[1].split("/")[0].split("--")[-1]
    rows_out, timing = [], []
    rctx_cache = {}
    for pi, (src, dest, src2, dest2) in enumerate(perms):
        t0 = time.time()
        # read context: clean (src->dest), word i
        if (src, dest) not in rctx_cache:
            rctx_cache.clear()
            rctx_cache[(src, dest)] = prefill(M, gen_prompt(prompt_df, src, dest), gen_common_suffixes(suffix_df[src], src, dest))
        rctx = rctx_cache[(src, dest)]
        wprompt = gen_prompt(prompt_df, src, dest) if cfg.write_prompt == "orig" else gen_prompt(prompt_df, src2, dest2)
        wctx = prefill(M, wprompt, gen_common_suffixes(sdf[src2], src2, dest2))
        torch.cuda.synchronize(); t_pref = time.time() - t0
        # clean read run -> c_new
        cand_r = {"clean": [candidates(x, dest) for x in suffix_df[dest]], "clean_en": [candidates(x, "en") for x in suffix_df["en"]]}
        sc_r = Scorer(tb, cand_r)
        rres = run_engine(M, rctx, [(nL, nL + 1)], torch.arange(W), None, sc_r,
                          orig_ans={"clean": oa(dest, suffix_df)}, max_new=cfg.max_new, record_resid=True)
        resid = rres["resid"]  # (nL, W, d)
        c_new = torch.einsum("lwd,wd->lw", resid, u_lat)
        steer = SteerSpec(u_old=u_lat[shuffle_idx], u_new=u_lat, c_new=c_new, mode=cfg.mode, alpha=cfg.alpha,
                          r_read=resid if cfg.mode.startswith("patch") else None)
        t_clean = time.time() - t0 - t_pref
        # write rows: row j queries sdf word j (= suffix word shuffle_idx[j]); inject word j
        cand = {
            "steer_to": [candidates(x, dest2) for x in suffix_df[dest2]],
            "steer_from": [candidates(x, dest2) for x in sdf[dest2]],
            "steer_to_en": [candidates(x, "en") for x in suffix_df["en"]],
            "steer_from_en": [candidates(x, "en") for x in sdf["en"]],
        }
        if cfg.latent_lang not in ("en", dest2):
            cand["steer_to_lat"] = [candidates(x, cfg.latent_lang) for x in suffix_df[cfg.latent_lang]]
        cand["steer_to_rdest"] = [candidates(x, dest) for x in suffix_df[dest]]  # word i in the READ language
        # does the model know word i -> dest2 in this write context? (unsteered, unshuffled)
        known = torch.ones(W, dtype=torch.bool)
        if cfg.known:
            kctx = prefill(M, wprompt, gen_common_suffixes(suffix_df[src2], src2, dest2))
            kres = run_engine(M, kctx, [(nL, nL + 1)], torch.arange(W), None,
                              Scorer(tb, {"k": [candidates(x, dest2) for x in suffix_df[dest2]]}), max_new=cfg.max_new)
            known = (kres["done"][0] & 1).bool()
            del kctx
        sc = Scorer(tb, cand)
        orig_ans = {"steer_to": oa(dest2, suffix_df), "steer_from": oa(dest2, sdf),
                    "steer_to_en": oa("en", suffix_df), "steer_from_en": oa("en", sdf)}
        # also unsteered write context for reference
        res_all = {k: [] for k in ["done"] + [f"orig_acc_{k}" for k in orig_ans] + [f"orig_prob_{k}" for k in orig_ans]}
        n_steps_tot = 0
        # chunks of whole start-groups (keeps trunk sharing inside a chunk)
        chunks, cur = [], []
        for s_ in range(nL):
            grp = [c for c in configs if c[0] == s_]
            if cur and len(cur) + len(grp) > cfg.cfg_chunk:
                chunks.append(cur); cur = []
            cur = cur + grp
        if cur: chunks.append(cur)
        jobs, gens = [], []
        c0 = 0
        for cc in chunks:
            for w0 in range(0, W, cfg.word_chunk):
                ww = torch.arange(w0, min(W, w0 + cfg.word_chunk))
                sc_w = Scorer(tb, {k: [v[i] for i in ww.tolist()] for k, v in cand.items()})
                oa_w = {k: v[ww] for k, v in orig_ans.items()}
                jobs.append((c0, w0))
                gens.append(lambda slot, cc=cc, ww=ww, sc_w=sc_w, oa_w=oa_w: engine_gen(
                    M, wctx, cc, ww, steer, sc_w, orig_ans=oa_w, max_new=cfg.max_new, slot=slot))
            c0 += len(cc)
        for (c0, w0), r in zip(jobs, run_pipelined(gens, cfg.inflight)):
            n_steps_tot += r["steps"]
            for k in res_all:
                res_all[k].append((c0, w0, r[k]))
        torch.cuda.synchronize()
        t_sweep = time.time() - t0 - t_pref - t_clean
        # assemble (C,W)
        def cat(k):
            C = len(configs)
            outk = torch.zeros(C, W, dtype=res_all[k][0][2].dtype)
            for c0, w0, v in res_all[k]:
                outk[c0:c0 + v.shape[0], w0:w0 + v.shape[1]] = v
            return outk
        done = cat("done")
        clean_acc = (rres["done"][0] & 1).bool().float().mean().item()
        clean_orig = rres["orig_acc_clean"][0].mean().item()
        for ci, (s, e) in enumerate(configs):
            rec = dict(src_lang=src, dest_lang=dest, src2=src2, dest2=dest2, l_start=s, l_end=e,
                       clean_exact=clean_acc, clean_orig=clean_orig)
            for ki, k in enumerate(sc.classes):
                rec[f"{k}_exact"] = ((done[ci] >> ki) & 1).float().mean().item()
            rec["known_frac"] = known.float().mean().item()
            rec["steer_to_known"] = (((done[ci] & 1).bool() & known).float().sum() / known.float().sum().clamp(min=1)).item()
            rows_out.append(rec)
        # orig metrics vectorised
        for k in orig_ans:
            acc_k = cat(f"orig_acc_{k}").mean(1); prob_k = cat(f"orig_prob_{k}").mean(1)
            for ci in range(len(configs)):
                rows_out[-len(configs) + ci][f"{k}_orig_acc"] = acc_k[ci].item()
                rows_out[-len(configs) + ci][f"{k}_orig_prob"] = prob_k[ci].item()
        dt = time.time() - t0
        timing.append(dict(perm=f"{src},{dest},{src2},{dest2}", prefill=t_pref, clean=t_clean, sweep=t_sweep,
                           total=dt, decode_steps=n_steps_tot, bad_tok=rctx.n_bad_tok + wctx.n_bad_tok))
        best = max(rows_out[-len(configs):], key=lambda r: r["steer_to_exact"])
        print(f"[{pi+1}/{len(perms)}] {src}->{dest} | {src2}->{dest2}: {dt:.1f}s (prefill {t_pref:.1f}, clean {t_clean:.1f}, sweep {t_sweep:.1f}) "
              f"clean exact {clean_acc:.2f} orig {clean_orig:.2f}; best steer_to_exact {best['steer_to_exact']:.2f} @ [{best['l_start']},{best['l_end']})", flush=True)
    res = pd.DataFrame(rows_out)
    tag = f"_{cfg.tag}" if cfg.tag else ""
    mtag = "" if (cfg.mode == "swap" and cfg.alpha == 1.0) else f"_{cfg.mode}{cfg.alpha:g}"
    stag = "" if cfg.shots == 4 else f"_shots{cfg.shots}"
    path = os.path.join(cfg.out, mb, f"steer_{cfg.latent_lang}_{cfg.direction}_{cfg.write_prompt}{mtag}{stag}{tag}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    res.to_csv(path, index=False)
    pd.DataFrame(timing).to_csv(path.replace(".csv", "_timing.csv"), index=False)
    print(f"saved {path}; total {time.time()-t_all:.1f}s")
    if PROF: print("PROFILE", {k: round(v, 2) for k, v in PROF.items()})
    print("peak GiB per device:", [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())],
          "weights:", [round(sum(p.numel() * p.element_size() for p in m.parameters() if p.device.index == i) / 2**30, 2) for i in range(torch.cuda.device_count())])
    return res, timing


if __name__ == "__main__":
    from utils.config_argparse import try_parse_args
    cfg = try_parse_args(Cfg())
    main(cfg)
