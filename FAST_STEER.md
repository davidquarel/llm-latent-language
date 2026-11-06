# fast_steer.py: fast steering sweep with exact scoring

Re-implementation of `run_steer2` (2025_translation_results.py).

## Speed and correctness

- **Speed:** about 20 s per language permutation for all 496 (start,end) windows on Llama-2-7b with 3xA4000. The original takes about 655 s per permutation (TransformerLens, 2 GPUs).
- **Scoring:** whole-answer greedy decoding compared at the byte level. Multi-token words and byte-fallback tokens (Korean, rare CJK) are scored correctly, so there is no need to restrict to single-token words.
- **Old metric:** the original first-token metric is still computed (`*_orig_acc`, `*_orig_prob`), so results compare one-to-one with old CSVs.

## Usage

    python fast_steer.py --model_name NousResearch/Llama-2-7b-hf --devices 0,1,2 --inflight 3 --cfg_chunk 32 \
        --layer_split 1.4,1,0.75 --latent_lang en --direction first --write_prompt orig --out out_fast

### Options

- `--latent_lang`: the language whose unembedding directions are swapped. `es` is the control.
- `--direction first|mean`: use the word's first token, or the mean over its tokens.
- `--write_prompt orig|own`:
  - `orig` is the mixed context `run_steer2` used: the (src,dest) few-shot prompt with a (src2,dest2) query.
  - `own` uses the (src2,dest2) prompt.
- `--grid orig|full`: `full` also includes windows ending after the last layer.
- `--perms "fr,de,ru,ko;..."`: restrict to specific permutations.

### Supported architectures

Llama, Mistral, Qwen2 and Yi blocks (RMSNorm, RoPE, GQA, SwiGLU). Gemma-2 is not supported yet.

## How it works

1. **Prefill once.** The prompt KV is computed once, and each word's suffix KV once. Only the steered last position is recomputed per window.
2. **Dedup.** Windows sharing a state at a layer are computed once:
   - all windows with start > l,
   - windows with the same start and end > l,
   - generated tokens that are identical.
3. **Attention.** bf16 batched matmuls with fp32 outputs. The shared prompt KV is never broadcast, and suffix KV is batched per word.
4. **Pipelining.** Per-chunk CUDA streams (a fixed pool, one per slot) with N chunks in flight. PyTorch device-to-device copies sync both devices' current streams, which otherwise serialises layer-sharded GPUs.

## Tools in fast_tools/

- **test_fast.py:** greedy outputs vs a plain HF forward-with-hooks decode. Reports logit gaps on mismatches.
- **bench_orig.py:** times the original `run_steer2` on N windows and dumps its derangement.
- **compare.py:** compares the original CSV with fast `*_orig_*` columns. They agree within bf16 noise.
- **crosstab.py:** row-level view of what the model outputs when the old metric says "correct".
- **analyze.py, table.py:** summaries and heatmaps.
- **kprof.py, syncdbg.py, micro.py, pf.py, flame_summ.py:** profiling helpers.

## Results

Report: https://cute.sus.cat/arena-tmp-share/2026-11-06-latent-steer-fast/
