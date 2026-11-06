set -x
P=~/david_quarel/ARENA_3.0/.venv/bin/python
A="--model_name /workspace/hfcache/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d --devices 0,1,2 --inflight 3 --cfg_chunk 32 --layer_split 1.4,1,0.75 --shuffle_path bench/orig_l2_7b.csv.shuffle.pt --out out_fast"
$P fast_steer.py $A --latent_lang en --direction first --write_prompt orig
$P fast_steer.py $A --latent_lang es --direction first --write_prompt orig
$P fast_steer.py $A --latent_lang en --direction first --write_prompt own
$P fast_steer.py $A --latent_lang es --direction mean --write_prompt orig
$P fast_steer.py $A --latent_lang en --direction mean --write_prompt orig
