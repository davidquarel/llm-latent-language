#!/bin/bash
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_fr_en_zh --src_lang fr --dest_lang zh --latent_lang en
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_fr_en_de --src_lang fr --dest_lang de --latent_lang en
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_fr_zh_en --src_lang fr --dest_lang en --latent_lang zh
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_fr_zh_de --src_lang fr --dest_lang de --latent_lang zh
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_fr_de_en --src_lang fr --dest_lang en --latent_lang de
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_fr_de_zh --src_lang fr --dest_lang zh --latent_lang de
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_en_fr_zh --src_lang en --dest_lang zh --latent_lang fr
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_en_fr_de --src_lang en --dest_lang de --latent_lang fr
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_en_zh_fr --src_lang en --dest_lang fr --latent_lang zh
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_en_zh_de --src_lang en --dest_lang de --latent_lang zh
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_en_de_fr --src_lang en --dest_lang fr --latent_lang de
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_en_de_zh --src_lang en --dest_lang zh --latent_lang de
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_zh_fr_en --src_lang zh --dest_lang en --latent_lang fr
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_zh_fr_de --src_lang zh --dest_lang de --latent_lang fr
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_zh_en_fr --src_lang zh --dest_lang fr --latent_lang en
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_zh_en_de --src_lang zh --dest_lang de --latent_lang en
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_zh_de_fr --src_lang zh --dest_lang fr --latent_lang de
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_zh_de_en --src_lang zh --dest_lang en --latent_lang de
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_de_fr_en --src_lang de --dest_lang en --latent_lang fr
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_de_fr_zh --src_lang de --dest_lang zh --latent_lang fr
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_de_en_fr --src_lang de --dest_lang fr --latent_lang en
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_de_en_zh --src_lang de --dest_lang zh --latent_lang en
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_de_zh_fr --src_lang de --dest_lang fr --latent_lang zh
python3 llama_experiment_template.py --use_tuned_lens False --log_file out/reject_lang_sweep/hook_reject_de_zh_en --src_lang de --dest_lang en --latent_lang zh