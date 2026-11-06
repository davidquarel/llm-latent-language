while pgrep -f "bash bench/run_(sweeps|ko).sh" > /dev/null; do sleep 30; done
~/david_quarel/ARENA_3.0/.venv/bin/python fast_tools/crosstab.py /workspace/hfcache/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d "(7,22)" "fr,de,ru,zh;fr,zh,ru,de;de,ru,zh,fr;fr,de,zh,ru;de,fr,ru,zh;zh,de,fr,ru"
