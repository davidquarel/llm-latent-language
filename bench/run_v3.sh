while ps aux | grep -v grep | grep -q "queue_runner.py bench/jobs_v2.txt"; do sleep 30; done
~/david_quarel/ARENA_3.0/.venv/bin/python fast_tools/queue_runner.py bench/jobs_v3.txt
