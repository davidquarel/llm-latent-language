# summarise py-spy raw (collapsed) output: self time by line and inclusive time by function within fast_steer.py
import sys, collections, re
incl = collections.Counter(); selfc = collections.Counter(); tot = 0
for line in open(sys.argv[1]):
    stack, n = line.rstrip().rsplit(" ", 1); n = int(n); tot += n
    frames = stack.split(";")
    seen = set()
    for f in frames:
        if "fast_steer.py" in f:
            key = re.sub(r"\(.*fast_steer.py:", "(fast_steer.py:", f)
            if key not in seen: incl[key] += n; seen.add(key)
    selfc[frames[-1][:150]] += n
print("total samples", tot)
print("--- inclusive, fast_steer.py lines ---")
for k, v in incl.most_common(25): print(f"{100*v/tot:5.1f}%  {k}")
print("--- self (leaf) ---")
for k, v in selfc.most_common(15): print(f"{100*v/tot:5.1f}%  {k}")
