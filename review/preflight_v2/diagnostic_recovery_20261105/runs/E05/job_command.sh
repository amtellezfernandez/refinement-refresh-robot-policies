F=$HOME/alba-cross-policy-20261101/ckpt/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json; sha256sum $F; python3 -c "
import json,sys; s=json.load(open(\"$F\"))[\"norm_stats\"]
for g in s: print(g, {k:[round(x,4) for x in v[:8]] for k,v in s[g].items()})
"