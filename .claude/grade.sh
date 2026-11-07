#!/bin/bash
# grade do passo 6: v0.4.0 e v0.3.0 x {1GB,2GB}; ordenados e u; 3 runs
cd /c/Users/blzro/Desktop/projects/nini-datacomp
CASES="1000000x10 1000000x50 1000000x300 10000000x10 10000000x50 10000000x300 1000000x10u 1000000x50u 1000000x300u 10000000x10u 10000000x50u"
t0=$(date +%s)
for ver in 040; do
  for lim in 1GB 2GB; do
    free=$(python -c "import shutil;print(int(shutil.disk_usage('.').free/2**30))")
    if [ "$free" -lt 15 ]; then echo "PARADO: disco $free GB"; exit 1; fi
    out=benchmarks/results/grade_${ver}_${lim}.json
    echo "== $ver $lim $(date +%T) (livre ${free} GB)"
    if [ $ver = 040 ]; then proj=(); else proj=(--project ../nini-datacomp-v030); fi
    uv run "${proj[@]}" python tools/bench_hash.py --cases $CASES --runs 3 --memory-limit $lim --out-json $out 2>&1 | grep -v "^  .*unordered/\|^  [0-9]*x[0-9]*:"
  done
done
echo "FIM $(( ($(date +%s)-t0)/60 )) min"
