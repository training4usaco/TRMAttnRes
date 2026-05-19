for ckpt in checkpoints/sudoku_v2/step_*.pt; do
    step=$(echo "$ckpt" | grep -o '[0-9]*\.pt' | grep -o '[0-9]*')
    echo -n "Step $step: "
    python eval.py --benchmark sudoku --checkpoint "$ckpt" 2>&1 | tail -1
done
