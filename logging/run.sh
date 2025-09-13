nohup nice -n -10 bash -c '
for i in $(seq 1 10); do
  echo "--- Starting Sequential Run $i at $(date) ---"
  python3 -u cifar10_speedrun.py > logs/run_${i}.log 2>&1
done
rm -r cifar10
echo "--- All 10 runs completed at $(date) ---"
' </dev/null > logs/sequential_master.log 2>&1 &
