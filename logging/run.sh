sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 1215,1410
sudo -E nohup nice -n -20 bash -c '
for i in $(seq 1 10); do
  echo "--- Starting Sequential Run $i at $(date) ---"
  /opt/python/3.10/bin/python3 -u cifar10_speedrun.py > logs/run_${i}.log 2>&1
  rm -r cifar10
done
sudo nvidia-smi -pm 0
sudo nvidia-smi -rac
echo "--- All 10 runs completed at $(date) ---"
' </dev/null > logs/sequential_master.log 2>&1 &
