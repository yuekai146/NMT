pids=$(lsof /dev/nvidia* | awk '{print $2}' | uniq | tail -n 8)
kill -9 $pids
