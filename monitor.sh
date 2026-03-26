
#!/bin/bash
LOG_FILE="/home/jerry/project/hardware_log.csv"
echo "Timestamp, CPU_Idle(%), RAM_Used(MB), GPU_Util(%), GPU_Mem_Used(MB)" > "$LOG_FILE"
echo "Starting monitoring... Press Ctrl+C to stop or kill the process in background"

while true; do
  TS=$(date "+%Y-%m-%d %H:%M:%S")
  # CPU idle % 
  CPU_IDLE=$(top -bn1 | grep "Cpu(s)" | awk -F',' '{print $4}' | awk '{print $1}')
  if [ -z "$CPU_IDLE" ]; then
    CPU_IDLE="0"
  fi
  # RAM used MB
  RAM_USED=$(free -m | awk '/^Mem:/ {print $3}')
  # GPU util
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  # GPU RAM used
  GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  
  echo "$TS, $CPU_IDLE, $RAM_USED, $GPU_UTIL, $GPU_MEM" >> "$LOG_FILE"
  sleep 1
done
