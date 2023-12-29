#!/bin/bash

log_file="datasets/arxiv_qa/generate_synth.log"
folder_path="datasets/arxiv_qa/"

while true; do
    current_time=$(date +"%Y-%m-%d %H:%M:%S")
    folder_size=$(du -sh "$folder_path" | awk '{print $1}')
    file_count=$(find "$folder_path" -type f | wc -l)

    echo "$current_time - Folder Size: $folder_size - File Count: $file_count" >> "$log_file"

    sleep 60  # Adjust the time interval (in seconds) between each log entry
done
