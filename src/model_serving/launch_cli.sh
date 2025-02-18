#!/bin/bash
sessname=tmux_registry_$1

root_dir=src/model_serving
cd $root_dir

# Create a new session named "$sessname", and run command
tmux new-session -d -s $sessname
tmux send-keys -t $sessname "source ~/.bashrc" Enter
tmux send-keys -t $sessname "source activate base" Enter
tmux send-keys -t $sessname "cd $root_dir" Enter
tmux send-keys -t $sessname "conda activate health_q" Enter
tmux send-keys -t $sessname "redis-server --daemonize yes" Enter
tmux send-keys -t $sessname "python model_registry_server.py --port $1 --model_name meta-llama/Llama-3.1-405B-Instruct-FP8" Enter

# wait until the server is ready (check whether output contains http)
while true; do
    output="$(tmux capture-pane -p -t $sessname)"
    if echo "$output" | grep -q "Running on http://"; then
        break
    fi
    sleep 1
done

registry_url="$(
  echo "$output" \
    | grep -F "* Running on http:" \
    | grep -F "$1" \
    | tail -n 1 \
    | head -n 1 \
    | sed -E 's/.*Running on (http[^ ]+).*/\1/'
)"

echo "$registry_url" >> $root_dir/registry_urls.txt

# Attach to session named "$sessname"
#tmux attach -t "$sessname"