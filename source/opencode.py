

# podman run --name opencode --replace --publish 8008:49374 --network "llm,tachyon_llm_llm" --userns=keep-id -v /local/repos:/local/repos -v opencode_config:/home/mallchad ghcr.io/anomalyco/opencode web --hostname "0.0.0.0" --port 49374
