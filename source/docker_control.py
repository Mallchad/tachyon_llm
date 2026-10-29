# open_webui
docker run -d -p 8009:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data -e WEBUI_SECRET_KEY=your-secret-key --name open_webui --replace --restart always --network llm ghcr.io/open-webui/open-webui:main

# lite_llm

# WARNING: Running it through sed and "-f -" changes the root from the docker
# yml to the working directory
sed --expression "s/<litellm_master_key>/${LITE_LLM_MASTER_KEY:?}/" --expression "s/<litellm_salt_key>/${LITE_LLM_SALT_KEY:?}/"  source/lite_llm_docker.yml | podman compose -f - --project-name tachyon_llm --podman-run-args="--replace"  up -d

# litellm compose from web
curl -sSL https://docs.litellm.ai/docker-compose.yml | docker compose -f - up -d

# opencode
podman run --name opencode --replace --publish 8008:49374 --network "llm,tachyon_llm_llm" --userns=keep-id -v /local/repos:/local/repos -v opencode_config:/home/mallchad ghcr.io/anomalyco/opencode web --hostname "0.0.0.0" --port 49374
# opencode 2.0 version
# commands are different
podman run --detach --name opencode --replace --publish 8008:49374 --network "llm,tachyon_llm_llm" --userns=keep-id -v /local/repos:/local/repos -v opencode_config:/home/mallchad ghcr.io/anomalyco/opencode:2.0.0 serve --hostname "0.0.0.0" --port 49374
