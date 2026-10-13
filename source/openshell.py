
import os
from typing import NamedTuple

import tachyon_library as tyon

xdg_runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
listen_address = "0.0.0.0"
gateway_port = 8007
gateway_address_and_port = f"{listen_address}:{gateway_port}:{gateway_port}"
gateway_port_bind = f"{gateway_port}:{gateway_port}"
tyon.tyon_log( f"XDG_RUNTIME_DIR {xdg_runtime_dir}" )

## NOTE: The nVIDIA documentation for podman and openshell is completely
## AI hallucinated, don't use it
tyon.execute_list ( [
    "podman",
    "run",
    "-d",
    "--name", "openshell-gateway",
    "--replace",
    "-p", gateway_port_bind,

    # Hack if we have file permissions issues from bind mount, don't use
    # NOTE: Using keep-id itself is fine but many docker containers expect root access, so it's iffy
    # "--userns=keep-id",

    ## Bind Mounts
    # Connect to podman control socket
    # NOTE: needs to match user permissions
    "-v", f"{xdg_runtime_dir}/podman/podman.sock:/var/run/podman.sock:U",

    ## Bind Volumes
    "-v", "openshell-state:/var/openshell",
    # Writes to this directory for some reason but no associated volume, so we provide our own
    "-v", "openshell-app-state:/app/.local",
    # Mount openshell runtime data into the host
    "-e", f"XDG_RUNTIME_DIR={xdg_runtime_dir}/openshell-state",
    "-e", "OPENSHELL_COMPUTE_DRIVER=podman",
    "-e", "OPENSHELL_PODMAN_SOCKET=/var/run/podman.sock",
    "-e", "OPENSHELL_DB_URL=sqlite:/var/openshell/openshell.db",
    "-e", "OPENSHELL_DISABLE_TLS=true",
    "ghcr.io/nvidia/openshell/gateway:latest",
    "--bind-address", listen_address,
    "--port", "8007",
    "--disable-tls"
])
