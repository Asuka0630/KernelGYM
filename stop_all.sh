#!/bin/bash
# Stop KernelGym API server, workers, and monitor, then optionally clear Redis keys.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

REDIS_HOST="localhost"
REDIS_PORT=""
REDIS_PASSWORD=""
REDIS_KEY_PREFIX="kernelgym"
API_PORT=""

if [ -f "${ENV_FILE}" ]; then
    API_PORT="$(grep "^API_PORT=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
    REDIS_HOST="$(grep "^REDIS_HOST=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
    REDIS_PORT="$(grep "^REDIS_PORT=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
    REDIS_PASSWORD="$(grep "^REDIS_PASSWORD=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
    REDIS_KEY_PREFIX="$(grep "^REDIS_KEY_PREFIX=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d ' ')"
fi

if [ -z "${REDIS_HOST}" ]; then
    REDIS_HOST="localhost"
fi
if [ -z "${REDIS_KEY_PREFIX}" ]; then
    REDIS_KEY_PREFIX="kernelgym"
fi

kill_processes() {
    local pattern="$1"
    local description="$2"

    echo "Stopping ${description}..."
    local pids=""
    if command -v pgrep >/dev/null 2>&1; then
        pids="$(pgrep -f "${pattern}" || true)"
    else
        pids="$(ps aux | grep "${pattern}" | grep -v grep | awk '{print $2}' || true)"
    fi

    if [ -z "${pids}" ]; then
        echo "No ${description} processes found."
        return
    fi

    echo "${pids}" | xargs -r kill -TERM || true
    sleep 2

    local remaining=""
    if command -v pgrep >/dev/null 2>&1; then
        remaining="$(pgrep -f "${pattern}" || true)"
    else
        remaining="$(ps aux | grep "${pattern}" | grep -v grep | awk '{print $2}' || true)"
    fi

    if [ -n "${remaining}" ]; then
        echo "Force killing ${description}..."
        echo "${remaining}" | xargs -r kill -KILL || true
    fi
}

echo "Stopping KernelGym processes..."

kill_processes "kernelgym.server.api.server" "KernelGym API server"
kill_processes "kernelgym.worker.worker_monitor" "KernelGym worker monitor"
kill_processes "kernelgym.worker.single_worker" "KernelGym GPU workers"
kill_processes "kernelgym.worker.gpu_worker" "KernelGym GPU worker core"
kill_processes "kernelgym.worker.compile_service" "KernelGym compile service"
kill_processes "uvicorn.*kernelgym" "Uvicorn server"

echo "Stopping multiprocessing worker processes..."

# Collect all KernelGym-related PIDs (including any we may have missed)
KG_PIDS=""
if command -v pgrep >/dev/null 2>&1; then
    KG_PIDS="$(pgrep -f "kernelgym" || true)"
else
    KG_PIDS="$(ps aux | grep "[k]ernelgym" | awk '{print $2}' || true)"
fi

# Kill multiprocessing processes ONLY if they are children of a KernelGym process.
kill_multiprocessing_if_kg_child() {
    local pattern="$1"
    local description="$2"

    echo "Stopping ${description}..."

    local all_mp_pids=""
    if command -v pgrep >/dev/null 2>&1; then
        all_mp_pids="$(pgrep -f "${pattern}" || true)"
    else
        all_mp_pids="$(ps aux | grep "${pattern}" | grep -v grep | awk '{print $2}' || true)"
    fi

    if [ -z "${all_mp_pids}" ]; then
        echo "No ${description} processes found."
        return
    fi

    # Filter: only keep processes whose parent is a KernelGym process
    local target_pids=""
    for mp_pid in ${all_mp_pids}; do
        local ppid=""
        ppid="$(ps -o ppid= -p "${mp_pid}" 2>/dev/null | tr -d ' ' || true)"
        if [ -n "${ppid}" ] && echo "${KG_PIDS}" | grep -qw "${ppid}"; then
            target_pids="${target_pids} ${mp_pid}"
        fi
    done

    # Also check grandparent (resource_tracker's parent is the spawn worker,
    # whose parent is the kernelgym process)
    for mp_pid in ${all_mp_pids}; do
        # Skip if already included
        if echo "${target_pids}" | grep -qw "${mp_pid}"; then
            continue
        fi
        local ppid=""
        ppid="$(ps -o ppid= -p "${mp_pid}" 2>/dev/null | tr -d ' ' || true)"
        if [ -n "${ppid}" ]; then
            local gppid=""
            gppid="$(ps -o ppid= -p "${ppid}" 2>/dev/null | tr -d ' ' || true)"
            if [ -n "${gppid}" ] && echo "${KG_PIDS}" | grep -qw "${gppid}"; then
                target_pids="${target_pids} ${mp_pid}"
            fi
        fi
    done

    if [ -z "${target_pids}" ]; then
        echo "No ${description} processes found (filtered by KernelGym ancestry)."
        return
    fi

    echo "${target_pids}" | xargs -r kill -TERM 2>/dev/null || true
    sleep 2

    # Check remaining and force-kill
    local remaining=""
    for mp_pid in ${target_pids}; do
        if kill -0 "${mp_pid}" 2>/dev/null; then
            remaining="${remaining} ${mp_pid}"
        fi
    done

    if [ -n "${remaining}" ]; then
        echo "Force killing ${description}..."
        echo "${remaining}" | xargs -r kill -KILL 2>/dev/null || true
    fi
}

kill_multiprocessing_if_kg_child "multiprocessing.spawn" "multiprocessing spawn workers"
kill_multiprocessing_if_kg_child "multiprocessing.resource_tracker" "multiprocessing resource tracker"

if command -v redis-cli >/dev/null 2>&1; then
    if [ -n "${REDIS_PORT}" ]; then
        REDIS_AUTH_ARGS=()
        if [ -n "${REDIS_PASSWORD}" ]; then
            REDIS_AUTH_ARGS=(-a "${REDIS_PASSWORD}" --no-auth-warning)
        fi
        echo "Clearing Redis keys with prefix '${REDIS_KEY_PREFIX}:' on ${REDIS_HOST}:${REDIS_PORT}..."
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" \
            --scan --pattern "${REDIS_KEY_PREFIX}:*" \
            | xargs -r redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" DEL >/dev/null 2>&1 || true

        echo "Shutting down Redis server on ${REDIS_HOST}:${REDIS_PORT}..."
        if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" SHUTDOWN 2>/dev/null; then
            echo "Redis server on port ${REDIS_PORT} shut down gracefully."
        else
            # Fallback: kill redis-server process bound to this port
            echo "redis-cli SHUTDOWN failed, trying to kill redis-server on port ${REDIS_PORT}..."
            REDIS_PID="$(ss -tlnp 2>/dev/null | grep ":${REDIS_PORT} " | grep -oP 'pid=\K[0-9]+' || true)"
            if [ -z "${REDIS_PID}" ]; then
                REDIS_PID="$(lsof -tiTCP:"${REDIS_PORT}" -sTCP:LISTEN 2>/dev/null || true)"
            fi
            if [ -n "${REDIS_PID}" ]; then
                kill -TERM "${REDIS_PID}" 2>/dev/null || true
                sleep 1
                if kill -0 "${REDIS_PID}" 2>/dev/null; then
                    kill -KILL "${REDIS_PID}" 2>/dev/null || true
                fi
                echo "Redis server (PID ${REDIS_PID}) on port ${REDIS_PORT} killed."
            else
                echo "No redis-server process found on port ${REDIS_PORT}; maybe already stopped."
            fi
        fi
    else
        echo "REDIS_PORT not set; skipping Redis cleanup."
    fi
else
    echo "redis-cli not found; skipping Redis cleanup."
fi

echo "KernelGym stopped."
