# 2026-05-12 118 LLM Ops Repo Sync Memo

## Scope

- Synchronized the local source mirror with the live 118 server runtime at `/home/metroai/llm-ops`.
- Included the current `app.py`, `README.md`, and `llm-ops.env.example`.
- Preserved the live-side GPU auto-refresh change so the web console now refreshes GPU and model runtime state every 10 seconds.

## Live Runtime

- Host: `192.168.1.118`
- Runtime path: `/home/metroai/llm-ops`
- Service: `metro-llm-ops.service`
- Web console: `http://192.168.1.118:8090/`

## Standard Source Path

- Local tracked source mirror: `/home/metro/work/llm-ops-118`
- This folder is intended to be the git record for 118 server changes.
- Before future pushes or snapshots, sync from the live 118 runtime first if hotfixes were applied directly on the server.

## Verified State

- `metro-llm-ops.service` restarted successfully after the GPU auto-refresh update.
- `GET /api/health` returned service, NMS monitor, autopilot, and GPU state successfully.
- Forced `POST /api/nms/autopilot/run` completed successfully with one selected target and one successful analysis.

## Notes

- GPU utilization can remain `0%` between requests even when VRAM is occupied by a loaded model.
- The console now updates GPU/VRAM/runtime metrics without requiring a manual action or a full page reload.
