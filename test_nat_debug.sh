#!/bin/bash
set -u
echo "=== 1. Nemotron direct ping ==="
curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"nemotron-3-nano","messages":[{"role":"user","content":"Reply with the single word: PONG"}],"max_tokens":20}' | python3 -m json.tool

echo ""
echo "=== 2. User nat agent (explicit tool-call request) ==="
curl -sS -X POST http://localhost:9000/api/agent/nat -H "Content-Type: application/json" -d '{"query":"Call find_resources for shelters in Manhattan and return the first one."}' | python3 -m json.tool

echo ""
echo "=== 3. Admin nat agent (city stats) ==="
curl -sS -X POST http://localhost:9001/api/admin/agent/nat -H "Content-Type: application/json" -d '{"query":"Give me city stats"}' | python3 -m json.tool
