#!/bin/bash

# Führe den ersten curl-Befehl aus und speichere die Ausgabe in einer Variablen
response=$(curl -k -s https://mlpc.coder.aifb.kit.edu:9443/api/endpoints/12/docker/containers/json --header 'X-API-Key: ptr_tmPi4pAYbfxZp95DIicqieo7WuxT9RpF3AlIDIEK/0g=')

# Extrahiere die ID des Containers mit dem Namen "/Shared_LocalAI"
container_id=$(echo $response | jq -r '.[] | select(.Names[] == "/Shared_LocalAI") | .Id')
# Überprüfe, ob eine ID extrahiert wurde
if [ -z "$container_id" ]; then
    echo "Keine Container-ID gefunden"
    exit 1
fi

# Gib die extrahierte ID aus
echo "Extrahierte Container-ID: $container_id"

# Führe den zweiten curl-Befehl mit der extrahierten ID aus
# curl -k -X POST "https://example.com/api/use-container" -H "Content-Type: application/json" -d "{\"container_id\": \"$container_id\"}"
curl -k --request POST --url https://mlpc.coder.aifb.kit.edu:9443/api/endpoints/12/docker/containers/$container_id/restart --header 'X-API-Key: ptr_tmPi4pAYbfxZp95DIicqieo7WuxT9RpF3AlIDIEK/0g='
echo "Container neu gestartet"