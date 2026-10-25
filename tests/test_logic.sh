#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2034,SC2116,SC2154,SC2181,SC2319,SC2320,SC2329  # wzorce celowe w harnessie testowym
# Testy logiki start.sh bez GPU: atrapa nvidia-smi + podmienione funkcje testujące serwer.
# Niczego nie instaluje i nie uruchamia modelu. Użycie: bash tests/test_logic.sh
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
check() {  # opis, warunek (kod wyjścia poprzedniej komendy w $?)
  if [[ "$2" == "0" ]]; then echo "PASS  $1"; PASS=$((PASS + 1)); else echo "FAIL  $1"; FAIL=$((FAIL + 1)); fi
}

mkdir -p "$TMP/fakebin"
cat > "$TMP/fakebin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *compute_cap*) echo "8.6" ;;
  *memory.total,memory.used,memory.free*) echo "49140, 30000, 19140" ;;
  *name,memory.total,driver_version*) echo "NVIDIA RTX A6000, 49140, 550.54.15" ;;
  "") echo "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |" ;;
esac
EOF
chmod +x "$TMP/fakebin/nvidia-smi"

export WORKSPACE="$TMP/ws" STATE_DIR="$TMP/ws/state" LLAMA_DIR="$TMP/ws/llama.cpp"
export PATH="$TMP/fakebin:$PATH"
QWEN_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
export QWEN_PORT
mkdir -p "$WORKSPACE" "$STATE_DIR/logs"

# shellcheck source=SCRIPTDIR/../start.sh
source "$ROOT/start.sh"
set +e; trap - ERR
C_R=""; C_G=""; C_Y=""; C_B=""; C_W=""; C_0=""

# --- GPU ---------------------------------------------------------------------
out=$(detect_gpu 2>&1); detect_gpu >/dev/null 2>&1
[[ "$GPU_NAME" == "NVIDIA RTX A6000" && "$VRAM_TOTAL_MIB" == "49140" && "$CUDA_DRIVER_VER" == "12.4" && "$CUDA_ARCHS" == "86" ]]
check "detect_gpu parsuje nazwę, VRAM, wersję CUDA i compute capability ($out)" $?

# --- nvidia-smi zwraca błąd ----------------------------------------------------
mkdir -p "$TMP/badbin"; printf '#!/bin/sh\necho "NVIDIA-SMI has failed"; exit 9\n' > "$TMP/badbin/nvidia-smi"; chmod +x "$TMP/badbin/nvidia-smi"
out=$(PATH="$TMP/badbin:$PATH" bash -c "source '$ROOT/start.sh'; detect_gpu" 2>&1); rc=$?
[[ $rc -ne 0 && "$out" == *"nvidia-smi zwrócił błąd"* ]]
check "detect_gpu kończy się czytelnym błędem, gdy sterownik nie działa" $?

# --- wybór wersji CUDA toolkit -------------------------------------------------
pick_minor() { local d=$1 M m; M=$(version_major "$d"); m=$(version_minor "$d")
  if (( M >= 13 )); then echo 8; elif (( M == 12 )); then echo $(( m > 8 ? 8 : m )); else echo none; fi; }
[[ "$(pick_minor 12.4)" == 4 && "$(pick_minor 12.9)" == 8 && "$(pick_minor 13.0)" == 8 && "$(pick_minor 11.8)" == none ]]
check "logika wyboru cuda-toolkit-12-x (12.4->12-4, 12.9->12-8, 13.0->12-8, 11.8->błąd)" $?
grep -q 'want_minor=$(( drv_minor > 8 ? 8 : drv_minor ))' "$ROOT/start.sh"
check "start.sh używa tej samej reguły wyboru wersji" $?

# --- vram_stats ----------------------------------------------------------------
[[ "$(vram_stats)" == "49140 30000 19140" ]]
check "vram_stats sumuje total/used/free z nvidia-smi" $?

# --- port: wolny / zajęty przez obcy proces -------------------------------------
port_in_use; [[ $? -ne 0 ]]; check "port_in_use: wolny port" $?
python3 -c "import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',$QWEN_PORT));s.listen();time.sleep(8)" &
LPID=$!; sleep 1
port_in_use; check "port_in_use: zajęty port" $?
out=$(check_port_free_or_ours 2>&1); rc=$?
[[ $rc -ne 0 && "$out" == *"Nie zatrzymuję cudzych procesów"* ]] && kill -0 "$LPID" 2>/dev/null
check "obcy proces na porcie: przerwanie bez zabijania procesu" $?
kill "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null

# --- wybór konfiguracji: 256K OOM (q8_0 i q4_0), 128K OK -------------------------
LLAMA_COMMIT_USED=$LLAMA_CPP_COMMIT; GPU_COUNT=1; NATIVE_CTX=262144
CALLS=()
test_config() { CALLS+=("$1/$2"); METRICS[gen_tps]=95
  if [[ "$1" == 262144 ]]; then FAIL_REASON="brak pamięci (symulacja)"; return 2; fi
  TEST_RESULT[long_context]=PASS; return 0; }
stop_server() { :; }
select_and_run >/dev/null 2>&1
[[ "${CALLS[*]}" == "262144/q8_0 262144/q4_0 131072/q8_0" && "$CHOSEN_CTX" == 131072 && "$CHOSEN_KV" == q8_0 ]]
check "kolejność prób: 256K q8_0 -> 256K q4_0 -> 128K q8_0 (wybrano $CHOSEN_CTX/$CHOSEN_KV)" $?
[[ -f "$SELECTED_FILE" ]] && grep -q '^SEL_CTX=131072$' "$SELECTED_FILE"
check "zapis wybranej konfiguracji do selected.env" $?

# --- błąd funkcjonalny: brak próby q4_0, przejście do mniejszego kontekstu --------
CALLS=(); rm -f "$SELECTED_FILE"
test_config() { CALLS+=("$1/$2"); if [[ "$1" == 262144 ]]; then FAIL_REASON="tool calling"; return 1; fi; return 0; }
select_and_run >/dev/null 2>&1
[[ "${CALLS[*]}" == "262144/q8_0 131072/q8_0" ]]
check "błąd funkcjonalny nie wymusza próby q4_0 ($(echo "${CALLS[*]}"))" $?

# --- natywny kontekst mniejszy niż kandydat -------------------------------------
CALLS=(); rm -f "$SELECTED_FILE"; NATIVE_CTX=131072
test_config() { CALLS+=("$1/$2"); return 0; }
out=$(select_and_run 2>&1); select_and_run >/dev/null 2>&1
[[ "${CALLS[0]}" == "131072/q8_0" && "$out" == *"Pomijam kontekst 262144"* ]]
check "kontekst większy niż natywny jest pomijany z ostrzeżeniem" $?
NATIVE_CTX=262144

# --- nic nie przechodzi -> błąd ------------------------------------------------
rm -f "$SELECTED_FILE"
out=$(test_config() { FAIL_REASON="OOM"; return 2; }; select_and_run 2>&1); rc=$?
[[ $rc -ne 0 && "$out" == *"Żadna konfiguracja"* ]]
check "brak działającej konfiguracji kończy się błędem z listą prób" $?

# --- ponowne uruchomienie: zapisana konfiguracja bez pełnych testów ------------
save_selection 131072 q8_0 >/dev/null
CALLS=()
test_config() { CALLS+=("full"); return 0; }
start_server() { echo "start $1 $2" >> "$TMP/starts"; }
wait_ready() { return 0; }
check_offload() { return 0; }
run_apitest() { TEST_RESULT[tool_call]=PASS; return 0; }
select_and_run >/dev/null 2>&1
[[ ${#CALLS[@]} -eq 0 && "$FROM_CACHE" == 1 && "$CHOSEN_CTX" == 131072 && "$(cat "$TMP/starts")" == "start 131072 q8_0" ]]
check "drugie uruchomienie używa selected.env bez ponownych testów 256K/128K" $?

FORCE_RETEST=1; CALLS=(); select_and_run >/dev/null 2>&1; FORCE_RETEST=0
[[ "${CALLS[*]}" == "full" ]]
check "FORCE_RETEST=1 wymusza pełne testy" $?

GPU_NAME="NVIDIA RTX 4090"; CALLS=(); select_and_run >/dev/null 2>&1; GPU_NAME="NVIDIA RTX A6000"
[[ "${CALLS[*]}" == "full" ]]
check "zmiana GPU unieważnia zapisaną konfigurację" $?

# --- opencode.json ---------------------------------------------------------------
write_opencode_config 131072
python3 - "$OPENCODE_FILE" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
p = c["provider"]["vast"]
m = p["models"]["qwen-coder"]
assert p["npm"] == "@ai-sdk/openai-compatible"
assert p["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
assert m["limit"] == {"context": 131072, "output": 8192}
assert m["tool_call"] is True and c["model"] == "vast/qwen-coder"
EOF
check "opencode.json: provider vast, baseURL :8080/v1, context=131072, output=8192" $?

# --- składnia --------------------------------------------------------------------
bash -n "$ROOT/start.sh"; check "bash -n start.sh" $?

echo; echo "Wynik: $PASS PASS, $FAIL FAIL"
(( FAIL == 0 ))
