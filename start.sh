#!/usr/bin/env bash
# =============================================================================
#  vast-qwen35-abliterated / start.sh
#
#  Jednokomendowy instalator i launcher modelu Qwen3.5-35B-A3B Abliterated
#  (GGUF Q4_K_M) na llama.cpp + CUDA, przygotowany pod Vast.ai.
#
#  Użycie:
#    bash start.sh            instalacja (jeśli trzeba) + wybór kontekstu + start serwera
#    bash start.sh retest     jak wyżej, ale ponownie testuje wszystkie konfiguracje
#    bash start.sh test       szybkie testy API działającego serwera
#    bash start.sh status     stan serwera i wybrana konfiguracja
#    bash start.sh stop       zatrzymuje serwer uruchomiony przez ten skrypt
#    bash start.sh help       pomoc
#
#  Wszystkie ustawienia z sekcji "Ustawienia" można nadpisać zmiennymi
#  środowiskowymi, np.:  LONG_TEST=0 bash start.sh
# =============================================================================
set -Eeuo pipefail

readonly SCRIPT_VERSION="1.0.0"

# ---- Ustawienia (można nadpisać zmiennymi środowiskowymi) -------------------
WORKSPACE="${WORKSPACE:-/workspace}"
LLAMA_DIR="${LLAMA_DIR:-$WORKSPACE/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE/models/Qwen3.5-35B-A3B-abliterated-GGUF}"
STATE_DIR="${STATE_DIR:-$WORKSPACE/vast-qwen35}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8081}"
CTX_CANDIDATES="${CTX_CANDIDATES:-262144 131072 65536 32768}"  # kolejność prób
KV_TYPES="${KV_TYPES:-q8_0 q4_0}"            # q4_0 tylko gdy q8_0 nie mieści się w VRAM
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-1024}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-1536}"  # wymagany zapas VRAM po teście długiego kontekstu
MIN_GEN_TPS="${MIN_GEN_TPS:-10}"              # poniżej tej prędkości generowania konfiguracja jest odrzucana
LONG_TEST="${LONG_TEST:-1}"                   # 1 = test wypełnienia kontekstu (~90%)
LONG_TEST_FRACTION="${LONG_TEST_FRACTION:-0.9}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-900}"           # sekundy na załadowanie modelu
LONG_TEST_TIMEOUT="${LONG_TEST_TIMEOUT:-2400}"
FORCE_RETEST="${FORCE_RETEST:-0}"
KEEP_LLAMA_CHECKOUT="${KEEP_LLAMA_CHECKOUT:-0}"  # 1 = nie przełączaj istniejącego llama.cpp na przypiętą wersję

# ---- Przypięte wersje ------------------------------------------------------
readonly LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
readonly LLAMA_CPP_TAG="v0.7.0"                                     # = build b11669
readonly LLAMA_CPP_COMMIT="620d2225cc436ab0b001e3200e590e3b286ef9ab"
readonly HF_REPO="amarck/Qwen3.5-35B-A3B-abliterated-GGUF"
readonly HF_REVISION="1d077ff70a3a61804892f7e848de466bad78ba20"
readonly MODEL_NAME="Qwen3.5-35B-A3B Abliterated"
readonly MODEL_QUANT="Q4_K_M"
readonly MODEL_ALIAS="qwen-coder"
# Pliki modelu: "nazwa|rozmiar w bajtach|sha256". Przy modelu dzielonym
# (…-00001-of-0000N.gguf) wpisz wszystkie części; serwer dostaje pierwszą,
# a llama.cpp sam doczytuje pozostałe.
readonly MODEL_FILES=(
  "abliterated_Q4_K_M.gguf|21169116896|e7335840ad8542dbe0b2cf895d298c85da5e67aa7bce8bcef467a8d599b1ae6e"
)
# Zalecane przez Qwen próbkowanie do kodowania (tryb thinking). Klient może je nadpisać.
readonly SERVER_SAMPLING=(--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0)

LOG_DIR="$STATE_DIR/logs"
BIN_DIR="$STATE_DIR/bin"
SERVER_LOG="$LOG_DIR/llama-server.log"
PID_FILE="$STATE_DIR/llama-server.pid"
CMD_FILE="$STATE_DIR/llama-server.cmd"
SELECTED_FILE="$STATE_DIR/selected.env"
OPENCODE_FILE="$STATE_DIR/opencode.json"
REPORT_FILE="$STATE_DIR/report.txt"
BUILD_DIR="$LLAMA_DIR/build-vastqwen"   # osobny katalog, żeby nie ruszać ewentualnego build/ użytkownika
LLAMA_BIN="$BUILD_DIR/bin/llama-server"
MODEL_FIRST="${MODEL_FILES[0]%%|*}"
MODEL_PATH="$MODEL_DIR/$MODEL_FIRST"
MODEL_SHA="${MODEL_FILES[0]##*|}"

# ---- Wyjście ---------------------------------------------------------------
if [[ -t 1 ]]; then
  C_R=$'\e[31m'; C_G=$'\e[32m'; C_Y=$'\e[33m'; C_B=$'\e[36m'; C_W=$'\e[1m'; C_0=$'\e[0m'
else
  C_R=""; C_G=""; C_Y=""; C_B=""; C_W=""; C_0=""
fi
info() { printf '%s[INFO]%s %s\n' "$C_B" "$C_0" "$*"; }
ok()   { printf '%s[ OK ]%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '%s[UWAGA]%s %s\n' "$C_Y" "$C_0" "$*"; }
die()  { printf '%s[BŁĄD]%s %s\n' "$C_R" "$C_0" "$*"; exit 1; }
step() { printf '\n%s==> %s%s\n' "$C_W" "$*" "$C_0"; }

on_error() {
  local rc=$? line=$1 cmd=$2
  printf '%s[BŁĄD]%s Nieoczekiwany błąd (kod %s) w linii %s: %s\n' "$C_R" "$C_0" "$rc" "$line" "$cmd"
  [[ -n "${RUN_LOG:-}" ]] && printf 'Pełny log: %s\n' "$RUN_LOG"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

as_root() {
  if [[ $EUID -eq 0 ]]; then "$@"
  elif command -v sudo >/dev/null 2>&1; then sudo "$@"
  else die "Potrzebne uprawnienia root do: $* (brak sudo)."
  fi
}

human_gib() { awk -v b="$1" 'BEGIN { printf "%.1f GiB", b / 1073741824 }'; }

# =============================================================================
#  Skrypty pomocnicze (Python, tylko biblioteka standardowa)
# =============================================================================
write_helpers() {
  mkdir -p "$BIN_DIR"

  cat > "$BIN_DIR/gguf_meta.py" <<'PYEOF'
#!/usr/bin/env python3
"""Czyta z nagłówka pliku GGUF architekturę i natywną długość kontekstu."""
import struct
import sys

SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main(path):
    with open(path, "rb") as f:
        def rd(fmt):
            size = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, f.read(size))[0]

        def rstr():
            return f.read(rd("Q")).decode("utf-8", "replace")

        def value(t):
            if t == 8:
                return rstr()
            if t == 9:
                at, n = rd("I"), rd("Q")
                return [value(at) for _ in range(n)]
            return rd(SCALARS[t])

        if f.read(4) != b"GGUF":
            sys.exit("to nie jest plik GGUF: " + path)
        rd("I")
        rd("Q")
        n_kv = rd("Q")
        meta = {}
        for _ in range(n_kv):
            key = rstr()
            meta[key] = value(rd("I"))
            arch = meta.get("general.architecture")
            if arch and arch + ".context_length" in meta:
                break
    arch = meta.get("general.architecture", "unknown")
    print("ARCH=" + str(arch))
    print("CTX=" + str(meta.get(arch + ".context_length", 0)))


if __name__ == "__main__":
    main(sys.argv[1])
PYEOF

  cat > "$BIN_DIR/apitest.py" <<'PYEOF'
#!/usr/bin/env python3
"""Testy API llama-server (OpenAI-compatible) dla vast-qwen35-abliterated.

Wypisuje linie:
  RESULT <test> PASS|WARN|FAIL <opis>
  METRIC <nazwa> <wartość>
Kod wyjścia 0 = brak FAIL, 1 = co najmniej jeden FAIL.
"""
import argparse
import collections
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zlib

FAILED = []

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time on the server (ISO 8601).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the project and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the project root"}},
                "required": ["path"],
            },
        },
    },
]
SYSTEM = {"role": "system", "content": "You are a coding assistant. Use the provided tools whenever they are relevant."}
TEXT_TOOL_CALL = re.compile(r"<tool_call>|<function=|\"name\"\s*:\s*\"(get_time|read_file)\"")


def result(name, status, msg):
    msg = " ".join(str(msg).split())[:400]
    print(f"RESULT {name} {status} {msg}", flush=True)
    if status == "FAIL":
        FAILED.append(name)


def metric(key, value):
    print(f"METRIC {key} {value}", flush=True)


class Api:
    def __init__(self, base, alias):
        self.base = base.rstrip("/")
        self.alias = alias

    def request(self, method, path, body=None, timeout=600):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None

    def chat(self, messages, timeout=600, **kw):
        body = {"model": self.alias, "messages": messages}
        body.update(kw)
        return self.request("POST", "/v1/chat/completions", body, timeout)

    def chat_stream(self, messages, timeout=600, **kw):
        body = {"model": self.alias, "messages": messages, "stream": True}
        body.update(kw)
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json"})
        calls, content, finish = {}, "", None
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta") or {}
                    content += delta.get("content") or ""
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    finish = ch.get("finish_reason") or finish
        return [calls[i] for i in sorted(calls)], content, finish


def looks_looping(text):
    """Heurystyka pętli: tekst bardzo dobrze się kompresuje albo jedna linia powtarza się wiele razy."""
    data = text.encode("utf-8")
    if len(data) >= 1500 and len(zlib.compress(data)) / len(data) < 0.08:
        return True
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 15]
    return bool(lines) and collections.Counter(lines).most_common(1)[0][1] >= 8


def first_choice(resp):
    ch = resp["choices"][0]
    return ch["message"], ch.get("finish_reason")


def t_models(api):
    try:
        data = api.request("GET", "/v1/models", timeout=30)
        ids = [m.get("id") for m in data.get("data", [])]
        if api.alias in ids:
            result("models", "PASS", f"/v1/models zawiera {api.alias}")
        else:
            result("models", "FAIL", f"/v1/models nie zawiera {api.alias}: {ids}")
    except Exception as e:  # noqa: BLE001
        result("models", "FAIL", f"/v1/models: {e}")


def t_generate(api):
    try:
        t0 = time.time()
        resp = api.chat([{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
                        max_tokens=4096)
        msg, finish = first_choice(resp)
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        timings = resp.get("timings") or {}
        if timings.get("predicted_per_second"):
            metric("gen_tps", round(timings["predicted_per_second"], 1))
        metric("gen_seconds", round(time.time() - t0, 1))
        metric("gen_tokens", (resp.get("usage") or {}).get("completion_tokens", 0))
        if finish == "length":
            result("generate", "FAIL", "finish_reason=length przy limicie 4096 tokenów - podejrzenie pętli generowania")
        elif looks_looping(content + "\n" + reasoning):
            result("generate", "FAIL", "odpowiedź wygląda na zapętloną (powtarzający się tekst)")
        elif not content.strip():
            result("generate", "FAIL", f"pusta odpowiedź (finish_reason={finish})")
        elif "391" not in content:
            result("generate", "WARN", f"odpowiedź bez oczekiwanego wyniku 391: {content[:120]!r}")
        else:
            result("generate", "PASS", f"finish_reason={finish}, odpowiedź={content.strip()[:40]!r}"
                   + (", z reasoning_content" if reasoning else ""))
    except Exception as e:  # noqa: BLE001
        result("generate", "FAIL", f"błąd zapytania: {e}")


def check_tool_message(name, msg, finish, want_tool):
    """Zwraca (status, opis, tool_call)."""
    calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    if not calls:
        if TEXT_TOOL_CALL.search(content):
            return "FAIL", ("model zwrócił wywołanie narzędzia jako zwykły tekst w content, "
                            f"a nie jako tool_calls: {content[:160]!r}"), None
        return "FAIL", f"brak tool_calls (finish_reason={finish}, content={content[:120]!r})", None
    tc = calls[0]
    fn = tc.get("function") or {}
    if tc.get("type") != "function" or fn.get("name") != want_tool:
        return "FAIL", f"oczekiwano funkcji {want_tool}, otrzymano {json.dumps(tc)[:200]}", None
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        return "FAIL", f"arguments to niepoprawny JSON: {fn.get('arguments')!r}", None
    if finish != "tool_calls":
        return "FAIL", f"finish_reason={finish}, oczekiwano tool_calls", None
    tc["_args"] = args
    return "PASS", f"finish_reason=tool_calls, {want_tool}({fn.get('arguments')})", tc


def ask_for_tool(api, name, prompt, want_tool, attempts=2):
    messages = [SYSTEM, {"role": "user", "content": prompt}]
    last = ("FAIL", "brak odpowiedzi", None)
    for attempt in range(1, attempts + 1):
        try:
            resp = api.chat(messages, tools=TOOLS, tool_choice="auto", max_tokens=4096)
            msg, finish = first_choice(resp)
            last = check_tool_message(name, msg, finish, want_tool)
            if last[0] == "PASS":
                suffix = "" if attempt == 1 else f" (udane w próbie {attempt}/{attempts})"
                return (last[0], last[1] + suffix, last[2]), messages, msg
        except Exception as e:  # noqa: BLE001
            last = ("FAIL", f"błąd zapytania: {e}", None)
    return last, messages, None


def t_tools(api):
    # 1) wywołanie bez argumentów
    (status, desc, tc), messages, msg = ask_for_tool(
        api, "tool_call", "What time is it right now? Use the get_time tool.", "get_time")
    result("tool_call", status, desc)

    # 2) kontynuacja po wyniku narzędzia
    if tc is None:
        result("tool_continue", "FAIL", "pominięty - brak poprawnego wywołania get_time")
    else:
        try:
            assistant = {"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": [{"id": tc.get("id") or "call_0", "type": "function",
                                         "function": {"name": "get_time", "arguments": tc["function"]["arguments"]}}]}
            tool_msg = {"role": "tool", "tool_call_id": tc.get("id") or "call_0",
                        "content": json.dumps({"time": "2026-10-25T14:32:07Z"})}
            resp = api.chat(messages + [assistant, tool_msg], tools=TOOLS, max_tokens=4096)
            msg2, finish2 = first_choice(resp)
            content = msg2.get("content") or ""
            if finish2 == "length" or looks_looping(content):
                result("tool_continue", "FAIL", f"podejrzenie pętli (finish_reason={finish2})")
            elif msg2.get("tool_calls"):
                result("tool_continue", "WARN", "model ponownie wywołał narzędzie zamiast odpowiedzieć")
            elif not content.strip():
                result("tool_continue", "FAIL", f"pusta odpowiedź po wyniku narzędzia (finish_reason={finish2})")
            elif TEXT_TOOL_CALL.search(content):
                result("tool_continue", "FAIL", f"wywołanie narzędzia jako tekst: {content[:160]!r}")
            elif ":32" not in content:
                result("tool_continue", "WARN", f"odpowiedź nie zawiera godziny z narzędzia: {content[:120]!r}")
            else:
                result("tool_continue", "PASS", f"finish_reason={finish2}, odpowiedź używa wyniku narzędzia")
        except Exception as e:  # noqa: BLE001
            result("tool_continue", "FAIL", f"błąd zapytania: {e}")

    # 3) wywołanie z argumentem (tak działa odczyt/edycja plików w OpenCode)
    (status, desc, tc), _, _ = ask_for_tool(
        api, "tool_args", "Show me the contents of the file src/main.py.", "read_file")
    if status == "PASS" and str(tc["_args"].get("path", "")).lstrip("./") != "src/main.py":
        status, desc = "FAIL", f"niepoprawny argument path: {tc['_args']}"
    result("tool_args", status, desc)

    # 4) to samo w trybie streaming (OpenCode używa streamingu)
    try:
        calls, content, finish = api.chat_stream(
            [SYSTEM, {"role": "user", "content": "What time is it right now? Use the get_time tool."}],
            tools=TOOLS, tool_choice="auto", max_tokens=4096)
        if calls and calls[0]["name"] == "get_time" and finish == "tool_calls":
            json.loads(calls[0]["arguments"] or "{}")
            result("tool_stream", "PASS", "streaming: finish_reason=tool_calls, get_time")
        elif not calls and TEXT_TOOL_CALL.search(content):
            result("tool_stream", "FAIL", f"streaming: wywołanie narzędzia jako tekst: {content[:160]!r}")
        else:
            result("tool_stream", "FAIL", f"streaming: calls={calls}, finish_reason={finish}")
    except Exception as e:  # noqa: BLE001
        result("tool_stream", "FAIL", f"streaming: {e}")


def t_long(api, ctx, fraction, timeout):
    needle = "AZURE-FALCON-7319"

    def record(i):
        return f"Record {i:06d}: sensor S{i % 97:02d} logged value {(i * 7919) % 10007} at step {i}.\n"

    try:
        sample = "".join(record(i) for i in range(400))
        per_line = len(api.request("POST", "/tokenize", {"content": sample})["tokens"]) / 400
        budget = int(ctx * fraction) - 600
        n = max(10, int(budget / per_line))
        question = ("\n\nQuestion: What is the secret project codename mentioned in the records above? "
                    "Reply with the codename only.")
        for _ in range(3):
            lines = [record(i) for i in range(n)]
            lines.insert(n // 10, f"IMPORTANT NOTE: The secret project codename is {needle}.\n")
            doc = "".join(lines) + question
            ntok = len(api.request("POST", "/tokenize", {"content": doc}, timeout=300)["tokens"])
            if ntok <= ctx - 600:
                break
            n = int(n * (ctx - 1200) / ntok)
        metric("long_prompt_tokens", ntok)
        t0 = time.time()
        resp = api.chat([{"role": "user", "content": doc}], timeout=timeout, max_tokens=256, temperature=0,
                        chat_template_kwargs={"enable_thinking": False})
        secs = time.time() - t0
        msg, finish = first_choice(resp)
        content = msg.get("content") or ""
        timings = resp.get("timings") or {}
        metric("long_seconds", round(secs, 1))
        if timings.get("prompt_per_second"):
            metric("long_prompt_tps", round(timings["prompt_per_second"], 1))
        pct = round(100 * ntok / ctx)
        if needle.lower() in content.lower():
            result("long_context", "PASS", f"{ntok} tokenów ({pct}% kontekstu) w {secs:.0f} s, kod odnaleziony")
        else:
            result("long_context", "FAIL", f"{ntok} tokenów ({pct}% kontekstu): nie odnaleziono kodu, "
                   f"odpowiedź={content[:120]!r}, finish_reason={finish}")
    except Exception as e:  # noqa: BLE001
        result("long_context", "FAIL", f"błąd testu długiego kontekstu: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["quick", "long"])
    p.add_argument("--base", default="http://127.0.0.1:8081")
    p.add_argument("--alias", default="qwen-coder")
    p.add_argument("--ctx", type=int, default=0)
    p.add_argument("--fraction", type=float, default=0.9)
    p.add_argument("--timeout", type=int, default=2400)
    a = p.parse_args()
    api = Api(a.base, a.alias)
    if a.mode == "quick":
        t_models(api)
        t_generate(api)
        t_tools(api)
    else:
        t_long(api, a.ctx, a.fraction, a.timeout)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
PYEOF
  chmod +x "$BIN_DIR/gguf_meta.py" "$BIN_DIR/apitest.py"
}

# =============================================================================
#  1-3. System, GPU, CUDA
# =============================================================================
detect_system() {
  [[ "$(uname -s)" == "Linux" ]] || die "System $(uname -s) nie jest obsługiwany. Uruchom skrypt na instancji Vast.ai (Linux)."
  [[ -r /etc/os-release ]] || die "Brak /etc/os-release - nie można rozpoznać dystrybucji."
  # shellcheck source=/dev/null
  OS_ID=$(. /etc/os-release && echo "${ID:-unknown}")
  # shellcheck source=/dev/null
  OS_VER=$(. /etc/os-release && echo "${VERSION_ID:-}")
  # shellcheck source=/dev/null
  OS_PRETTY=$(. /etc/os-release && echo "${PRETTY_NAME:-$OS_ID}")
  CPU_ARCH=$(uname -m)
  command -v apt-get >/dev/null 2>&1 || die "Brak apt-get. Skrypt wspiera obrazy oparte na Debianie/Ubuntu (np. PyTorch z Vast.ai)."
  RAM_TOTAL_GIB=$(awk '/MemTotal/ { printf "%.0f", $2 / 1048576 }' /proc/meminfo)
  ok "System: $OS_PRETTY ($CPU_ARCH), RAM: ${RAM_TOTAL_GIB} GiB"
}

detect_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "Nie znaleziono nvidia-smi - instancja nie ma dostępu do GPU NVIDIA/CUDA. Uruchom instancję z GPU (szablon PyTorch/CUDA)."
  local q
  q=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>&1) \
    || die "nvidia-smi zwrócił błąd (sterownik/GPU niedostępne): $q"
  [[ -n "$q" ]] || die "nvidia-smi nie widzi żadnego GPU."
  GPU_COUNT=$(printf '%s\n' "$q" | wc -l)
  GPU_NAME=$(printf '%s\n' "$q" | head -1 | cut -d, -f1 | sed 's/^ *//; s/ *$//')
  DRIVER_VER=$(printf '%s\n' "$q" | head -1 | cut -d, -f3 | tr -d ' ')
  VRAM_TOTAL_MIB=$(printf '%s\n' "$q" | awk -F', *' '{ s += $2 } END { print s }')
  CUDA_DRIVER_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
  [[ -n "$CUDA_DRIVER_VER" ]] || die "Nie udało się odczytać wersji CUDA obsługiwanej przez sterownik (nvidia-smi)."
  local caps
  caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' .' | sort -u | paste -sd';' || true)
  if [[ "$caps" =~ ^[0-9]+(\;[0-9]+)*$ ]]; then CUDA_ARCHS="$caps"; else CUDA_ARCHS="native"; fi
  ok "GPU: ${GPU_COUNT}x $GPU_NAME, VRAM łącznie: ${VRAM_TOTAL_MIB} MiB, sterownik $DRIVER_VER (CUDA $CUDA_DRIVER_VER), arch: $CUDA_ARCHS"
  if (( GPU_COUNT > 1 )); then warn "Wykryto $GPU_COUNT GPU - llama.cpp rozłoży model między nie (split-mode layer)."; fi
}

version_major() { echo "${1%%.*}"; }
version_minor() { local v=${1#*.}; echo "${v%%.*}"; }

find_nvcc() {
  local c
  c=$(command -v nvcc 2>/dev/null || true)
  if [[ -n "$c" ]]; then echo "$c"; return 0; fi
  [[ -x /usr/local/cuda/bin/nvcc ]] && { echo /usr/local/cuda/bin/nvcc; return 0; }
  local cands=(/usr/local/cuda-*/bin/nvcc)
  local i
  for (( i=${#cands[@]}-1; i>=0; i-- )); do
    [[ -x "${cands[i]}" ]] && { echo "${cands[i]}"; return 0; }
  done
  return 1
}

nvcc_version() { "$1" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1; }

detect_cuda_toolkit() {
  NVCC=""; NVCC_VER=""; NEED_CUDA_INSTALL=0
  if NVCC=$(find_nvcc); then
    NVCC_VER=$(nvcc_version "$NVCC")
    if (( $(version_major "$NVCC_VER") > $(version_major "$CUDA_DRIVER_VER") )); then
      warn "nvcc $NVCC_VER jest nowszy niż CUDA $CUDA_DRIVER_VER obsługiwana przez sterownik - doinstaluję zgodny toolkit."
      NEED_CUDA_INSTALL=1
    else
      ok "CUDA toolkit: nvcc $NVCC_VER ($NVCC)"
    fi
  else
    warn "Brak nvcc (CUDA toolkit) - zostanie doinstalowany z oficjalnego repozytorium NVIDIA."
    NEED_CUDA_INSTALL=1
  fi
}

install_cuda_toolkit() {
  (( NEED_CUDA_INSTALL )) || return 0
  local drv_major drv_minor want_minor distro arch_path
  drv_major=$(version_major "$CUDA_DRIVER_VER"); drv_minor=$(version_minor "$CUDA_DRIVER_VER")
  if (( drv_major >= 13 )); then want_minor=8
  elif (( drv_major == 12 )); then want_minor=$(( drv_minor > 8 ? 8 : drv_minor ))
  else die "Sterownik obsługuje CUDA $CUDA_DRIVER_VER, a llama.cpp wymaga CUDA >= 12.0. Wybierz instancję z nowszym sterownikiem."
  fi
  [[ "$OS_ID" == "ubuntu" ]] || die "Automatyczna instalacja CUDA toolkit wspiera tylko Ubuntu (wykryto $OS_ID). Zainstaluj CUDA toolkit 12.x ręcznie."
  distro="ubuntu${OS_VER//./}"
  case "$CPU_ARCH" in x86_64) arch_path=x86_64 ;; aarch64) arch_path=sbsa ;; *) die "Nieobsługiwana architektura CPU: $CPU_ARCH" ;; esac
  if ! dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null | grep -q "install ok installed"; then
    local tmp; tmp=$(mktemp -d)
    info "Pobieram klucz repozytorium NVIDIA (developer.download.nvidia.com, $distro/$arch_path)"
    curl -fsSL --retry 3 -o "$tmp/cuda-keyring.deb" \
      "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch_path/cuda-keyring_1.1-1_all.deb" \
      || die "Nie udało się pobrać cuda-keyring dla $distro/$arch_path."
    as_root dpkg -i "$tmp/cuda-keyring.deb" >/dev/null
    rm -rf "$tmp"
  fi
  as_root apt-get update -qq
  local m pkg=""
  for (( m=want_minor; m>=0; m-- )); do
    if apt-cache show "cuda-toolkit-12-$m" >/dev/null 2>&1; then pkg="cuda-toolkit-12-$m"; break; fi
  done
  [[ -n "$pkg" ]] || die "W repozytorium NVIDIA nie ma pakietu cuda-toolkit-12-x zgodnego ze sterownikiem (CUDA $CUDA_DRIVER_VER)."
  info "Instaluję $pkg (to może potrwać kilka minut)"
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$pkg" >/dev/null
  NVCC="/usr/local/cuda-12.${pkg##*-}/bin/nvcc"
  [[ -x "$NVCC" ]] || NVCC=$(find_nvcc) || die "Po instalacji $pkg nadal nie ma nvcc."
  NVCC_VER=$(nvcc_version "$NVCC")
  ok "Zainstalowano CUDA toolkit: nvcc $NVCC_VER"
}

# =============================================================================
#  4-5. Dysk i zależności
# =============================================================================
check_disk() {
  mkdir -p "$WORKSPACE" 2>/dev/null || die "Nie można utworzyć $WORKSPACE."
  local need=0 entry name size part have
  for entry in "${MODEL_FILES[@]}"; do
    IFS='|' read -r name size _ <<<"$entry"
    have=0
    if [[ -f "$MODEL_DIR/$name" ]]; then have=$size
    elif [[ -f "$MODEL_DIR/$name.part" ]]; then have=$(stat -c %s "$MODEL_DIR/$name.part")
    fi
    part=$(( size - have )); (( part > 0 )) && need=$(( need + part ))
  done
  [[ -x "$LLAMA_BIN" ]] || need=$(( need + 5 * 1073741824 ))
  need=$(( need + 3 * 1073741824 ))  # zapas na logi i pliki tymczasowe
  local avail; avail=$(df -PB1 "$WORKSPACE" | awk 'NR == 2 { print $4 }')
  if (( avail < need )); then
    die "Za mało miejsca na $WORKSPACE: wolne $(human_gib "$avail"), potrzeba ok. $(human_gib "$need")."
  fi
  ok "Dysk $WORKSPACE: wolne $(human_gib "$avail"), potrzeba ok. $(human_gib "$need")"
  if (( NEED_CUDA_INSTALL )); then
    local root_avail; root_avail=$(df -PB1 / | awk 'NR == 2 { print $4 }')
    (( root_avail > 8 * 1073741824 )) || die "Za mało miejsca na / dla CUDA toolkit: wolne $(human_gib "$root_avail"), potrzeba ok. 8 GiB."
  fi
}

ensure_packages() {
  local pkgs=(git cmake build-essential curl ca-certificates python3 libssl-dev aria2 iproute2 procps util-linux)
  local missing=() p
  for p in "${pkgs[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p")
  done
  if (( ${#missing[@]} )); then
    info "Instaluję brakujące pakiety: ${missing[*]}"
    as_root apt-get update -qq
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${missing[@]}" >/dev/null
    ok "Pakiety zainstalowane"
  else
    ok "Wszystkie wymagane pakiety są już zainstalowane"
  fi
  local cv; cv=$(cmake --version | sed -n 's/^cmake version \([0-9]*\.[0-9]*\).*/\1/p')
  if (( $(version_major "$cv") < 3 || ( $(version_major "$cv") == 3 && $(version_minor "$cv") < 18 ) )); then
    info "CMake $cv jest za stary (wymagane >= 3.18) - instaluję cmake 3.31.6 z PyPI do ~/.local"
    python3 -m pip install --user --quiet "cmake==3.31.6" || die "Nie udało się zainstalować nowszego CMake."
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

# =============================================================================
#  6-7. llama.cpp
# =============================================================================
ensure_llama_cpp() {
  if [[ -e "$LLAMA_DIR" ]]; then
    git -C "$LLAMA_DIR" rev-parse --git-dir >/dev/null 2>&1 \
      || die "$LLAMA_DIR istnieje, ale nie jest repozytorium git. Nie nadpisuję go - przenieś katalog albo ustaw LLAMA_DIR=..."
    local remote; remote=$(git -C "$LLAMA_DIR" remote get-url origin 2>/dev/null || true)
    [[ "$remote" =~ (ggml-org|ggerganov)/llama\.cpp ]] \
      || die "$LLAMA_DIR to inne repozytorium (origin: ${remote:-brak}). Nie nadpisuję go - ustaw LLAMA_DIR=..."
    local head; head=$(git -C "$LLAMA_DIR" rev-parse HEAD)
    if [[ "$head" != "$LLAMA_CPP_COMMIT" ]]; then
      if [[ "$KEEP_LLAMA_CHECKOUT" == "1" ]] && grep -q '"qwen35moe"' "$LLAMA_DIR/src/llama-arch.cpp" 2>/dev/null; then
        warn "KEEP_LLAMA_CHECKOUT=1: zostaję przy llama.cpp ${head:0:10} (obsługuje qwen35moe), bez przypiętej wersji."
        LLAMA_COMMIT_USED=$head
        return 0
      fi
      [[ -z "$(git -C "$LLAMA_DIR" status --porcelain --untracked-files=no)" ]] \
        || die "$LLAMA_DIR ma niezatwierdzone zmiany - nie przełączam wersji. Zatwierdź/odłóż zmiany albo ustaw LLAMA_DIR=..."
      info "Przełączam llama.cpp z ${head:0:10} na przypiętą wersję $LLAMA_CPP_TAG (${LLAMA_CPP_COMMIT:0:10})"
      local depth=()
      [[ "$(git -C "$LLAMA_DIR" rev-parse --is-shallow-repository)" == "true" ]] && depth=(--depth 1)
      git -C "$LLAMA_DIR" fetch -q "${depth[@]}" origin "refs/tags/$LLAMA_CPP_TAG:refs/tags/$LLAMA_CPP_TAG"
      git -C "$LLAMA_DIR" -c advice.detachedHead=false checkout -q --detach "$LLAMA_CPP_COMMIT"
    else
      ok "llama.cpp $LLAMA_CPP_TAG już pobrany ($LLAMA_DIR)"
    fi
  else
    info "Pobieram llama.cpp $LLAMA_CPP_TAG do $LLAMA_DIR"
    git clone -q --depth 1 --branch "$LLAMA_CPP_TAG" -c advice.detachedHead=false "$LLAMA_CPP_REPO" "$LLAMA_DIR"
  fi
  local now; now=$(git -C "$LLAMA_DIR" rev-parse HEAD)
  [[ "$now" == "$LLAMA_CPP_COMMIT" ]] \
    || die "Commit llama.cpp ($now) różni się od przypiętego ($LLAMA_CPP_COMMIT) - przerywam ze względów bezpieczeństwa."
  grep -q '"qwen35moe"' "$LLAMA_DIR/src/llama-arch.cpp" \
    || die "Ta wersja llama.cpp nie obsługuje architektury qwen35moe."
  LLAMA_COMMIT_USED=$now
  ok "llama.cpp @ ${now:0:10} (zweryfikowany commit, obsługuje qwen35moe)"
}

cpu_jobs() {
  local jobs; jobs=$(nproc)
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    local q p; read -r q p < /sys/fs/cgroup/cpu.max
    [[ "$q" != "max" ]] && (( (q + p - 1) / p < jobs )) && jobs=$(( (q + p - 1) / p ))
  fi
  local mem_gb; mem_gb=$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)
  (( jobs > mem_gb / 3 )) && jobs=$(( mem_gb / 3 ))   # kompilacja kerneli CUDA zużywa dużo RAM
  (( jobs > 32 )) && jobs=32
  (( jobs < 1 )) && jobs=1
  echo "$jobs"
}

build_llama() {
  local cmake_flags=(
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS"
    "-DCMAKE_CUDA_COMPILER=$NVCC" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
  )
  local stamp stamp_file="$BUILD_DIR/.vastqwen-stamp"
  stamp="$LLAMA_COMMIT_USED|nvcc $NVCC_VER|$(grep -m1 'model name' /proc/cpuinfo || true)|${cmake_flags[*]}"
  if [[ -x "$LLAMA_BIN" && -f "$stamp_file" && "$(cat "$stamp_file")" == "$stamp" ]]; then
    ok "llama-server już skompilowany z CUDA ($LLAMA_BIN)"
    return 0
  fi
  local jobs; jobs=$(cpu_jobs)
  local blog="$LOG_DIR/build.log"
  info "Kompiluję llama-server z CUDA (arch $CUDA_ARCHS, $jobs wątków) - zwykle 5-15 min. Log: $blog"
  PATH="$(dirname "$NVCC"):$PATH"; export PATH
  if ! { cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" "${cmake_flags[@]}" \
         && cmake --build "$BUILD_DIR" --config Release --target llama-server -j "$jobs"; } >"$blog" 2>&1; then
    tail -n 40 "$blog"
    die "Kompilacja llama.cpp nie powiodła się. Pełny log: $blog"
  fi
  [[ -x "$LLAMA_BIN" ]] || die "Po kompilacji brak pliku $LLAMA_BIN."
  printf '%s' "$stamp" > "$stamp_file"
  ok "Skompilowano: $LLAMA_BIN"
}

llama_version() { "$LLAMA_BIN" --version 2>&1 | grep -m1 '^version:' || echo "nieznana"; }

# =============================================================================
#  8-9. Model
# =============================================================================
download_model() {
  mkdir -p "$MODEL_DIR"
  local entry name size sha dest url
  for entry in "${MODEL_FILES[@]}"; do
    IFS='|' read -r name size sha <<<"$entry"
    dest="$MODEL_DIR/$name"
    local verified="$dest.sha256-ok"
    if [[ -f "$dest" ]]; then
      local cur; cur=$(stat -c %s "$dest")
      if [[ "$cur" == "$size" && -f "$verified" && "$(cat "$verified")" == "$sha $(stat -c %Y "$dest")" ]]; then
        ok "Model $name już pobrany i zweryfikowany ($(human_gib "$size"))"
        continue
      fi
      if [[ "$cur" == "$size" ]]; then
        info "Weryfikuję SHA256 istniejącego pliku $name (ok. 1-2 min)"
        if [[ "$(sha256sum "$dest" | cut -d' ' -f1)" == "$sha" ]]; then
          echo "$sha $(stat -c %Y "$dest")" > "$verified"
          ok "Model $name zweryfikowany"
          continue
        fi
      fi
      warn "Plik $dest ma zły rozmiar/sumę - przenoszę go do $dest.bad i pobieram ponownie."
      mv -f "$dest" "$dest.bad"
    fi
    url="https://huggingface.co/$HF_REPO/resolve/$HF_REVISION/$name?download=true"
    info "Pobieram $name ($(human_gib "$size")) z huggingface.co/$HF_REPO @ ${HF_REVISION:0:10}"
    local auth=()
    if command -v aria2c >/dev/null 2>&1; then
      [[ -n "${HF_TOKEN:-}" ]] && auth=(--header="Authorization: Bearer $HF_TOKEN")
      aria2c -x 16 -s 16 -k 64M -c --file-allocation=none --auto-file-renaming=false --allow-overwrite=true \
        --console-log-level=warn --summary-interval=30 --download-result=hide "${auth[@]}" \
        -d "$MODEL_DIR" -o "$name.part" "$url" \
        || die "Pobieranie $name nie powiodło się (aria2c). Uruchom skrypt ponownie - pobieranie zostanie wznowione."
    else
      [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")
      curl -fL --retry 10 --retry-delay 5 -C - "${auth[@]}" -o "$dest.part" "$url" \
        || die "Pobieranie $name nie powiodło się (curl). Uruchom skrypt ponownie - pobieranie zostanie wznowione."
    fi
    [[ "$(stat -c %s "$dest.part")" == "$size" ]] || die "Pobrany plik $name ma zły rozmiar. Uruchom skrypt ponownie."
    info "Weryfikuję SHA256 $name"
    [[ "$(sha256sum "$dest.part" | cut -d' ' -f1)" == "$sha" ]] \
      || { mv -f "$dest.part" "$dest.bad"; die "Suma SHA256 $name się nie zgadza (plik przeniesiony do $dest.bad). Uruchom ponownie."; }
    mv -f "$dest.part" "$dest"
    rm -f "$dest.part.aria2"
    echo "$sha $(stat -c %Y "$dest")" > "$verified"
    ok "Pobrano i zweryfikowano $name"
  done
}

read_model_meta() {
  local out; out=$(python3 "$BIN_DIR/gguf_meta.py" "$MODEL_PATH") || die "Nie udało się odczytać metadanych GGUF."
  MODEL_ARCH=$(sed -n 's/^ARCH=//p' <<<"$out")
  NATIVE_CTX=$(sed -n 's/^CTX=//p' <<<"$out")
  [[ "$MODEL_ARCH" == "qwen35moe" ]] || warn "Nieoczekiwana architektura modelu: $MODEL_ARCH"
  ok "Model: architektura $MODEL_ARCH, natywny kontekst $NATIVE_CTX tokenów"
}

# =============================================================================
#  10-11. Serwer
# =============================================================================
port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$QWEN_PORT") 2>/dev/null; }

our_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid; pid=$(cat "$PID_FILE")
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q llama-server \
    && { echo "$pid"; return 0; }
  return 1
}

describe_port_owner() {
  if command -v ss >/dev/null 2>&1; then ss -ltnp "sport = :$QWEN_PORT" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then lsof -nP -iTCP:"$QWEN_PORT" -sTCP:LISTEN 2>/dev/null || true
  else echo "(brak ss/lsof - nie można ustalić procesu)"
  fi
}

check_port_free_or_ours() {
  port_in_use || return 0
  our_pid >/dev/null && return 0
  warn "Port $QWEN_PORT jest zajęty przez inny proces:"
  describe_port_owner
  die "Nie zatrzymuję cudzych procesów. Zwolnij port $QWEN_PORT (np. zatrzymaj powyższy proces) i uruchom skrypt ponownie."
}

stop_server() {
  local pid
  pid=$(our_pid) || { rm -f "$PID_FILE"; return 0; }
  info "Zatrzymuję llama-server uruchomiony przez ten skrypt (PID $pid)"
  kill "$pid" 2>/dev/null || true
  local i
  for (( i=0; i<30; i++ )); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  local j
  for (( j=0; j<10; j++ )); do port_in_use || break; sleep 1; done
}

server_args() {  # ctx kv
  SERVER_ARGS=(
    -m "$MODEL_PATH" --alias "$MODEL_ALIAS" --host "$QWEN_HOST" --port "$QWEN_PORT"
    -c "$1" -ngl all -fa on -ctk "$2" -ctv "$2" -np 1 -b "$BATCH" -ub "$UBATCH"
    --jinja --fit off --no-webui "${SERVER_SAMPLING[@]}"
  )
}

start_server() {  # ctx kv
  server_args "$1" "$2"
  check_port_free_or_ours
  [[ -f "$SERVER_LOG" ]] && mv -f "$SERVER_LOG" "$SERVER_LOG.prev"
  printf '%q ' "$LLAMA_BIN" "${SERVER_ARGS[@]}" > "$CMD_FILE"
  info "Start: llama-server -c $1 -ctk/-ctv $2 -ngl all -fa on --jinja -np 1 (log: $SERVER_LOG)"
  setsid nohup "$LLAMA_BIN" "${SERVER_ARGS[@]}" >>"$SERVER_LOG" 2>&1 </dev/null &
  echo $! > "$PID_FILE"
}

log_has_oom() {
  grep -Eiq 'out of memory|cudaMalloc failed|failed to allocate|unable to allocate|alloc_buffer: allocating .* failed|failed to create context' \
    "$SERVER_LOG" 2>/dev/null
}

# 0 = gotowy, 2 = brak pamięci, 1 = inny błąd
wait_ready() {
  local pid; pid=$(cat "$PID_FILE")
  local t0=$SECONDS code
  while (( SECONDS - t0 < LOAD_TIMEOUT )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log_has_oom && { FAIL_REASON="brak pamięci przy ładowaniu (CUDA OOM)"; return 2; }
      FAIL_REASON="serwer zakończył działanie przy starcie"; return 1
    fi
    log_has_oom && { FAIL_REASON="CUDA OOM w logu przy ładowaniu"; return 2; }
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$QWEN_HOST:$QWEN_PORT/health" || true)
    [[ "$code" == "200" ]] && { ok "Serwer gotowy po $(( SECONDS - t0 )) s"; return 0; }
    sleep 3
  done
  FAIL_REASON="przekroczono czas ładowania ($LOAD_TIMEOUT s)"; return 1
}

check_offload() {
  local line; line=$(grep -o 'offloaded [0-9]*/[0-9]* layers to GPU' "$SERVER_LOG" | tail -1 || true)
  if [[ -z "$line" ]]; then warn "Nie znaleziono w logu informacji o liczbie warstw na GPU."; return 0; fi
  local n m; n=$(sed 's/offloaded \([0-9]*\)\/.*/\1/' <<<"$line"); m=$(sed 's/.*\/\([0-9]*\) layers.*/\1/' <<<"$line")
  OFFLOAD_INFO="$n/$m"
  if [[ "$n" != "$m" ]]; then FAIL_REASON="tylko $n/$m warstw na GPU"; return 1; fi
  ok "Wszystkie warstwy na GPU ($n/$m)"
}

vram_stats() {  # -> "total used free" (MiB, suma GPU)
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits \
    | awk -F', *' '{ t += $1; u += $2; f += $3 } END { print t, u, f }'
}

declare -A TEST_RESULT=() METRICS=()
run_apitest() {  # mode [args...]
  local out rc=0 line
  out=$(python3 "$BIN_DIR/apitest.py" "$@" --base "http://$QWEN_HOST:$QWEN_PORT" --alias "$MODEL_ALIAS") || rc=$?
  printf '%s\n' "$out" >> "$LOG_DIR/apitest.log"
  while IFS= read -r line; do
    case "$line" in
      "RESULT "*)
        local name status msg
        read -r _ name status msg <<<"$line"
        TEST_RESULT[$name]=$status
        case "$status" in
          PASS) ok "test $name: $msg" ;;
          WARN) warn "test $name: $msg" ;;
          *) printf '%s[FAIL]%s test %s: %s\n' "$C_R" "$C_0" "$name" "$msg" ;;
        esac ;;
      "METRIC "*)
        local key val
        read -r _ key val <<<"$line"
        METRICS[$key]=$val ;;
      *) [[ -n "$line" ]] && printf '       %s\n' "$line" ;;
    esac
  done <<<"$out"
  return "$rc"
}

# Pełny test jednej konfiguracji. 0 = OK, 2 = problem z pamięcią, 3 = za wolno, 1 = błąd funkcjonalny.
test_config() {  # ctx kv
  local ctx=$1 kv=$2 rc
  FAIL_REASON=""; OFFLOAD_INFO="?"
  step "Test konfiguracji: kontekst $ctx, KV cache $kv"
  stop_server
  start_server "$ctx" "$kv"
  wait_ready && rc=0 || rc=$?
  if (( rc != 0 )); then
    tail -n 15 "$SERVER_LOG" | sed 's/^/       /'
    stop_server; return "$rc"
  fi
  check_offload || { stop_server; return 2; }
  read -r VRAM_T VRAM_U VRAM_F < <(vram_stats)
  info "VRAM po załadowaniu: zajęte $VRAM_U MiB / $VRAM_T MiB, wolne $VRAM_F MiB"
  if (( VRAM_F < MIN_FREE_VRAM_MIB )); then
    FAIL_REASON="za mały zapas VRAM ($VRAM_F MiB < $MIN_FREE_VRAM_MIB MiB)"; stop_server; return 2
  fi
  if ! run_apitest quick; then
    FAIL_REASON="testy API/narzędzi nie przeszły"
    log_has_oom && { FAIL_REASON="CUDA OOM podczas testów"; stop_server; return 2; }
    stop_server; return 1
  fi
  if [[ "$LONG_TEST" == "1" ]]; then
    info "Test długiego kontekstu (~$(awk -v f="$LONG_TEST_FRACTION" 'BEGIN { print f * 100 }')% z $ctx tokenów) - może potrwać kilka minut"
    if ! run_apitest long --ctx "$ctx" --fraction "$LONG_TEST_FRACTION" --timeout "$LONG_TEST_TIMEOUT"; then
      if log_has_oom || ! our_pid >/dev/null; then FAIL_REASON="OOM/awaria serwera przy długim kontekście"; stop_server; return 2; fi
      FAIL_REASON="test długiego kontekstu nie przeszedł"; stop_server; return 1
    fi
  else
    TEST_RESULT[long_context]="SKIP"
  fi
  log_has_oom && { FAIL_REASON="CUDA OOM w logu"; stop_server; return 2; }
  our_pid >/dev/null || { FAIL_REASON="serwer przestał działać"; return 1; }
  read -r VRAM_T VRAM_U VRAM_F < <(vram_stats)
  info "VRAM po testach: zajęte $VRAM_U MiB, wolne $VRAM_F MiB"
  if (( VRAM_F < MIN_FREE_VRAM_MIB )); then
    FAIL_REASON="za mały zapas VRAM po testach ($VRAM_F MiB)"; stop_server; return 2
  fi
  local tps=${METRICS[gen_tps]:-}
  if [[ -n "$tps" ]] && awk -v t="$tps" -v m="$MIN_GEN_TPS" 'BEGIN { exit !(t < m) }'; then
    FAIL_REASON="za wolne generowanie ($tps tok/s < $MIN_GEN_TPS)"; stop_server; return 3
  fi
  return 0
}

save_selection() {  # ctx kv
  {
    echo "# Konfiguracja wybrana przez start.sh $SCRIPT_VERSION - $(date -Is)"
    printf 'SEL_CTX=%q\n' "$1"
    printf 'SEL_KV=%q\n' "$2"
    printf 'SEL_BATCH=%q\n' "$BATCH"
    printf 'SEL_UBATCH=%q\n' "$UBATCH"
    printf 'SEL_LLAMA_COMMIT=%q\n' "$LLAMA_COMMIT_USED"
    printf 'SEL_MODEL_SHA=%q\n' "$MODEL_SHA"
    printf 'SEL_GPU=%q\n' "$GPU_COUNT x $GPU_NAME $VRAM_TOTAL_MIB"
    printf 'SEL_LONG_TEST=%q\n' "${TEST_RESULT[long_context]:-SKIP}"
    printf 'SEL_GEN_TPS=%q\n' "${METRICS[gen_tps]:-?}"
    printf 'SEL_DATE=%q\n' "$(date -Is)"
  } > "$SELECTED_FILE"
  ok "Zapisano sprawdzoną konfigurację: $SELECTED_FILE"
}

load_selection() {
  [[ -f "$SELECTED_FILE" ]] || return 1
  local SEL_CTX="" SEL_KV="" SEL_BATCH="" SEL_UBATCH="" SEL_LLAMA_COMMIT="" SEL_MODEL_SHA="" SEL_GPU=""
  # shellcheck disable=SC2034  # SEL_GEN_TPS jest w pliku tylko informacyjnie
  local SEL_LONG_TEST="" SEL_GEN_TPS="" SEL_DATE=""
  # shellcheck disable=SC1090
  source "$SELECTED_FILE"
  if [[ "$SEL_LLAMA_COMMIT" != "$LLAMA_COMMIT_USED" || "$SEL_MODEL_SHA" != "$MODEL_SHA" \
        || "$SEL_GPU" != "$GPU_COUNT x $GPU_NAME $VRAM_TOTAL_MIB" || "$SEL_BATCH" != "$BATCH" || "$SEL_UBATCH" != "$UBATCH" ]]; then
    info "Zapisana konfiguracja dotyczy innego GPU/wersji - zostanie przetestowana od nowa."
    return 1
  fi
  CHOSEN_CTX=$SEL_CTX; CHOSEN_KV=$SEL_KV; CACHED_LONG_TEST=$SEL_LONG_TEST; CACHED_DATE=$SEL_DATE
  return 0
}

select_and_run() {
  CHOSEN_CTX=""; CHOSEN_KV=""; CACHED_LONG_TEST=""; FROM_CACHE=0; ATTEMPTS_SUMMARY=""
  local -a attempts=()

  # Konfiguracja z poprzedniego uruchomienia
  if [[ "$FORCE_RETEST" != "1" ]] && load_selection; then
    server_args "$CHOSEN_CTX" "$CHOSEN_KV"
    local expected; expected=$(printf '%q ' "$LLAMA_BIN" "${SERVER_ARGS[@]}")
    step "Zapisana konfiguracja z $CACHED_DATE: kontekst $CHOSEN_CTX, KV $CHOSEN_KV"
    if our_pid >/dev/null && [[ -f "$CMD_FILE" && "$(cat "$CMD_FILE")" == "$expected" ]]; then
      info "Serwer z tą konfiguracją już działa (PID $(our_pid)) - sprawdzam go bez restartu."
    else
      stop_server
      start_server "$CHOSEN_CTX" "$CHOSEN_KV"
      if ! wait_ready; then
        warn "Start zapisanej konfiguracji nie powiódł się ($FAIL_REASON) - testuję od nowa."
        stop_server; CHOSEN_CTX=""
      fi
    fi
    if [[ -n "$CHOSEN_CTX" ]]; then
      check_offload || true
      if run_apitest quick; then
        FROM_CACHE=1; TEST_RESULT[long_context]="${CACHED_LONG_TEST:-SKIP} (zapisany)"; return 0
      fi
      warn "Szybkie testy zapisanej konfiguracji nie przeszły - testuję wszystkie konfiguracje od nowa."
      stop_server; CHOSEN_CTX=""
    fi
  fi

  # Pełne testy: od największego kontekstu
  local ctx kv rc functional_fail=0
  for ctx in $CTX_CANDIDATES; do
    if (( ctx > NATIVE_CTX )); then
      warn "Pomijam kontekst $ctx: model natywnie obsługuje $NATIVE_CTX tokenów (rozszerzanie RoPE/YaRN nie jest stosowane)."
      continue
    fi
    for kv in $KV_TYPES; do
      TEST_RESULT=(); METRICS=()
      test_config "$ctx" "$kv" && rc=0 || rc=$?
      if (( rc == 0 )); then
        attempts+=("$ctx $kv: OK")
        CHOSEN_CTX=$ctx; CHOSEN_KV=$kv
        ok "Wybrano konfigurację: kontekst $ctx, KV cache $kv"
        save_selection "$ctx" "$kv"
        ATTEMPTS_SUMMARY=$(printf '%s\n' "${attempts[@]}")
        return 0
      fi
      attempts+=("$ctx $kv: ODRZUCONA - $FAIL_REASON")
      warn "Konfiguracja $ctx/$kv odrzucona: $FAIL_REASON"
      (( rc == 1 )) && functional_fail=1
      (( rc == 2 )) || break   # kolejny typ KV ma sens tylko przy braku pamięci
    done
  done
  stop_server
  printf '\nPróby:\n'; printf '  %s\n' "${attempts[@]}"
  if (( functional_fail )); then
    die "Żadna konfiguracja nie przeszła testów funkcjonalnych (API/tool calling). Szczegóły: $LOG_DIR/apitest.log i $SERVER_LOG"
  fi
  die "Żadna konfiguracja nie zmieściła się w VRAM z wymaganym zapasem. Szczegóły: $SERVER_LOG"
}

# =============================================================================
#  12-13. OpenCode i raport
# =============================================================================
write_opencode_config() {  # ctx
  cat > "$OPENCODE_FILE" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "vast": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Vast.ai (llama.cpp)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "$MODEL_ALIAS": {
          "name": "$MODEL_NAME $MODEL_QUANT",
          "tool_call": true,
          "reasoning": true,
          "limit": {
            "context": $1,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "vast/$MODEL_ALIAS"
}
EOF
  python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$OPENCODE_FILE" || die "Wygenerowany opencode.json jest niepoprawny."
}

tool_calling_status() {
  local t
  for t in tool_call tool_continue tool_args tool_stream; do
    [[ "${TEST_RESULT[$t]:-FAIL}" == "FAIL" ]] && { echo "FAIL"; return; }
  done
  for t in tool_call tool_continue tool_args tool_stream; do
    [[ "${TEST_RESULT[$t]}" == "WARN" ]] && { echo "PASS (z ostrzeżeniami)"; return; }
  done
  echo "PASS"
}

print_report() {
  local vt vu vf api_status="DOWN" status="STOPPED"
  read -r vt vu vf < <(vram_stats)
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$QWEN_HOST:$QWEN_PORT/health" || true)" == "200" ]] && api_status="OK"
  our_pid >/dev/null && status="RUNNING"
  local ssh_host="${PUBLIC_IPADDR:-<IP_INSTANCJI>}" ssh_port="${VAST_TCP_PORT_22:-<PORT_SSH>}"
  {
    printf '\n%s============================================================%s\n' "$C_W" "$C_0"
    printf ' MODEL:          %s\n' "$MODEL_NAME ($HF_REPO)"
    printf ' QUANTIZATION:   %s\n' "$MODEL_QUANT"
    printf ' GPU:            %s\n' "${GPU_COUNT}x $GPU_NAME"
    printf ' CONTEXT:        %s (natywny max modelu: %s)\n' "$CHOSEN_CTX" "$NATIVE_CTX"
    printf ' KV CACHE:       %s (flash attention: on)\n' "${CHOSEN_KV^^}"
    printf ' GPU LAYERS:     %s\n' "${OFFLOAD_INFO:-all}"
    printf ' VRAM TOTAL:     %s MiB\n' "$vt"
    printf ' VRAM USED:      %s MiB\n' "$vu"
    printf ' VRAM FREE:      %s MiB\n' "$vf"
    printf ' LLAMA.CPP:      %s (%s, %s)\n' "$(llama_version)" "$LLAMA_CPP_TAG" "${LLAMA_COMMIT_USED:0:10}"
    printf ' GENERATION:     %s tok/s\n' "${METRICS[gen_tps]:-?}"
    printf ' API:            http://%s:%s/v1  (%s)\n' "$QWEN_HOST" "$QWEN_PORT" "$api_status"
    printf ' TOOL CALLING:   %s\n' "$(tool_calling_status)"
    printf ' LONG CONTEXT:   %s\n' "${TEST_RESULT[long_context]:-SKIP}"
    printf ' CONFIG:         %s\n' "$( (( FROM_CACHE )) && echo "z pliku $SELECTED_FILE" || echo "przetestowana teraz, zapisana w $SELECTED_FILE")"
    printf ' STATUS:         %s\n' "$status"
    printf '%s============================================================%s\n' "$C_W" "$C_0"
    [[ -n "$ATTEMPTS_SUMMARY" ]] && printf '\nTestowane konfiguracje:\n%s\n' "  ${ATTEMPTS_SUMMARY//$'\n'/$'\n'  }"
    printf '\nLogi:\n'
    printf '  serwer:      %s\n' "$SERVER_LOG"
    printf '  testy API:   %s\n' "$LOG_DIR/apitest.log"
    printf '  instalator:  %s\n' "$RUN_LOG"
    printf '  kompilacja:  %s\n' "$LOG_DIR/build.log"
    printf '\nTunel SSH z Windows (PowerShell) - Windows:8080 -> Vast.ai:%s:\n' "$QWEN_PORT"
    printf '  ssh -p %s root@%s -N -L 8080:127.0.0.1:%s\n' "$ssh_port" "$ssh_host" "$QWEN_PORT"
    printf '  (IP i port SSH znajdziesz w panelu Vast.ai: Instances -> przycisk ">_" / "Connect")\n'
    printf '\nTest API z Windows (po zestawieniu tunelu):\n'
    printf '  curl.exe http://127.0.0.1:8080/v1/models\n'
    printf '\nKonfiguracja OpenCode (kontekst %s): %s\n' "$CHOSEN_CTX" "$OPENCODE_FILE"
    printf '  Skopiuj na Windows do %%USERPROFILE%%\\.config\\opencode\\opencode.json albo do katalogu projektu:\n'
    printf '  scp -P %s root@%s:%s .\\opencode.json\n\n' "$ssh_port" "$ssh_host" "$OPENCODE_FILE"
    cat "$OPENCODE_FILE"
    printf '\nZatrzymanie serwera: bash %s stop    Status: bash %s status\n' "$0" "$0"
  } | tee "$REPORT_FILE"
}

# =============================================================================
#  Komendy
# =============================================================================
setup_logging() {
  mkdir -p "$LOG_DIR" "$BIN_DIR"
  RUN_LOG="$LOG_DIR/start-$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "$RUN_LOG") 2>&1
}

cmd_start() {
  setup_logging
  printf '%svast-qwen35-abliterated %s - instalator %s%s\n' "$C_W" "$SCRIPT_VERSION" "$MODEL_NAME $MODEL_QUANT" "$C_0"
  step "1/12 System";          detect_system
  step "2/12 GPU";             detect_gpu
  step "3/12 CUDA";            detect_cuda_toolkit
  step "4/12 Dysk";            check_disk
  step "5/12 Zależności";      ensure_packages; install_cuda_toolkit
  write_helpers
  check_port_free_or_ours
  step "6/12 llama.cpp";       ensure_llama_cpp
  step "7/12 Kompilacja";      build_llama
  step "8/12 Model";           download_model
  step "9/12 Metadane modelu"; read_model_meta
  step "10/12 Wybór kontekstu i start serwera"; select_and_run
  step "11/12 OpenCode";       write_opencode_config "$CHOSEN_CTX"; ok "Zapisano $OPENCODE_FILE"
  step "12/12 Raport";         print_report
}

cmd_stop() {
  if our_pid >/dev/null; then stop_server; ok "Serwer zatrzymany."
  else
    info "Nie działa żaden llama-server uruchomiony przez ten skrypt."
    port_in_use && { warn "Port $QWEN_PORT jest zajęty przez inny proces (nie zatrzymuję go):"; describe_port_owner; }
  fi
  return 0
}

cmd_status() {
  local pid
  if pid=$(our_pid); then ok "llama-server działa (PID $pid), polecenie: $(cat "$CMD_FILE" 2>/dev/null)"
  else warn "llama-server (z tego skryptu) nie działa."
  fi
  [[ -f "$SELECTED_FILE" ]] && { info "Wybrana konfiguracja ($SELECTED_FILE):"; grep '^SEL_' "$SELECTED_FILE" | sed 's/^/       /'; }
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$QWEN_HOST:$QWEN_PORT/health" || true)" == "200" ]]; then
    ok "API: http://$QWEN_HOST:$QWEN_PORT/v1 odpowiada"
  else warn "API: http://$QWEN_HOST:$QWEN_PORT nie odpowiada"
  fi
  command -v nvidia-smi >/dev/null 2>&1 && { read -r vt vu vf < <(vram_stats); info "VRAM: zajęte $vu / $vt MiB, wolne $vf MiB"; }
  return 0
}

cmd_test() {
  setup_logging
  write_helpers
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$QWEN_HOST:$QWEN_PORT/health" || true)" == "200" ]] \
    || die "Serwer na http://$QWEN_HOST:$QWEN_PORT nie odpowiada. Uruchom: bash $0"
  if run_apitest quick; then ok "Wszystkie szybkie testy przeszły. TOOL CALLING: $(tool_calling_status)"
  else die "Część testów nie przeszła (szczegóły powyżej i w $LOG_DIR/apitest.log)."
  fi
}

usage() { sed -n '3,17p' "$0" | sed 's/^# \{0,2\}//'; }

main() {
  case "${1:-start}" in
    start)   cmd_start ;;
    retest)  FORCE_RETEST=1; cmd_start ;;
    test)    cmd_test ;;
    status)  cmd_status ;;
    stop)    cmd_stop ;;
    helpers) write_helpers; echo "$BIN_DIR" ;;
    help|-h|--help) usage ;;
    *) usage; exit 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
