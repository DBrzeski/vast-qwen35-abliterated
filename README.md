# vast-qwen35-abliterated

Jednokomendowy instalator i launcher modelu **Qwen3.5-35B-A3B Abliterated (GGUF Q4_K_M)** na **llama.cpp + CUDA** dla instancji Vast.ai. Wystawia API zgodne z OpenAI na `127.0.0.1:8081` i jest przygotowany pod agenta programistycznego [OpenCode](https://opencode.ai).

| | |
|---|---|
| Model | [`amarck/Qwen3.5-35B-A3B-abliterated-GGUF`](https://huggingface.co/amarck/Qwen3.5-35B-A3B-abliterated-GGUF), plik `abliterated_Q4_K_M.gguf` (21,2 GB, jeden plik) |
| Silnik | llama.cpp **v0.7.0** (build b11669, commit `620d2225`), kompilowany z CUDA |
| API | `http://127.0.0.1:8081/v1`, alias modelu `qwen-coder` |
| Kontekst | próby: 262144 → 131072 → 65536 → 32768, wybierany największy, który przejdzie testy |

## Uruchomienie (terminal JupyterLab na Vast.ai)

```bash
curl -fsSL https://raw.githubusercontent.com/DBrzeski/vast-qwen35-abliterated/main/start.sh -o /workspace/start.sh && bash /workspace/start.sh
```

Pierwsze uruchomienie trwa zwykle 20–40 minut. Składają się na to: kompilacja llama.cpp, pobranie 21 GB i testy kontekstów. Każde kolejne uruchomienie korzysta z już pobranego modelu, gotowej kompilacji i zapisanej konfiguracji, więc zajmuje 1–2 minuty.

## Co robi `start.sh`

1. Wykrywa system (Linux z `apt`), GPU (`nvidia-smi`), wersję CUDA sterownika, compute capability i RAM.
2. Sprawdza `nvcc`. Jeśli go brakuje, instaluje `cuda-toolkit-12-x` zgodny ze sterownikiem z oficjalnego repozytorium NVIDIA.
3. Sprawdza wolne miejsce na `/workspace` (model + kompilacja + zapas).
4. Doinstalowuje tylko brakujące pakiety: `git cmake build-essential curl python3 libssl-dev aria2 iproute2`.
5. Pobiera llama.cpp do `/workspace/llama.cpp` w przypiętej wersji v0.7.0 i weryfikuje hash commitu. Jeśli katalog istnieje w innej wersji i nie ma lokalnych zmian, przełącza go na tę wersję. Jeśli ma zmiany albo jest innym repozytorium, przerywa bez nadpisywania.
6. Kompiluje sam `llama-server` z CUDA do osobnego katalogu `build-vastqwen/`. Robi to tylko raz: stempel zawiera commit, wersję nvcc, CPU i flagi.
7. Pobiera model z Hugging Face przypiętego do rewizji `1d077ff7`. Pobieranie idzie przez aria2c (16 połączeń) z wznawianiem i kończy się weryfikacją **SHA256**. Kolejne uruchomienia nie pobierają go ponownie.
8. Odczytuje z nagłówka GGUF natywny kontekst modelu (`qwen35moe.context_length = 262144`) i nie konfiguruje nic powyżej niego.
9. Testuje konfiguracje kontekstu od największej (szczegóły niżej), zapisuje zwycięską w `/workspace/vast-qwen35/selected.env` i zostawia serwer uruchomiony w tle.
10. Generuje `opencode.json` z faktycznie wybranym kontekstem i wypisuje raport.

### Parametry serwera

```text
llama-server -m abliterated_Q4_K_M.gguf --alias qwen-coder --host 127.0.0.1 --port 8081
  -c <wybrany> -ngl all -fa on -ctk <q8_0|q4_0> -ctv <q8_0|q4_0> -np 1 -b 2048 -ub 1024
  --jinja --fit off --no-webui --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0
```

- `--jinja` włącza wbudowany w GGUF szablon czatu Qwen3.5. llama.cpp v0.7.0 rozpoznaje jego format wywołań narzędzi (`<tool_call><function=…>`, parser „Qwen3-Coder”) i zamienia go na `tool_calls` w formacie OpenAI.
- `-ngl all`, `--fit off`: cały model ma być w VRAM. Jeśli się nie zmieści, konfiguracja jest odrzucana, a nie po cichu częściowo przenoszona do RAM.
- `-fa on` włącza Flash Attention, które jest też wymagane do kwantyzacji cache V.
- Thinking (`reasoning_content`) zostaje włączony, bo tak działa szablon Qwen3.5. Próbkowanie odpowiada zaleceniom Qwen do kodowania i klient może je nadpisać.

### Wybór kontekstu i testy stabilności

Kolejność prób: `262144/q8_0 → 262144/q4_0 → 131072/q8_0 → 131072/q4_0 → 65536/… → 32768/…`. Wariant `q4_0` jest próbowany tylko wtedy, gdy `q8_0` nie mieści się w pamięci.

Dla każdej konfiguracji skrypt:

1. uruchamia serwer i czeka na `/health` = 200, przerywając przy awarii procesu albo komunikacie o OOM w logu;
2. sprawdza w logu, czy wszystkie warstwy trafiły na GPU (`offloaded N/N layers`);
3. mierzy VRAM przez `nvidia-smi` i wymaga co najmniej 1536 MiB wolnego;
4. sprawdza, czy `/v1/models` zawiera `qwen-coder`;
5. testuje generowanie: odpowiedź musi kończyć się `finish_reason=stop`, nie może mieć cech pętli (`length` przy limicie 4096 albo powtarzający się tekst); przy okazji mierzy prędkość w tok/s;
6. testuje tool calling:
   - `get_time` musi zwrócić `finish_reason=tool_calls` i poprawny JSON w `arguments`;
   - kontynuacja po wyniku narzędzia (`role: tool`) musi dać zwykłą odpowiedź;
   - `read_file` musi dostać argument `path`;
   - to samo jest sprawdzane w trybie streaming, z którego korzysta OpenCode;
   - wywołania narzędzi zwrócone jako zwykły tekst w `content` (`<tool_call>`, `<function=`) są wykrywane i kończą test wynikiem **FAIL**;
7. testuje długi kontekst: wysyła prompt wypełniający ~90% kontekstu (np. ~235 tys. tokenów dla 256K) z ukrytym kodem i sprawdza, czy model go odnajduje (needle-in-a-haystack);
8. ponownie sprawdza log pod kątem OOM, mierzy VRAM i potwierdza, że serwer działa;
9. odrzuca konfigurację, jeśli generowanie jest wolniejsze niż 10 tok/s.

Jeśli test funkcjonalny (np. tool calling) nie przejdzie, skrypt przechodzi do mniejszego kontekstu. Jeśli nic nie przejdzie, kończy się błędem i wypisuje listę prób.

**Szacunek VRAM (nie pomiar):** tylko 10 z 40 warstw Qwen3.5-35B-A3B ma pełną atencję z 2 głowicami KV, a pozostałe to DeltaNet ze stałym stanem. Dzięki temu KV cache dla 256K w `q8_0` zajmuje ok. 2,9 GB. Z wagami 21,2 GB i buforami obliczeń daje to ok. 25–28 GB, więc 256K powinno zmieścić się na RTX A6000 48 GB. Faktyczne wartości mierzy skrypt i podaje je w raporcie.

## Raport

Na koniec skrypt wypisuje (i zapisuje do `/workspace/vast-qwen35/report.txt`) wartości odczytane ze środowiska, np.:

```text
 MODEL:          Qwen3.5-35B-A3B Abliterated (amarck/Qwen3.5-35B-A3B-abliterated-GGUF)
 QUANTIZATION:   Q4_K_M
 GPU:            1x NVIDIA RTX A6000
 CONTEXT:        <wybrany> (natywny max modelu: 262144)
 KV CACHE:       Q8_0 (flash attention: on)
 VRAM TOTAL / USED / FREE: <z nvidia-smi>
 LLAMA.CPP:      version: 0.7.0-dev (build 11669, commit 620d2225c) (v0.7.0, 620d2225cc)
 API:            http://127.0.0.1:8081/v1  (OK)
 TOOL CALLING:   PASS
 STATUS:         RUNNING
```

## Połączenie z Windows (tunel SSH)

IP i port SSH znajdziesz w panelu Vast.ai (Instances → „Connect”). Skrypt wypisuje gotową komendę, jeśli instancja udostępnia zmienne `PUBLIC_IPADDR` i `VAST_TCP_PORT_22`.

```powershell
ssh -p <PORT_SSH> root@<IP_INSTANCJI> -N -L 8080:127.0.0.1:8081
```

Okno z tunelem musi pozostać otwarte. Sprawdzenie API w drugim oknie PowerShell:

```powershell
curl.exe http://127.0.0.1:8080/v1/models
```

```powershell
curl.exe http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{\"model\":\"qwen-coder\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":512}'
```

Serwer nasłuchuje tylko na `127.0.0.1` i nie jest wystawiony publicznie. Dostęp jest możliwy wyłącznie przez tunel SSH.

## OpenCode

Skrypt generuje `/workspace/vast-qwen35/opencode.json` z faktycznie wybranym kontekstem. Plik [`opencode.json`](opencode.json) w repozytorium to wersja dla 262144. Jeśli instalator wybierze mniejszy kontekst, użyj wygenerowanej wersji albo zmień `limit.context`.

Pobranie na Windows (przez `scp`):

```powershell
scp -P <PORT_SSH> root@<IP_INSTANCJI>:/workspace/vast-qwen35/opencode.json $env:USERPROFILE\.config\opencode\opencode.json
```

Plik możesz też umieścić w katalogu projektu (`opencode.json`). Jeśli masz już globalną konfigurację OpenCode, scal sekcję `provider.vast` zamiast nadpisywać plik.

Format (`provider` / `npm: @ai-sdk/openai-compatible` / `options.baseURL` / `models.*.limit`) odpowiada przykładowi „llama.cpp” z [dokumentacji OpenCode](https://opencode.ai/docs/providers/). Aktualny schemat `https://opencode.ai/config.json` oznacza pole `provider` jako przestarzałe na rzecz nowego `providers`, ale nadal je obsługuje. Dokumentacja wciąż używa `provider`, więc ten format działa również ze starszymi wersjami OpenCode.

Uwaga: przy włączonym thinking tokeny rozumowania liczą się do limitu `output` (8192). Jeśli odpowiedzi będą ucinane, zwiększ `limit.output`, np. do 16384.

## Komendy

```bash
bash /workspace/start.sh           # instalacja / start (idempotentne)
bash /workspace/start.sh status    # stan serwera, konfiguracja, VRAM
bash /workspace/start.sh test      # szybkie testy API i tool calling działającego serwera
bash /workspace/start.sh retest    # ponowne testy wszystkich kontekstów (np. po zmianie ustawień)
bash /workspace/start.sh stop      # zatrzymanie serwera uruchomionego przez skrypt
```

Wybrane zmienne środowiskowe (wszystkie są na początku `start.sh`):

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `CTX_CANDIDATES` | `262144 131072 65536 32768` | konteksty do sprawdzenia, w kolejności |
| `KV_TYPES` | `q8_0 q4_0` | typy KV cache |
| `LONG_TEST` | `1` | `0` wyłącza test wypełnienia kontekstu (szybciej, ale słabsza weryfikacja) |
| `MIN_FREE_VRAM_MIB` | `1536` | wymagany zapas VRAM |
| `MIN_GEN_TPS` | `10` | minimalna akceptowalna prędkość generowania |
| `QWEN_PORT` | `8081` | port serwera |
| `FORCE_RETEST` | `0` | `1` = ignoruj zapisaną konfigurację |
| `KEEP_LLAMA_CHECKOUT` | `0` | `1` = nie przełączaj istniejącego llama.cpp na v0.7.0 (o ile obsługuje qwen35moe) |
| `HF_TOKEN` | – | niepotrzebny (model jest publiczny); użyty tylko, jeśli ustawiony |

## Pliki i logi

| Ścieżka | Zawartość |
|---|---|
| `/workspace/llama.cpp/build-vastqwen/bin/llama-server` | skompilowany serwer |
| `/workspace/models/Qwen3.5-35B-A3B-abliterated-GGUF/` | model |
| `/workspace/vast-qwen35/selected.env` | sprawdzona konfiguracja (usuń albo użyj `retest`, aby testować od nowa) |
| `/workspace/vast-qwen35/opencode.json` | wygenerowana konfiguracja OpenCode |
| `/workspace/vast-qwen35/logs/llama-server.log` | log serwera |
| `/workspace/vast-qwen35/logs/apitest.log` | wyniki testów API |
| `/workspace/vast-qwen35/logs/start-*.log` | logi instalatora |
| `/workspace/vast-qwen35/logs/build.log` | log kompilacji |

## Bezpieczeństwo i zależności

- llama.cpp jest przypięty do tagu `v0.7.0`, a skrypt sprawdza hash commitu `620d2225cc436ab0b001e3200e590e3b286ef9ab` i przerywa, jeśli się nie zgadza.
- Kompilacja nie pobiera prebudowanego web UI (`LLAMA_USE_PREBUILT_UI=OFF`, serwer działa z `--no-webui`).
- Model jest przypięty do rewizji HF `1d077ff70a3a61804892f7e848de466bad78ba20` i weryfikowany sumą SHA256 `e7335840…b1ae6e`.
- Pakiety systemowe pochodzą z repozytoriów dystrybucji. CUDA toolkit (tylko gdy go brakuje) pochodzi z `developer.download.nvidia.com`.
- Skrypt nie zabija obcych procesów: jeśli port 8081 jest zajęty przez coś innego, pokazuje ten proces i przerywa.
- Repozytorium nie zawiera wag, kluczy, tokenów ani haseł.
- Plik `security_cv.gguf` (control vector z repozytorium modelu) **nie** jest pobierany ani używany.

## Testy repozytorium

`tests/test_logic.sh` testuje logikę `start.sh` bez GPU. Podstawia atrapę `nvidia-smi` i sprawdza:
- parsowanie GPU i CUDA;
- wybór wersji toolkitu;
- wykrywanie zajętego portu bez zabijania procesu;
- kolejność prób kontekstu i KV;
- zapis i ponowne użycie konfiguracji;
- generowanie `opencode.json`.

```bash
bash tests/test_logic.sh
```
