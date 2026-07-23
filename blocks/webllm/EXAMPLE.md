# webllm block — authoring

**"Ask this page"** — an on-device AI that summarises the current page or answers
questions about it, running entirely in the visitor's browser. No backend, no API
key, and nothing the visitor types leaves their device. It **prefers a model
that's already on the machine** and only downloads one as a last resort.

Paste this table into a Google Doc / Word / SharePoint document. Every cell is
optional; defaults are shown.

| webllm             |                                     |
| ------------------ | ----------------------------------- |
| provider           | builtin                             |
| model              | Qwen2.5-0.5B-Instruct-q4f16_1-MLC   |
| placeholder        | Ask a question about this page…     |
| max-context-chars  | 8000                                |
| system-prompt      | (see default below)                 |
| local-server       | true                                |
| local-base         | http://localhost:11434              |
| local-model        | (first installed model)             |

**Fields**

- **provider** — which on-device path to use, so each can be tested in isolation:
  - `builtin` *(default when blank)* — Chrome's built-in model only.
  - `local` — the local AI server only (Ollama / LM Studio; see below).
  - `download` — the WebLLM in-browser download only (WebGPU).
  - `auto` — try all three in order (built-in → local server → download), using
    the first that works, and offering the download only as an explicit fallback.
- **model** — an MLC model id from the WebLLM prebuilt list, used only for the
  *download fallback*. Leave blank to use a small default (Qwen2.5-0.5B),
  auto-selecting an f32 build when the GPU lacks 16-bit shader support. Larger
  models give better answers but download slower.
- **placeholder** — placeholder text for the question box.
- **max-context-chars** — how much of the page's text is fed to the model. Small
  models are slow and have limited context; the default (8000 ≈ ~2000 tokens) is
  a safe balance. The block extracts readable text from `<main>`, excluding this
  block, nav, header, footer, and forms.
- **system-prompt** — overrides the grounding instructions. The default tells the
  model to answer **only** from the page content and to admit when something is
  not on the page.
- **local-server** — set to `false` to skip probing for a local AI server (no
  localhost request is ever made). Default `true`.
- **local-base** — base URL of an OpenAI-compatible local server (Ollama / LM
  Studio). Default `http://localhost:11434`.
- **local-model** — model name to request from the local server. Blank = use the
  first model the server reports as installed.

**How it works / what to expect**

- The block renders instantly with an **"Enable on-device AI"** button. Nothing is
  detected, downloaded, or computed until the visitor clicks it — this keeps LCP
  and the JS bundle untouched, so it is safe on a performance-sensitive EDS page.
- On click, it tries providers **in order**, using the first that works:
  1. **Chrome's built-in model** (Prompt API / `LanguageModel`, Gemini Nano) — the
     browser manages one model shared across all sites: no per-site download, no
     server, no CORS.
  2. **A local AI server** you already run (Ollama / LM Studio on `localhost`) —
     no download, and you can run a large, high-quality model. Only reachable when
     the server's CORS/origin config allows this site; the probe fails gracefully.
  3. **WebLLM download** (WebGPU) — the fallback. Because it means downloading
     hundreds of MB, it is offered only after the no-download options fail and
     requires a **separate, explicit click**. The browser caches the weights, so
     later visits skip the download.
- The download fallback runs inference on **WebGPU** (the GPU), not WebAssembly —
  so unlike the wasmsmith blocks in this repo, that path is a click-to-load
  progressive enhancement, not a compiled kernel. Where no provider is available
  the block shows a clear message instead.

**Enabling a local AI server (Ollama / LM Studio)**

The block only *calls* the server; the server decides whether to answer a
cross-origin request, so **CORS is configured on the server side**.

- **Ollama** — set the `OLLAMA_ORIGINS` environment variable to the origin your
  page is served from, then restart Ollama. Prefer a specific origin over `*`
  (`*` lets any website drive your local model):
  - Running it in a terminal: `OLLAMA_ORIGINS="http://localhost:3000" ollama serve`
  - macOS app (launchd): `launchctl setenv OLLAMA_ORIGINS "http://localhost:3000"`, then relaunch Ollama.
  - Linux (systemd): `systemctl edit ollama.service` → add `Environment="OLLAMA_ORIGINS=http://localhost:3000"` under `[Service]` → `daemon-reload` + `restart`.
  - Windows: set a `OLLAMA_ORIGINS` user env var (or `setx`), then relaunch from the tray.
- **LM Studio** — start its local server (Developer / Local Server tab) and enable
  the **CORS** toggle there.

Two caveats:

- Ollama already allows `localhost`/`127.0.0.1` origins by default, so the block
  served from `http://localhost:3000` often connects with no change — you only
  need `OLLAMA_ORIGINS` when the page is on a different origin.
- For a **public HTTPS page** reaching `http://localhost`, Chrome's Private
  Network Access rules can require the server to answer a preflight with
  `Access-Control-Allow-Private-Network: true` (which Ollama does not send). So
  the local-server path is realistically a **local-dev / internal-network**
  feature, not something a public visitor gets.

**Notes**

- First-token latency and download size make this unsuitable for eager/lazy page
  phases; it is intentionally user-triggered only.
- Answers are grounded in the page text but small models can still be wrong —
  present it as an assistant, not an authority.
