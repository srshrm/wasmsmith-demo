# On-device "Ask this page" — LLM provider comparison

**Purpose:** compare the ways an EDS block can run an "summarise / ask this page"
AI feature, with pros and cons, to decide what (if anything) to ship. All four
client-side options below are live on one page for hands-on testing.

**Live test page:** <https://main--wasmsmith-demo--srshrm.aem.page/webllm>

The page carries the same block four times, each pinned to one provider so you
can try them side by side (in order): **built-in → download → local → auto**.
Each renders instantly and does nothing until you click its button.

---

## How to test each block (prerequisites)

The blocks light up only when their provider is actually available on **your**
machine/browser — that availability is itself the main finding, so it's worth
seeing which ones work for you.

| Block | What you need to test it | If unavailable |
| --- | --- | --- |
| **built-in** | A recent **Chrome/Edge** with the built-in model (Gemini Nano) enabled — needs supported hardware and, on some builds, a flag. | Shows "built-in AI unavailable"; nothing downloads. |
| **download** | Any browser with **WebGPU** (most modern desktop Chrome/Edge/Safari). First click downloads ~380 MB (Qwen2.5-0.5B), then runs on your GPU. | Shows a WebGPU message. |
| **local** | **Ollama** (or LM Studio) running on *your* machine with a model pulled, and `OLLAMA_ORIGINS` set to `https://main--wasmsmith-demo--srshrm.aem.page` (then restart Ollama). | Shows "no local AI server found". |
| **auto** | Nothing special — it picks the best of the three above that it finds. | Falls back to offering the download. |

**Enabling a local AI server (Ollama / LM Studio)**

The block only *calls* the server; the server decides whether to answer a
cross-origin request, so **CORS is configured on the server side**.

- **Ollama** — set the `OLLAMA_ORIGINS` environment variable to the origin your
  page is served from, then restart Ollama. Prefer a specific origin over `*`
  (`*` lets any website drive your local model):
    - Running it in a terminal: `OLLAMA_ORIGINS="https://main--wasmsmith-demo--srshrm.aem.page" ollama serve`
    - macOS app (launchd): `launchctl setenv OLLAMA_ORIGINS "https://main--wasmsmith-demo--srshrm.aem.page"`, then relaunch Ollama.
- **LM Studio** — start its local server (Developer / Local Server tab) and enable
  the **CORS** toggle there.
---

## Pros / cons

**Built-in (Chrome's model) — the cleanest "pre-installed" idea**
- ✅ No per-site download, no server, no CORS; the browser ships & manages one model.
- ✅ Fully private; zero runtime cost.
- ❌ Chrome/Edge only, still experimental/origin-trial, hardware-gated.
- ❌ Small model — quality ceiling similar to the weak download model.

**Local server (Ollama / LM Studio)**
- ✅ No download; the user can run a large, high-quality model.
- ✅ Fully private; zero runtime cost.
- ❌ Reach is near-zero on the public web — requires install + a model + CORS config.
- ❌ Public HTTPS → localhost is fragile (CORS + Private Network Access).

**Download (WebLLM on WebGPU)**
- ✅ Works on most modern desktops without any install; fully private; free after download.
- ✅ Click-gated, so it never touches page-load performance ("keep it 100").
- ❌ The unwinnable trade-off: the model small enough to download (~380 MB) is **too weak to trust**; a model good enough (1.5B–3B) is a **1–3 GB** download (minutes of waiting, large memory).
- ❌ No WebGPU = no feature (older browsers, much of mobile).

**Hosted (server/edge function calling a hosted model, e.g. Claude API)**
- ✅ Best answer quality; works on **every** browser and device; nothing to download.
- ✅ Full control over model, version, and prompt.
- ❌ Needs a backend and has per-use cost.
- ❌ Gives up the pure-client privacy story (input goes to the server).

---

- **Watch-item:** Chrome's built-in `LanguageModel` is the right long-term shape
  ("pre-installed" model the page just calls). Adopt it as progressive
  enhancement with feature-detection + fallback when it matures.