// webllm.js — "Ask this page": AI summary + Q&A over the CURRENT page's text,
// fully client-side (no backend, no API key). Answers are grounded only in the
// page content that is extracted and handed to the model.
//
// PROVIDER CASCADE (the point of this block): on the single "Enable on-device AI"
// click we detect the best available on-device provider, in this order —
//   1. Chrome's built-in model (Prompt API / `LanguageModel`, Gemini Nano) — the
//      browser ships & manages one model shared across all sites: NO per-site
//      download, no server, no CORS. The clean "pre-installed model" path.
//   2. A local LLM server the user already runs (Ollama / LM Studio on
//      `localhost`, OpenAI-compatible). NO download, high-quality model — but
//      only reachable when CORS/origin config allows it (probe fails gracefully).
//   3. WebLLM in-browser download (WebGPU) — the fallback. Only chosen as an
//      EXPLICIT last resort, because it means downloading hundreds of MB.
//
// IMPORTANT loading contract (this is why it fits an EDS site at all):
//   * decorate() does ZERO network + ZERO compute. It renders a cheap static UI
//     and only does synchronous capability sniffs to pick a button label.
//   * All detection (availability checks, the localhost probe) and any download
//     happen ONLY after the user clicks. Nothing here touches LCP or the bundle.
//   * The WebLLM fallback runs inference on WebGPU (GPU compute shaders), NOT
//     WASM — so this is deliberately not a wasmsmith kernel.

import { readBlockConfig } from '../../scripts/aem.js';

// Pin the CDN version for reproducibility. esm.run serves an ES module build.
const WEBLLM_CDN = 'https://esm.run/@mlc-ai/web-llm@0.2.79';

// Friendly download-size hints for the models we suggest by default. Sizes are
// approximate q4-quantized weight footprints; used only for the button label.
const SIZE_HINTS = {
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC': '~380 MB',
  'Qwen2.5-0.5B-Instruct-q4f32_1-MLC': '~530 MB',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC': '~950 MB',
  'Llama-3.2-1B-Instruct-q4f16_1-MLC': '~730 MB',
  'Llama-3.2-1B-Instruct-q4f32_1-MLC': '~1.1 GB',
  'Phi-3.5-mini-instruct-q4f16_1-MLC': '~2.2 GB',
  'gemma-2-2b-it-q4f16_1-MLC': '~1.6 GB',
};

// Intro copy per provider, so each block on a page explains what it actually
// uses (built-in vs local vs download) instead of the generic cascade text.
const INTRO = {
  builtin: 'Summarise or ask about this page using Chrome’s built-in AI (Gemini Nano) — it runs on your device with no download and no server. Nothing you type leaves your device.',
  local: 'Summarise or ask about this page using a local AI server (e.g. Ollama or LM Studio) running on your own machine — no download, and nothing you type leaves your device.',
  download: 'Summarise or ask about this page using an AI model downloaded into your browser and run on your GPU (WebGPU). The first use downloads the model (cached afterwards); nothing you type leaves your device.',
  auto: 'Summarise or ask about this page with AI that runs on your device — it prefers your browser’s built-in model or a local AI server, and only offers to download a model as a last resort. Nothing you type leaves your device.',
};

function readConfig(block) {
  const raw = readBlockConfig(block);
  const str = (v) => (Array.isArray(v) ? v.join(', ') : `${v ?? ''}`).trim();
  const model = str(raw.model);
  const maxChars = parseInt(str(raw['max-context-chars'] || raw.maxcontextchars), 10);
  const localRaw = str(raw['local-server'] || raw.localserver).toLowerCase();
  const providerRaw = str(raw.provider).toLowerCase();
  const providerMap = {
    builtin: 'builtin',
    'built-in': 'builtin',
    chrome: 'builtin',
    local: 'local',
    'local-server': 'local',
    ollama: 'local',
    download: 'download',
    webllm: 'download',
    auto: 'auto',
  };
  return {
    // Explicit author choice wins; otherwise we pick a compatible default once
    // we know whether the GPU supports 16-bit shaders (see pickDefaultModel).
    model: model || '',
    placeholder: str(raw.placeholder) || 'Ask a question about this page…',
    systemPrompt: str(raw['system-prompt'] || raw.systemprompt)
      || 'You answer questions and write summaries based ONLY on the page content provided below. '
        + 'Be concise and factual. If the answer is not in the content, say you could not find it on this page. '
        + 'Do not use outside knowledge.',
    // Cap how much page text we feed the model — small models are slow and have
    // limited effective context. ~8000 chars ≈ ~2000 tokens.
    maxContextChars: Number.isFinite(maxChars) ? maxChars : 8000,
    // Local server (Ollama / LM Studio) probe. Author can disable it or point it
    // elsewhere. Default: probe localhost, auto-pick the first installed model.
    localServer: localRaw !== 'false' && localRaw !== 'off' && localRaw !== 'no',
    // Strip trailing slash(es) so `${localBase}/api/tags` never doubles up — a
    // `//` path makes Ollama redirect, and the redirect response carries no CORS
    // header, which surfaces as a misleading "No Access-Control-Allow-Origin".
    localBase: (str(raw['local-base'] || raw.localbase) || 'http://localhost:11434').replace(/\/+$/, ''),
    localModel: str(raw['local-model'] || raw.localmodel) || '',
    // Which provider to use. Default (blank) = Chrome's built-in model. `local`
    // forces the local AI server; `download` forces the WebLLM download; `auto`
    // runs the full cascade (built-in → local server → download).
    provider: providerMap[providerRaw] || 'builtin',
  };
}

// Secure-context + WebGPU capability probe. Returns { ok, reason, hasF16 }.
async function probeWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return { ok: false, reason: 'no WebGPU in this browser' };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: 'no compatible GPU adapter for WebGPU' };
    return { ok: true, hasF16: adapter.features && adapter.features.has('shader-f16') };
  } catch (err) {
    return { ok: false, reason: `WebGPU init failed: ${err.message}` };
  }
}

function pickDefaultModel(hasF16) {
  // f16 weights are smaller/faster but require the shader-f16 feature; fall back
  // to the f32 build when the GPU lacks it so the demo still runs.
  return hasF16
    ? 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC'
    : 'Qwen2.5-0.5B-Instruct-q4f32_1-MLC';
}

// Pull the readable text of the current page, excluding this block's own UI and
// non-content chrome. Returns a whitespace-collapsed string capped at maxChars.
function extractPageText(maxChars) {
  const root = document.querySelector('main') || document.body;
  const clone = root.cloneNode(true);
  clone
    .querySelectorAll('.webllm, script, style, noscript, nav, header, footer, aside, form')
    .forEach((el) => el.remove());
  const text = (clone.textContent || '').replace(/\s+/g, ' ').trim();
  return text.slice(0, maxChars);
}

// ---------------------------------------------------------------------------
// Providers. Each exposes:
//   present()              — cheap SYNCHRONOUS sniff (safe in decorate)
//   isDownload             — true if selecting it triggers a big weight download
//   async detect()         — { ok, reason?, needsDownload?, ...extra } (may hit
//                            the network / call async APIs; only run on click)
//   async load({onProgress, ...detect}) — returns a session:
//       { label, async run(userContent, onDelta) -> {text, statusText},
//         interrupt() }
// ---------------------------------------------------------------------------

// 1. Chrome's built-in Prompt API (Gemini Nano). Browser-managed, shared model.
function builtinProvider(config) {
  const api = () => window.LanguageModel
    || (window.ai && window.ai.languageModel);
  let base = null;
  let controller = null;
  return {
    id: 'builtin',
    isDownload: false,
    present: () => !!api(),
    async detect() {
      const lm = api();
      if (!lm) return { ok: false, reason: 'no built-in AI' };
      try {
        let state;
        if (typeof lm.availability === 'function') {
          state = await lm.availability();
        } else if (typeof lm.capabilities === 'function') {
          const caps = await lm.capabilities();
          const map = { readily: 'available', 'after-download': 'downloadable', no: 'unavailable' };
          state = map[caps.available] || 'unavailable';
        }
        if (!state || state === 'unavailable') {
          return { ok: false, reason: 'built-in AI unavailable on this device' };
        }
        return { ok: true, needsDownload: state !== 'available' };
      } catch (err) {
        return { ok: false, reason: `built-in AI check failed: ${err.message}` };
      }
    },
    async load({ onProgress }) {
      const lm = api();
      const monitor = (m) => {
        m.addEventListener('downloadprogress', (e) => {
          const pct = e.total ? Math.round((e.loaded / e.total) * 100) : 0;
          onProgress(pct, `Preparing built-in AI… ${pct}%`);
        });
      };
      try {
        base = await lm.create({
          initialPrompts: [{ role: 'system', content: config.systemPrompt }],
          monitor,
        });
      } catch (err) {
        // Older builds took a `systemPrompt` string instead of initialPrompts.
        base = await lm.create({ systemPrompt: config.systemPrompt, monitor });
      }
      return {
        label: "Chrome's built-in AI (on-device, no download)",
        async run(userContent, onDelta) {
          controller = new AbortController();
          // Clone off the base session so each ask starts with a clean history
          // (keeps answers grounded only in the page text we pass in).
          let session = base;
          if (typeof base.clone === 'function') {
            try {
              session = await base.clone({ signal: controller.signal });
            } catch (err) {
              session = base;
            }
          }
          const stream = session.promptStreaming(userContent, { signal: controller.signal });
          let text = '';
          // eslint-disable-next-line no-restricted-syntax
          for await (const chunk of stream) {
            // Chrome has shipped both cumulative and delta chunks; handle both.
            if (chunk.startsWith(text)) text = chunk;
            else text += chunk;
            onDelta(text);
          }
          return { text, statusText: "Done · Chrome's built-in AI" };
        },
        interrupt() {
          if (controller) controller.abort();
        },
      };
    },
  };
}

// 2. A local OpenAI-compatible server (Ollama / LM Studio) on localhost.
function localServerProvider(config) {
  let controller = null;
  let model = config.localModel;
  return {
    id: 'local',
    isDownload: false,
    // No cheap sniff — reachability needs a network probe, done in detect().
    present: () => false,
    async detect() {
      if (!config.localServer) return { ok: false, reason: 'local server probe disabled' };
      try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 1500);
        const res = await fetch(`${config.localBase}/api/tags`, { signal: ctrl.signal });
        clearTimeout(timer);
        if (!res.ok) return { ok: false, reason: 'local AI server not responding' };
        const data = await res.json();
        const models = (data.models || []).map((m) => m.name).filter(Boolean);
        if (!model) [model] = models;
        if (!model) return { ok: false, reason: 'local AI server has no models installed' };
        return { ok: true };
      } catch (err) {
        return { ok: false, reason: 'no local AI server found' };
      }
    },
    async load() {
      return {
        label: `your local AI server (${model})`,
        async run(userContent, onDelta) {
          controller = new AbortController();
          const res = await fetch(`${config.localBase}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
              model,
              stream: true,
              temperature: 0.3,
              messages: [
                { role: 'system', content: config.systemPrompt },
                { role: 'user', content: userContent },
              ],
            }),
          });
          if (!res.ok || !res.body) throw new Error(`server returned ${res.status}`);
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buf = '';
          let text = '';
          // Parse one SSE line; declared once (not in the loop) so it can safely
          // close over `text`. OpenAI-compatible servers send `data: {json}`.
          const pushLine = (line) => {
            const s = line.trim();
            if (!s.startsWith('data:')) return;
            const payload = s.slice(5).trim();
            if (!payload || payload === '[DONE]') return;
            try {
              const json = JSON.parse(payload);
              text += json.choices?.[0]?.delta?.content || '';
              onDelta(text);
            } catch (err) {
              // ignore keep-alive / non-JSON lines
            }
          };
          let done = false;
          while (!done) {
            // eslint-disable-next-line no-await-in-loop
            const { value, done: streamDone } = await reader.read();
            done = streamDone;
            if (value) buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop() || '';
            lines.forEach(pushLine);
          }
          return { text, statusText: `Done · your local AI server (${model})` };
        },
        interrupt() {
          if (controller) controller.abort();
        },
      };
    },
  };
}

// 3. WebLLM in-browser download (WebGPU). The fallback: big one-time download.
function webllmProvider(config) {
  let engine = null;
  return {
    id: 'webllm',
    isDownload: true,
    present: () => typeof navigator !== 'undefined' && !!navigator.gpu,
    async detect() {
      const probe = await probeWebGPU();
      if (!probe.ok) return { ok: false, reason: probe.reason };
      const modelId = config.model || pickDefaultModel(probe.hasF16);
      return { ok: true, needsDownload: true, modelId };
    },
    async load({ onProgress, modelId }) {
      const webllm = await import(/* webpackIgnore: true */ WEBLLM_CDN);
      engine = await webllm.CreateMLCEngine(modelId, {
        initProgressCallback: (report) => {
          const pct = Math.round((report.progress || 0) * 100);
          onProgress(pct, report.text || `Loading… ${pct}%`);
        },
      });
      return {
        label: `${modelId} · your GPU`,
        async run(userContent, onDelta) {
          const chunks = await engine.chat.completions.create({
            stream: true,
            stream_options: { include_usage: true },
            temperature: 0.3,
            messages: [
              { role: 'system', content: config.systemPrompt },
              { role: 'user', content: userContent },
            ],
          });
          let text = '';
          let statusText = 'Done · generated locally on your GPU';
          // eslint-disable-next-line no-restricted-syntax
          for await (const chunk of chunks) {
            text += chunk.choices?.[0]?.delta?.content || '';
            onDelta(text);
            if (chunk.usage) {
              const tps = chunk.usage.extra?.decode_tokens_per_s;
              statusText = tps
                ? `Done · ${Math.round(tps)} tokens/s on your GPU`
                : statusText;
            }
          }
          return { text, statusText };
        },
        interrupt() {
          if (engine) engine.interruptGenerate();
        },
      };
    },
  };
}

/**
 * loads and decorates the block
 * @param {Element} block The block element
 */
export default async function decorate(block) {
  const config = readConfig(block);
  block.textContent = '';

  // ---- static UI (no network, no compute) -------------------------------
  const card = document.createElement('div');
  card.className = 'webllm-card';

  const intro = document.createElement('p');
  intro.className = 'webllm-intro';
  intro.textContent = INTRO[config.provider] || INTRO.auto;

  const status = document.createElement('p');
  status.className = 'webllm-status';
  status.setAttribute('aria-live', 'polite');

  const progressWrap = document.createElement('div');
  progressWrap.className = 'webllm-progress';
  progressWrap.hidden = true;
  const progressBar = document.createElement('div');
  progressBar.className = 'webllm-progress-bar';
  progressWrap.append(progressBar);

  const loadBtn = document.createElement('button');
  loadBtn.type = 'button';
  loadBtn.className = 'webllm-btn webllm-load';

  // Controls shown only after a provider is ready.
  const tools = document.createElement('div');
  tools.className = 'webllm-tools';
  tools.hidden = true;

  const summarizeBtn = document.createElement('button');
  summarizeBtn.type = 'button';
  summarizeBtn.className = 'webllm-btn webllm-secondary';
  summarizeBtn.textContent = 'Summarise this page';

  const askForm = document.createElement('form');
  askForm.className = 'webllm-ask';
  const input = document.createElement('textarea');
  input.className = 'webllm-input';
  input.rows = 2;
  input.placeholder = config.placeholder;
  input.setAttribute('aria-label', config.placeholder);
  const askBtn = document.createElement('button');
  askBtn.type = 'submit';
  askBtn.className = 'webllm-btn';
  askBtn.textContent = 'Ask';
  askForm.append(input, askBtn);

  const stopBtn = document.createElement('button');
  stopBtn.type = 'button';
  stopBtn.className = 'webllm-btn webllm-stop';
  stopBtn.textContent = 'Stop';
  stopBtn.hidden = true;

  tools.append(summarizeBtn, askForm, stopBtn);

  const output = document.createElement('div');
  output.className = 'webllm-output';
  output.setAttribute('aria-live', 'polite');
  output.hidden = true;

  card.append(intro, loadBtn, progressWrap, status, tools, output);
  block.append(card);

  // ---- provider selection ------------------------------------------------
  const providers = [
    builtinProvider(config),
    localServerProvider(config),
    webllmProvider(config),
  ];
  const byId = { builtin: providers[0], local: providers[1], download: providers[2] };
  const builtinPresent = providers[0].present();
  const webgpuPresent = providers[2].present();
  const defaultModelId = config.model || 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';
  const defaultSizeHint = SIZE_HINTS[defaultModelId] || 'a large one-time download';

  function disableWith(reason) {
    loadBtn.disabled = true;
    loadBtn.textContent = 'On-device AI unavailable';
    status.textContent = reason;
    status.classList.add('webllm-warn');
  }

  // `provider` forces one path so built-in and local can be tested in isolation.
  // Default (nothing mentioned) = Chrome's built-in model. `auto` = full cascade.
  const mode = config.provider; // 'builtin' | 'local' | 'download' | 'auto'

  // Downloads (WebLLM) require explicit consent. A forced `download` allows it
  // directly; `auto` keeps it off until the no-download options fail.
  let allowDownload = mode === 'download' || (mode === 'auto' && !builtinPresent && webgpuPresent);

  if (mode === 'builtin') {
    if (!builtinPresent) {
      disableWith('This browser has no built-in AI (the LanguageModel / Prompt API). Try a recent Chrome or Edge with built-in AI enabled, or set provider=local.');
      return;
    }
    loadBtn.textContent = 'Enable Chrome built-in AI';
  } else if (mode === 'local') {
    if (!config.localServer) {
      disableWith('The local AI server is turned off in config (local-server=false).');
      return;
    }
    loadBtn.textContent = 'Connect to local AI server';
  } else if (mode === 'download') {
    if (!webgpuPresent) {
      disableWith('This browser has no WebGPU support, which the in-browser download model needs.');
      return;
    }
    loadBtn.textContent = `Load AI model (${defaultSizeHint}, cached after first use)`;
  } else {
    // auto — full cascade
    if (!builtinPresent && !webgpuPresent && !config.localServer) {
      disableWith('This browser has no built-in AI and no WebGPU support. Try a recent Chrome or Edge, or run a local AI server.');
      return;
    }
    if (builtinPresent) {
      loadBtn.textContent = 'Enable on-device AI (no per-site download)';
    } else if (webgpuPresent) {
      loadBtn.textContent = `Load AI model (${defaultSizeHint}, cached after first use)`;
    } else {
      loadBtn.textContent = 'Enable on-device AI';
    }
  }

  // ---- state -------------------------------------------------------------
  let session = null;
  let activating = false;
  let generating = false;

  function onProgress(pct, text) {
    progressWrap.hidden = false;
    progressBar.style.width = `${pct}%`;
    status.textContent = text;
  }

  async function start(provider, detected) {
    loadBtn.disabled = true;
    status.classList.remove('webllm-warn');
    if (detected.needsDownload) {
      progressWrap.hidden = false;
      status.textContent = 'Preparing the model…';
    } else {
      status.textContent = 'Connecting…';
    }
    try {
      session = await provider.load({ onProgress, modelId: detected.modelId });
      progressWrap.hidden = true;
      loadBtn.hidden = true;
      tools.hidden = false;
      status.textContent = `Ready · ${session.label}`;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('[webllm] provider load failed', err);
      progressWrap.hidden = true;
      loadBtn.disabled = false;
      status.textContent = `Could not start: ${err.message}`;
      status.classList.add('webllm-warn');
      session = null;
    }
  }

  // Forced single provider (provider=builtin | local | download).
  async function activateSingle(provider) {
    if (session || activating) return;
    activating = true;
    loadBtn.disabled = true;
    status.classList.remove('webllm-warn');
    status.textContent = provider.isDownload ? 'Preparing the model…' : 'Checking availability…';
    const detected = await provider.detect();
    activating = false;
    if (detected.ok) {
      await start(provider, detected);
      return;
    }
    loadBtn.disabled = false;
    loadBtn.textContent = 'Retry';
    status.textContent = `Not available: ${detected.reason}`;
    status.classList.add('webllm-warn');
  }

  // Full cascade (provider=auto): built-in → local server → download.
  async function activateAuto() {
    if (session || activating) return;
    activating = true;
    loadBtn.disabled = true;
    status.classList.remove('webllm-warn');
    status.textContent = 'Looking for on-device AI…';

    const reasons = [];
    let started = false;
    // eslint-disable-next-line no-restricted-syntax
    for (const provider of providers) {
      const skip = provider.isDownload && !allowDownload;
      if (!skip) {
        // eslint-disable-next-line no-await-in-loop
        const detected = await provider.detect();
        if (detected.ok) {
          // eslint-disable-next-line no-await-in-loop
          await start(provider, detected);
          started = true;
          break;
        }
        reasons.push(detected.reason);
      }
    }
    activating = false;
    if (started) return;

    if (!allowDownload && webgpuPresent) {
      // No-download options failed; offer the download as an explicit next click.
      allowDownload = true;
      loadBtn.disabled = false;
      loadBtn.textContent = `Download a model to run in-browser (${defaultSizeHint})`;
      status.textContent = 'No built-in or local AI was found. You can download a model to run in your browser instead.';
    } else {
      loadBtn.disabled = false;
      loadBtn.textContent = 'Retry';
      status.textContent = `On-device AI is unavailable. ${reasons.filter(Boolean).join(' · ')}`;
      status.classList.add('webllm-warn');
    }
  }

  // ---- generation --------------------------------------------------------
  function setBusy(busy) {
    generating = busy;
    summarizeBtn.disabled = busy;
    askBtn.disabled = busy;
    input.disabled = busy;
    stopBtn.hidden = !busy;
  }

  async function run(task) {
    if (!session || generating) return;
    const pageText = extractPageText(config.maxContextChars);
    if (!pageText) {
      output.hidden = false;
      output.textContent = 'There is no readable page content to work with.';
      return;
    }
    const userContent = `PAGE CONTENT:\n${pageText}\n\nTASK:\n${task}`;
    setBusy(true);
    output.hidden = false;
    output.textContent = '';
    status.classList.remove('webllm-warn');
    try {
      const result = await session.run(userContent, (text) => {
        output.textContent = text;
      });
      status.textContent = result.statusText || 'Done';
      if (!output.textContent.trim()) output.textContent = '(the model returned an empty response)';
    } catch (err) {
      // interrupt()/abort rejects the stream; treat that as a clean stop.
      if (!/interrupt|abort/i.test(`${err.name} ${err.message}`)) {
        // eslint-disable-next-line no-console
        console.warn('[webllm] generation failed', err);
        output.textContent = `Generation failed: ${err.message}`;
      }
    } finally {
      setBusy(false);
    }
  }

  loadBtn.addEventListener('click', () => {
    if (mode === 'auto') activateAuto();
    else activateSingle(byId[mode]);
  });
  summarizeBtn.addEventListener('click', () => run('Summarise this page in 3–5 short bullet points.'));
  askForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (q) run(q);
  });
  stopBtn.addEventListener('click', () => {
    if (session && generating) session.interrupt();
  });
}
