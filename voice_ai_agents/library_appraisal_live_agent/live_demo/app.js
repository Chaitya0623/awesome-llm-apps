const transcriptEl = document.querySelector("#transcript");
const teamFeedEl = document.querySelector("#teamFeed");
const neededListEl = document.querySelector("#neededList");
const readinessEl = document.querySelector("#readiness");
const callStatus = document.querySelector("#callStatus");
const modelLabel = document.querySelector("#modelLabel");
const micButton = document.querySelector("#micButton");
const cameraButton = document.querySelector("#cameraButton");
const sweepButton = document.querySelector("#sweepButton");
const newAppraisalButton = document.querySelector("#newAppraisalButton");
const cameraStage = document.querySelector("#cameraStage");
const cameraPreview = document.querySelector("#cameraPreview");
const sweepTag = document.querySelector("#sweepTag");
const frameCanvas = document.querySelector("#frameCanvas");
const textForm = document.querySelector("#textForm");
const textInput = document.querySelector("#textInput");
const localeForm = document.querySelector("#localeForm");
const localeInput = document.querySelector("#localeInput");
const packetDialog = document.querySelector("#packetDialog");
const packetMarkdownEl = document.querySelector("#packetMarkdown");
const stampEl = document.querySelector("#stamp");

const DEFAULT_API_ORIGIN = "http://127.0.0.1:4178";
const API_ORIGIN = window.location.protocol === "file:" ? DEFAULT_API_ORIGIN : window.location.origin;
const WS_ORIGIN = API_ORIGIN.replace(/^http/, "ws");
const FRAME_INTERVAL_MS = 1000;
const FRAME_WIDTH = 1024; // wider than a damage photo: spine text needs the pixels
const OUTPUT_RATE = 24000;

if (window.location.protocol === "file:") {
  window.location.replace(`${API_ORIGIN}/index.html`);
}

let sessionId = "";
let state = null;
let liveSocket = null;
let liveReady = false;
let audioContext = null;
let micStream = null;
let inputProcessor = null;
let playbackContext = null;
let playhead = 0;
const playingSources = new Set();
let cameraStream = null;
let frameTimer = null;
let framesSent = 0;
let transcript = [];
const seenEntries = new Set();

const STATUS_LABEL = { pending: "queued", pricing: "pricing…", failed: "no price found" };
const REPEATABLE = new Set(["shelving", "furniture"]);
const avatar = window.appraisalAvatar;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function money(value, currency) {
  if (value == null || value === "") return "—";
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency, maximumFractionDigits: 0 }).format(value);
  } catch (_) {
    return `${Math.round(value).toLocaleString()} ${currency}`;
  }
}

function setStatus(text, tone = "") {
  callStatus.textContent = text;
  callStatus.className = `pill ${tone}`.trim();
}

function appendSystem(text) {
  transcript.push({ id: crypto.randomUUID(), speaker: "System", text, final: true });
  renderTranscript();
}

// ------------------------------------------------------------------ rendering

function render() {
  if (!state) return;
  renderLedger();
  renderFloorPlan();
  renderNeeded();
  renderTeam();
  document.querySelector("#downloadPacket").href = sessionId ? `${API_ORIGIN}/api/sessions/${sessionId}/packet` : "#";
}

function valueCell(entry, qty = 1) {
  if (!entry.price) return `<span class="${entry.status}">${STATUS_LABEL[entry.status] || entry.status}</span>`;
  const p = entry.price;
  const source = p.sources && p.sources[0]
    ? ` · <a href="${escapeHtml(p.sources[0])}" target="_blank" rel="noopener">source</a>` : "";
  return `${money(p.mid * qty, p.currency)}<span class="sub">${money(p.low * qty, p.currency)}–${money(p.high * qty, p.currency)}${source}</span>`;
}

function freshClass(id) {
  const fresh = !seenEntries.has(id);
  seenEntries.add(id);
  return fresh ? ' class="fresh"' : "";
}

function renderLedger() {
  const { totals, room, locale, books, items, review } = state;
  const currency = totals.currency;
  document.querySelector("#pageLocale").textContent = locale.city || locale.country
    ? `Priced for ${locale.label} (${locale.currency})` : "Pricing location not set";
  if (document.activeElement !== localeInput && (locale.city || locale.country)) localeInput.value = locale.label;

  document.querySelector("#totalMid").textContent = totals.grand.mid ? money(totals.grand.mid, currency) : "—";
  document.querySelector("#totalRange").textContent = totals.grand.mid
    ? `range ${money(totals.grand.low, currency)} – ${money(totals.grand.high, currency)}`
    : state.pending ? `${state.pending} waiting for a local price` : "Waiting for the first shelf";
  document.querySelector("#bookCount").textContent = totals.books.count;
  document.querySelector("#bookSub").textContent = `${totals.books.priced} priced`;
  document.querySelector("#itemCount").textContent = totals.items.count;
  document.querySelector("#itemSub").textContent = `${totals.items.priced} priced`;
  const pct = Math.round(room.range_pct * 100);
  document.querySelector("#floorArea").textContent = room.floor_m2 ? `${room.floor_m2} m²` : "—";
  document.querySelector("#floorSub").textContent = room.floor_m2
    ? `±${pct}%, ${room.calibrated ? "calibrated" : "uncalibrated"}` : "needs a wide room pass";
  document.querySelector("#roomLine").textContent = room.samples
    ? `Room ${room.width_m} × ${room.depth_m} m, ceiling ${room.ceiling_m} m · walls ${room.wall_m2} m² · total surface ${room.total_surface_m2} m² · shelving ${room.shelf_linear_m} m`
    : room.shelf_linear_m ? `Shelving so far: ${room.shelf_linear_m} m` : "";

  document.querySelector("#bookTally").textContent = books.length ? `(${books.length})` : "";
  document.querySelector("#itemTally").textContent = items.length ? `(${items.length})` : "";
  document.querySelector("#bookRows").innerHTML = books.map((b) => `<tr${freshClass(b.id)}>
      <td>${escapeHtml(b.title)}${b.price && b.price.collectible ? '<span class="flag">collectible?</span>' : ""}<span class="sub author-inline">${escapeHtml(b.author)}</span>${b.confidence < 0.6 ? '<span class="sub">unclear spine</span>' : ""}</td>
      <td class="author-col">${escapeHtml(b.author) || "—"}</td>
      <td class="num">${valueCell(b)}</td></tr>`).join("");
  document.querySelector("#itemRows").innerHTML = items.map((i) => `<tr${freshClass(i.id)}>
      <td>${escapeHtml(i.name)}<span class="sub">${escapeHtml(i.description || i.size_hint || i.category)}</span>${REPEATABLE.has(i.category) && !i.quantity_confirmed ? '<span class="sub">count to confirm</span>' : ""}</td>
      <td class="num">${i.quantity}</td>
      <td class="num">${valueCell(i, i.quantity)}</td></tr>`).join("");
  document.querySelector("#bookEmpty").hidden = books.length > 0;
  document.querySelector("#itemEmpty").hidden = items.length > 0;

  const hasContent = books.length + items.length > 0;
  stampEl.hidden = !hasContent;
  stampEl.textContent = review.routing.replaceAll("_", " ");
  stampEl.classList.toggle("ok", review.routing === "standard_contents");
}

function renderFloorPlan() {
  const figure = document.querySelector("#floorPlan");
  const plan = state.floor_plan;
  figure.hidden = !plan;
  if (!plan) return;
  const img = document.querySelector("#floorPlanImage");
  if (img.dataset.version !== String(plan.version)) {
    img.dataset.version = plan.version;
    img.src = `${API_ORIGIN}/api/sessions/${sessionId}/floor-plan?v=${plan.version}`;
  }
  const r = plan.room;
  const stale = r.floor_m2 !== state.room.floor_m2
    ? ' <span class="stale">· measurements changed since drawn, ask for a redraw</span>' : "";
  document.querySelector("#floorPlanCaption").innerHTML =
    `Floor plan v${plan.version}, an illustration. Measured ${r.width_m} × ${r.depth_m} m, floor ${r.floor_m2} m²${stale}`;
}

function renderNeeded() {
  const { locale, books, items, room, review } = state;
  const entries = [...books, ...items];
  const checks = [
    ["Pricing location set", Boolean(locale.city || locale.country)],
    ["Shelves swept", books.length > 0],
    ["Wide pass of the room", room.samples >= 3],
    ["One real measurement confirmed", room.calibrated],
    ["Everything priced", entries.length > 0 && entries.every((e) => e.status === "priced" || e.status === "failed")],
  ];
  const extras = [...review.specialist_referrals, ...(review.unconfirmed_counts || []), ...review.measurement_notes].slice(0, 4);
  neededListEl.innerHTML = checks.map(([label, done]) =>
    `<li class="${done ? "done" : ""}"><span class="tick-box"></span><span>${label}</span></li>`).join("")
    + extras.map((text) => `<li><span class="tick-box"></span><span>${escapeHtml(text)}</span></li>`).join("");
  const doneCount = checks.filter(([, done]) => done).length;
  readinessEl.textContent = `${doneCount} of ${checks.length}`;
  readinessEl.className = `pill ${doneCount === checks.length ? "live" : "neutral"}`;
}

function renderTeam() {
  const activity = [...(state.tool_activity || [])].reverse();
  teamFeedEl.innerHTML = activity.length
    ? activity.map((item) => `<li><span class="team-dot ${item.phase}"></span><span>${escapeHtml(item.headline)}
        <span class="team-meta">${escapeHtml(item.name)}${item.duration_ms ? ` · ${(item.duration_ms / 1000).toFixed(1)}s` : ""}</span></span></li>`).join("")
    : '<li><span class="team-dot"></span><span>Waiting for the call to start</span></li>';
}

function renderTranscript() {
  transcriptEl.innerHTML = transcript.map((turn) => {
    const role = turn.speaker === "Claimant" ? "claimant" : "agent";
    return `<div class="turn ${role}${turn.final ? "" : " partial"}">${escapeHtml(turn.text)}</div>`;
  }).join("");
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function upsertStreamingTurn(speaker, text, final = false, id = crypto.randomUUID()) {
  const existing = transcript.find((turn) => turn.id === id);
  if (existing) Object.assign(existing, { text, final });
  else transcript.push({ id, speaker, text, final });
  renderTranscript();
}

function applyServerState(nextState) {
  state = nextState;
  setSweeping(state.sweeping);
  render();
}

// ------------------------------------------------------------------ session + live socket

async function api(path, options = {}) {
  const response = await fetch(`${API_ORIGIN}${path}`, { credentials: "same-origin", ...options });
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || response.statusText);
  return response.json();
}

async function createSession(resume = true) {
  const previous = sessionStorage.getItem("appraisalSession");
  let payload = null;
  if (previous && resume) {
    try { payload = await api(`/api/sessions/${previous}`); } catch (_) { sessionStorage.removeItem("appraisalSession"); }
  } else if (previous) {
    await api(`/api/sessions/${previous}`, { method: "DELETE" }).catch(() => {});
    sessionStorage.removeItem("appraisalSession");
  }
  payload = payload || await api("/api/sessions", { method: "POST" });
  sessionId = payload.session_id;
  sessionStorage.setItem("appraisalSession", sessionId);
  transcript = [];
  seenEntries.clear();
  renderTranscript();
  applyServerState(payload.state);
  modelLabel.textContent = payload.has_api_key ? "Gemini 3.8 Live · ADK appraisal team" : "Add GOOGLE_API_KEY to .env to start";
  const health = await api("/api/health").catch(() => null);
  if (health) avatar?.configure(health.avatar);
}

function send(message) {
  if (liveSocket && liveSocket.readyState === WebSocket.OPEN) liveSocket.send(JSON.stringify(message));
}

function connectLive() {
  return new Promise((resolve, reject) => {
    const avatarMode = avatar?.supported === false ? "&avatar=off" : "";
    avatar?.connecting();
    const socket = new WebSocket(`${WS_ORIGIN}/ws/live?session_id=${encodeURIComponent(sessionId)}${avatarMode}`);
    liveSocket = socket;
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "ready") {
        liveReady = true;
        setStatus("Live", "live");
        sweepButton.disabled = false;
        if (cameraStream) send({ type: "camera_state", enabled: true });
        resolve();
      } else if (message.type === "session") {
        avatar?.configure(message.avatar);
        modelLabel.textContent = `${message.model} · vision ${message.vision_model}`;
      } else if (message.type === "transcript") {
        upsertStreamingTurn(message.speaker, message.text, message.final, message.id);
      } else if (message.type === "tool") {
        if (state) {
          state.tool_activity = [...(state.tool_activity || []).filter((t) => t.id !== message.id), message];
          renderTeam();
        }
      } else if (message.type === "audio") {
        playPcm24(message.data);
      } else if (message.type === "avatar_video") {
        avatar?.append(message.data);
      } else if (message.type === "turn_complete") {
        avatar?.finishTurn();
      } else if (message.type === "state") {
        applyServerState(message.state);
      } else if (message.type === "interrupted") {
        stopPlayback();
        avatar?.interrupt();
      } else if (message.type === "error") {
        appendSystem(message.message);
      }
    };
    socket.onerror = () => reject(new Error("Could not reach the live server."));
    socket.onclose = () => {
      if (liveSocket === socket) stopLiveVoice(false);
      reject(new Error("Live connection closed."));
    };
  });
}

// ------------------------------------------------------------------ audio

function resampleToPcm16(input, inputRate, outputRate) {
  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(input.length / ratio);
  const pcm = new Int16Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const index = i * ratio;
    const before = Math.floor(index);
    const after = Math.min(before + 1, input.length - 1);
    const weight = index - before;
    const sample = Math.max(-1, Math.min(1, input[before] * (1 - weight) + input[after] * weight));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm;
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

function playPcm24(base64) {
  if (!playbackContext) playbackContext = new AudioContext({ sampleRate: OUTPUT_RATE });
  const binary = atob(base64);
  const pcm = new Int16Array(binary.length / 2);
  for (let i = 0; i < pcm.length; i += 1) {
    pcm[i] = binary.charCodeAt(i * 2) | (binary.charCodeAt(i * 2 + 1) << 8);
  }
  const buffer = playbackContext.createBuffer(1, pcm.length, OUTPUT_RATE);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i += 1) channel[i] = pcm[i] / 0x8000;
  const source = playbackContext.createBufferSource();
  source.buffer = buffer;
  source.connect(playbackContext.destination);
  const at = Math.max(playhead, playbackContext.currentTime + 0.02);
  source.start(at);
  playhead = at + buffer.duration;
  playingSources.add(source);
  source.onended = () => playingSources.delete(source);
}

function stopPlayback() {
  playingSources.forEach((source) => { try { source.stop(); } catch (_) {} });
  playingSources.clear();
  playhead = 0;
}

async function startLiveVoice() {
  micButton.disabled = true;
  setStatus("Connecting");
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    audioContext = new AudioContext();
    if (!playbackContext) playbackContext = new AudioContext({ sampleRate: OUTPUT_RATE });
    await playbackContext.resume();
    await connectLive();
    const source = audioContext.createMediaStreamSource(micStream);
    inputProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    inputProcessor.onaudioprocess = (event) => {
      if (!liveReady) return;
      const pcm16 = resampleToPcm16(event.inputBuffer.getChannelData(0), audioContext.sampleRate, 16000);
      send({ type: "audio", data: arrayBufferToBase64(pcm16.buffer) });
    };
    source.connect(inputProcessor);
    inputProcessor.connect(audioContext.destination);
    micButton.classList.add("on");
    micButton.querySelector(".round-label").textContent = "Hang up";
  } catch (error) {
    appendSystem(error.message || "Microphone permission is needed to talk to the agent.");
    stopLiveVoice();
  } finally {
    micButton.disabled = false;
  }
}

function stopLiveVoice(closeSocket = true) {
  liveReady = false;
  if (closeSocket && liveSocket && liveSocket.readyState === WebSocket.OPEN) {
    liveSocket.send(JSON.stringify({ type: "close" }));
    liveSocket.close();
  }
  liveSocket = null;
  inputProcessor?.disconnect();
  micStream?.getTracks().forEach((track) => track.stop());
  audioContext?.close();
  inputProcessor = micStream = audioContext = null;
  stopPlayback();
  avatar?.reset();
  setStatus("Offline");
  setSweeping(false);
  sweepButton.disabled = true;
  micButton.classList.remove("on");
  micButton.querySelector(".round-label").textContent = "Talk";
}

// ------------------------------------------------------------------ camera

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
  } catch (_) {
    appendSystem("Camera permission is needed to scan the library.");
    return;
  }
  cameraPreview.srcObject = cameraStream;
  cameraStage.hidden = false;
  cameraButton.classList.add("on");
  cameraButton.querySelector(".round-label").textContent = "Hide camera";
  send({ type: "camera_state", enabled: true });
  frameTimer = setInterval(sendFrame, FRAME_INTERVAL_MS);
}

function stopCamera() {
  clearInterval(frameTimer);
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
  cameraStage.hidden = true;
  cameraButton.classList.remove("on");
  cameraButton.querySelector(".round-label").textContent = "Show camera";
  send({ type: "camera_state", enabled: false });
}

function sendFrame() {
  if (!liveReady || !cameraPreview.videoWidth) return;
  const scale = Math.min(1, FRAME_WIDTH / cameraPreview.videoWidth);
  frameCanvas.width = Math.round(cameraPreview.videoWidth * scale);
  frameCanvas.height = Math.round(cameraPreview.videoHeight * scale);
  frameCanvas.getContext("2d").drawImage(cameraPreview, 0, 0, frameCanvas.width, frameCanvas.height);
  const dataUrl = frameCanvas.toDataURL("image/jpeg", 0.8);
  send({ type: "video", data: dataUrl.split(",")[1] });
  framesSent += 1;
  document.querySelector("#frameCount").textContent = state ? state.frames_scanned : framesSent;
}

function setSweeping(on) {
  sweepButton.classList.toggle("on", on);
  sweepButton.querySelector(".round-label").textContent = on ? "Stop sweep" : "Sweep";
  sweepTag.hidden = !on;
}

// ------------------------------------------------------------------ wiring

micButton.addEventListener("click", () => (liveSocket ? stopLiveVoice() : startLiveVoice()));
cameraButton.addEventListener("click", () => (cameraStream ? stopCamera() : startCamera()));
sweepButton.addEventListener("click", () => send({ type: "sweep", enabled: !(state && state.sweeping) }));
newAppraisalButton.addEventListener("click", async () => {
  stopLiveVoice();
  if (cameraStream) stopCamera();
  await createSession(false);
});
textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  if (!liveReady) return appendSystem("Start the call with Talk, then type.");
  send({ type: "text", text, id: crypto.randomUUID() });
  textInput.value = "";
});
localeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!liveReady) return appendSystem("Start the call with Talk, then set the location.");
  send({ type: "locale", text: localeInput.value });
});
document.querySelector("#openPacket").addEventListener("click", () => {
  packetMarkdownEl.textContent = state && state.packet_markdown
    ? state.packet_markdown
    : "No packet yet. Ask the agent to prepare the appraisal packet when the sweep is done.";
  packetDialog.showModal();
});
document.querySelector("#closePacket").addEventListener("click", () => packetDialog.close());

if (avatar) {
  avatar.onFallback = () => {
    stopLiveVoice();
    setStatus("Voice mode · tap Talk to reconnect");
  };
}

createSession().catch((error) => appendSystem(error.message));
