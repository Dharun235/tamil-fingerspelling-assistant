const video = document.querySelector('#video');
const canvas = document.querySelector('#canvas');
const ctx = canvas.getContext('2d');
const message = document.querySelector('#message');
const preview = document.querySelector('#preview');
const status = document.querySelector('#status');
const timer = document.querySelector('#timer');
const bar = document.querySelector('#progress-bar');
const connection = document.querySelector('#connection');
const capture = document.createElement('canvas');
const captureCtx = capture.getContext('2d');
const textInput = document.querySelector('#text-input');
const referenceEmpty = document.querySelector('#reference-empty');
const referenceImage = document.querySelector('#reference-image');
const referenceLabel = document.querySelector('#reference-label');
let latest = null;
let references = new Map();
let socket;
let frameTimer;
let stream;
let running = false;
let stopping = false;
const startButton = document.querySelector('#start-camera');

async function loadReferences() {
  try {
    const response = await fetch('/api/references');
    const payload = await response.json();
    references = new Map(Object.entries(payload.references || {}));
  } catch {
    referenceEmpty.textContent = 'Reference images unavailable';
  }
}

function lastCharacter(value) {
  if (!value) return '';
  if (window.Intl?.Segmenter) {
    const segments = [...new Intl.Segmenter('ta', { granularity: 'grapheme' }).segment(value)];
    return segments.at(-1)?.segment || '';
  }
  return Array.from(value).at(-1) || '';
}

function updateReference() {
  const character = lastCharacter(textInput.value.trim());
  const reference = references.get(character);
  if (!reference) {
    referenceImage.style.display = 'none';
    referenceEmpty.style.display = 'block';
    referenceLabel.textContent = character ? `No reference for ${character}` : 'Reference sign appears here';
    return;
  }
  referenceImage.src = reference.url;
  referenceImage.style.display = 'block';
  referenceEmpty.style.display = 'none';
  referenceLabel.textContent = `${character} · class ${reference.class_id}`;
}

function drawResult() {
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (!latest) return;
  for (const hand of latest.hands) {
    const points = hand.landmarks.map(([x, y]) => [x * canvas.width, y * canvas.height]);
    const chains = { T: [1,2,3,4], I: [5,6,7,8], M: [9,10,11,12], R: [13,14,15,16], P: [17,18,19,20] };
    for (const [finger, chain] of Object.entries(chains)) {
      ctx.strokeStyle = hand.states[finger] ? '#42d866aa' : '#f04f5daa';
      ctx.lineWidth = Math.max(10, canvas.width / 55);
      ctx.lineCap = 'round';
      for (let i = 1; i < chain.length; i++) {
        ctx.beginPath(); ctx.moveTo(...points[chain[i-1]]); ctx.lineTo(...points[chain[i]]); ctx.stroke();
      }
    }
    ctx.fillStyle = '#ffffff';
    for (const [x, y] of points) { ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill(); }
  }
}

function render() {
  drawResult();
  requestAnimationFrame(render);
}

async function start() {
  if (running) return;
  stopping = false;
  startButton.disabled = true;
  connection.textContent = 'Requesting camera permission...';
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera unavailable. Open this page at http://127.0.0.1:8000, not a file or 0.0.0.0 URL.');
  }
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  video.srcObject = stream;
  await video.play();
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    connection.textContent = 'Connected — camera active';
    running = true;
    startButton.textContent = 'Stop camera';
    startButton.classList.add('stop');
    frameTimer = setInterval(sendFrame, 120);
  };
  socket.onerror = () => {
    if (!stopping) connection.textContent = 'Server connection failed';
  };
  socket.onclose = () => {
    if (frameTimer) clearInterval(frameTimer);
    frameTimer = null;
    if (!stopping) {
      running = false;
      startButton.disabled = false;
      startButton.textContent = 'Enable camera';
      startButton.classList.remove('stop');
      connection.textContent = 'Disconnected from server';
    }
  };
  socket.onmessage = event => {
    latest = JSON.parse(event.data);
    if (latest.error) return;
    message.textContent = latest.message || '(empty)';
    preview.textContent = `Preview: ${latest.preview || '—'}`;
    status.textContent = latest.status;
    timer.textContent = `${latest.stable_ms}/${latest.commit_ms} ms`;
    bar.style.width = `${Math.round((latest.progress || 0) * 100)}%`;
  };
}

function stop() {
  stopping = true;
  running = false;
  if (frameTimer) clearInterval(frameTimer);
  frameTimer = null;
  if (socket && socket.readyState <= WebSocket.OPEN) socket.close();
  socket = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.pause();
  video.srcObject = null;
  latest = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startButton.disabled = false;
  startButton.textContent = 'Enable camera';
  startButton.classList.remove('stop');
  connection.textContent = 'Camera stopped';
  status.textContent = 'PAUSED';
  timer.textContent = '0/350 ms';
  bar.style.width = '0%';
}

function sendFrame() {
  if (!socket || socket.readyState !== WebSocket.OPEN || !video.videoWidth) return;
  capture.width = 640; capture.height = 480;
  captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
  capture.toBlob(blob => { if (blob && socket.readyState === WebSocket.OPEN) socket.send(blob); }, 'image/jpeg', .7);
}

function showStartError(error) {
  startButton.disabled = false;
  connection.textContent = `Camera error (${error.name || 'unknown'}): ${error.message}`;
}

startButton.onclick = () => (running ? stop() : start().catch(showStartError));
textInput.addEventListener('input', updateReference);
loadReferences();
render();
start().catch(showStartError);
