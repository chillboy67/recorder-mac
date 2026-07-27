const API_BASE = window.location.origin;

const state = {
  mode: 'audio',
  speakingPart: 'part2',
  audio: null,
  result: null,
  mediaRecorder: null,
  recordingChunks: [],
  recordingStartedAt: null,
  recordingTimerId: null,
  recordingUrl: null,
};

const els = {
  audioMode: document.querySelector('#audioMode'),
  textMode: document.querySelector('#textMode'),
  partButtons: document.querySelectorAll('[data-speaking-part]'),
  dropzone: document.querySelector('#dropzone'),
  recorderPanel: document.querySelector('#recorderPanel'),
  audioInput: document.querySelector('#audioInput'),
  audioName: document.querySelector('#audioName'),
  startRecordButton: document.querySelector('#startRecordButton'),
  stopRecordButton: document.querySelector('#stopRecordButton'),
  clearRecordButton: document.querySelector('#clearRecordButton'),
  recordingDot: document.querySelector('#recordingDot'),
  recordingStatus: document.querySelector('#recordingStatus'),
  recordingTimer: document.querySelector('#recordingTimer'),
  recordingPreview: document.querySelector('#recordingPreview'),
  studentField: document.querySelector('#studentField'),
  textLabel: document.querySelector('#textLabel'),
  textInput: document.querySelector('#textInput'),
  assistantNotesInput: document.querySelector('#assistantNotesInput'),
  submitButton: document.querySelector('#submitButton'),
  copyButton: document.querySelector('#copyButton'),
  errorBox: document.querySelector('#errorBox'),
  emptyState: document.querySelector('#emptyState'),
  warnings: document.querySelector('#warnings'),
  report: document.querySelector('#report'),
};

els.audioMode.addEventListener('click', () => setMode('audio'));
els.textMode.addEventListener('click', () => setMode('text'));
els.partButtons.forEach((button) => {
  button.addEventListener('click', () => setSpeakingPart(button.dataset.speakingPart));
});
els.audioInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0] || null;
  if (file) {
    setRecordedAudio(file, file.name, false);
  } else {
    clearRecording();
  }
  updateSubmitState();
});
els.textInput.addEventListener('input', updateSubmitState);
els.assistantNotesInput.addEventListener('input', updateSubmitState);
els.startRecordButton.addEventListener('click', startRecording);
els.stopRecordButton.addEventListener('click', stopRecording);
els.clearRecordButton.addEventListener('click', clearRecording);
els.submitButton.addEventListener('click', submit);
els.copyButton.addEventListener('click', copyReport);

['dragenter', 'dragover'].forEach((name) => {
  els.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropzone.classList.add('dragging');
  });
});
['dragleave', 'drop'].forEach((name) => {
  els.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove('dragging');
  });
});
els.dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    setRecordedAudio(file, file.name, false);
    updateSubmitState();
  }
});

function setMode(mode) {
  state.mode = mode;
  els.audioMode.classList.toggle('active', mode === 'audio');
  els.textMode.classList.toggle('active', mode === 'text');
  els.dropzone.classList.toggle('hidden', mode !== 'audio');
  els.recorderPanel.classList.toggle('hidden', mode !== 'audio');
  els.studentField.classList.toggle('hidden', mode === 'audio');
  els.textLabel.textContent = mode === 'audio'
    ? '学生发言'
    : '学生发言';
  updateSubmitState();
}

function setSpeakingPart(speakingPart) {
  state.speakingPart = speakingPart || 'part2';
  els.partButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.speakingPart === state.speakingPart);
  });
}

function updateSubmitState() {
  const hasText = Boolean(els.textInput.value.trim());
  const isRecording = state.mediaRecorder?.state === 'recording';
  els.submitButton.disabled = isRecording || (state.mode === 'audio' ? !(state.audio || hasText) : !hasText);
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setError('当前浏览器不支持直接录音，请改用上传音频。');
    return;
  }
  setError('');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    state.mediaRecorder = recorder;
    state.recordingChunks = [];
    recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) state.recordingChunks.push(event.data);
    });
    recorder.addEventListener('stop', () => finishRecording(stream, recorder.mimeType));
    recorder.start();
    state.recordingStartedAt = Date.now();
    state.recordingTimerId = window.setInterval(updateRecordingTimer, 250);
    updateRecordingUi(true);
    updateRecordingTimer();
  } catch (error) {
    setError('无法访问麦克风，请在浏览器里允许麦克风权限。');
  }
}

function stopRecording() {
  if (state.mediaRecorder?.state === 'recording') {
    state.mediaRecorder.stop();
  }
}

function finishRecording(stream, mimeType) {
  stream.getTracks().forEach((track) => track.stop());
  window.clearInterval(state.recordingTimerId);
  state.recordingTimerId = null;
  const type = mimeType || 'audio/webm';
  const extension = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
  const blob = new Blob(state.recordingChunks, { type });
  const file = new File([blob], `ielts-recording-${new Date().toISOString().replace(/[:.]/g, '-')}.${extension}`, { type });
  setRecordedAudio(file, '已录音，可试听或直接生成反馈', true);
  state.mediaRecorder = null;
  state.recordingChunks = [];
  updateRecordingUi(false);
  updateSubmitState();
}

function setRecordedAudio(file, label, showPreview) {
  if (state.recordingUrl) URL.revokeObjectURL(state.recordingUrl);
  state.audio = file;
  els.audioName.textContent = label;
  els.clearRecordButton.disabled = false;
  if (showPreview) {
    state.recordingUrl = URL.createObjectURL(file);
    els.recordingPreview.src = state.recordingUrl;
    els.recordingPreview.classList.remove('hidden');
    els.recordingStatus.textContent = '录音完成，可以试听、清除或生成反馈';
  } else {
    state.recordingUrl = null;
    els.recordingPreview.removeAttribute('src');
    els.recordingPreview.classList.add('hidden');
    els.recordingStatus.textContent = '已选择音频文件，可以生成反馈';
  }
}

function clearRecording() {
  if (state.mediaRecorder?.state === 'recording') stopRecording();
  if (state.recordingUrl) URL.revokeObjectURL(state.recordingUrl);
  state.audio = null;
  state.recordingUrl = null;
  state.recordingChunks = [];
  els.audioInput.value = '';
  els.audioName.textContent = '上传雅思口语音频';
  els.recordingPreview.removeAttribute('src');
  els.recordingPreview.classList.add('hidden');
  els.recordingStatus.textContent = '可直接录音，录完后点生成反馈';
  els.recordingTimer.textContent = '00:00';
  els.clearRecordButton.disabled = true;
  updateRecordingUi(false);
  updateSubmitState();
}

function updateRecordingUi(isRecording) {
  els.recordingDot.classList.toggle('active', isRecording);
  els.startRecordButton.disabled = isRecording;
  els.stopRecordButton.disabled = !isRecording;
  els.clearRecordButton.disabled = isRecording || !state.audio;
  els.recordingStatus.textContent = isRecording ? '正在录音，本地保存到浏览器内存' : els.recordingStatus.textContent;
}

function updateRecordingTimer() {
  if (!state.recordingStartedAt) return;
  const seconds = Math.floor((Date.now() - state.recordingStartedAt) / 1000);
  const minutes = String(Math.floor(seconds / 60)).padStart(2, '0');
  const rest = String(seconds % 60).padStart(2, '0');
  els.recordingTimer.textContent = `${minutes}:${rest}`;
}

async function submit() {
  setBusy(true);
  setError('');
  setResult(null);
  try {
    if (state.mode === 'text' || !state.audio) {
      const response = await fetch(`${API_BASE}/api/analyze/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: els.textInput.value,
          assistant_notes: els.assistantNotesInput.value || null,
          speaking_part: state.speakingPart,
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      setResult(await response.json());
    } else {
      const form = new FormData();
      form.append('audio', state.audio, state.audio.name || 'ielts-recording.webm');
      form.append('assistant_notes', els.assistantNotesInput.value);
      form.append('speaking_part', state.speakingPart);
      const response = await fetch(`${API_BASE}/api/analyze/audio`, { method: 'POST', body: form });
      if (!response.ok) throw new Error(await response.text());
      const queued = await response.json();
      await pollJob(queued.job_id);
    }
  } catch (error) {
    setError(error.message || '分析失败');
  } finally {
    setBusy(false);
  }
}

async function pollJob(jobId) {
  for (;;) {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    setResult(payload);
    if (payload.status === 'complete' || payload.status === 'failed') return;
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

function setBusy(isBusy) {
  els.submitButton.disabled = isBusy;
  els.submitButton.classList.toggle('busy', isBusy);
  const label = els.submitButton.querySelector('.primary-label');
  if (label) label.textContent = isBusy ? '分析中…' : '生成反馈';
  if (!isBusy) updateSubmitState();
}

function setError(message) {
  els.errorBox.textContent = message;
  els.errorBox.classList.toggle('hidden', !message);
}

function setResult(result) {
  state.result = result;
  els.emptyState.classList.toggle('hidden', Boolean(result));
  els.copyButton.disabled = !result?.report_markdown;
  els.report.innerHTML = result?.report_markdown ? renderMarkdown(result.report_markdown) : '';
  els.warnings.classList.toggle('hidden', !result?.warnings?.length);
  els.warnings.innerHTML = result?.warnings?.map((warning) => `<p>${escapeHtml(warning)}</p>`).join('') || '';
  if (result?.status === 'failed') {
    setError(result.error || '分析失败');
  }
}

async function copyReport() {
  if (state.result?.report_markdown) {
    await navigator.clipboard.writeText(state.result.report_markdown);
  }
}

function renderMarkdown(markdown) {
  return markdown.split('\n').map((line) => {
    const safe = escapeHtml(line);
    if (line.startsWith('# ')) return `<h1>${escapeHtml(line.slice(2))}</h1>`;
    if (line.startsWith('## ')) return `<h2>${escapeHtml(line.slice(3))}</h2>`;
    if (line.startsWith('- ')) return `<li>${escapeHtml(line.slice(2))}</li>`;
    if (!line.trim()) return '<div class="space"></div>';
    return `<p>${safe}</p>`;
  }).join('');
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  })[char]);
}


setMode('audio');
