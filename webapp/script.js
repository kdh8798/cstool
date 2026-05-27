const micBtn = document.querySelector('#micBtn');
const micText = document.querySelector('#micText');
const statusText = document.querySelector('#statusText');
const chatArea = document.querySelector('#chatArea');
const recognizingBox = document.querySelector('#recognizingBox');
const wordCard = document.querySelector('#wordCard');
const collapseBtn = document.querySelector('#collapseBtn');
const closeBtn = document.querySelector('#closeBtn');
const bottombar = document.querySelector('.bottombar');
const recordPopup = document.querySelector('#recordPopup');
const recordTime = document.querySelector('#recordTime');
const audioBars = document.querySelector('#audioBars');
const API_BASE_URL = '';

let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recognition;
let recordStartTime = null;
let recordTimerInterval = null;
let audioContext = null;
let analyser = null;
let micStream = null;
let animationFrameId = null;
let mockScenarioTimer = null;

/** Mock: 버튼 → 음성 인식 중 → 3초 후 고정 사용자 발화 */
const USE_MOCK_VOICE_SCENARIO = false;
const MOCK_USER_UTTERANCE = '목이 아프고 кашель 있어요';
/** 러시아어 칩 옆 한국어 뜻 (없으면 칩만 표시) */
const RU_WORD_HINTS = {
  'кашель': '기침',
  'я': '나는',
  'хочу': '원하다',
};
const MOCK_RECOGNITION_DELAY_MS = 3000;
const CYRILLIC_WORD_RE = /[\u0400-\u04FF]+/g;
const CYRILLIC_TEST_RE = /[\u0400-\u04FF]/;

/**
 * 사용자 말풍선 세그먼트 (음성인식 모델 연동용)
 *
 * @typedef {Object} TextSegment
 * @property {'text'} type
 * @property {string} value - 일반 텍스트
 *
 * @typedef {Object} ChipSegment
 * @property {'chip'} type
 * @property {string} word - 러시아어 등 강조할 단어
 * @property {string} [hint] - 괄호 안 번역 (예: 나는)
 *
 * @typedef {TextSegment|ChipSegment} MessageSegment
 */

const conversationId = new URLSearchParams(window.location.search).get('id');
const chatTitleEl = document.querySelector('#chatTitle');

if (!conversationId) {
  window.location.replace('/');
}

/** @type {Array<{id:string, role:'assistant'|'user', text?:string, segments?:MessageSegment[]}>} */
let chatMessages = [];

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * 세그먼트 배열 → 사용자 말풍선 HTML
 * @param {MessageSegment[]} segments
 */
function renderUserMessageHtml(segments) {
  return segments
    .map((seg) => {
      if (seg.type === 'chip') {
        const word = escapeHtml(seg.word);
        const hint = seg.hint
          ? ` <span class="word-hint">(${escapeHtml(seg.hint)})</span>`
          : '';
        return `<span class="word-chip">${word}</span>${hint}`;
      }
      return escapeHtml(seg.value);
    })
    .join('');
}

/**
 * 한·러 혼합 문장 → 러시아어(키릴) 구간은 단어 칩으로 분리
 * @param {string} text
 * @param {Record<string, string>} [hintMap]
 * @returns {MessageSegment[]}
 */
function lookupRuHint(word, hintMap = RU_WORD_HINTS) {
  return hintMap[word] || hintMap[word.toLowerCase()];
}

function segmentsFromMixedUtterance(text, hintMap = RU_WORD_HINTS) {
  if (!text) return [{ type: 'text', value: '' }];

  const segments = [];
  let lastIndex = 0;

  for (const match of text.matchAll(CYRILLIC_WORD_RE)) {
    const start = match.index ?? 0;
    const word = match[0];

    if (start > lastIndex) {
      segments.push({ type: 'text', value: text.slice(lastIndex, start) });
    }

    const hint = lookupRuHint(word, hintMap);
    segments.push(hint ? { type: 'chip', word, hint } : { type: 'chip', word });
    lastIndex = start + word.length;
  }

  if (lastIndex < text.length) {
    segments.push({ type: 'text', value: text.slice(lastIndex) });
  }

  return segments.length ? segments : [{ type: 'text', value: text }];
}

/**
 * 사용자 세그먼트 안의 텍스트에 섞인 키릴 문자열도 칩으로 분리
 * @param {MessageSegment[]} segments
 * @param {Record<string, string>} [hintMap]
 * @returns {MessageSegment[]}
 */
function normalizeUserSegments(segments, hintMap = RU_WORD_HINTS) {
  if (!Array.isArray(segments) || !segments.length) {
    return [{ type: 'text', value: '' }];
  }

  return segments.flatMap((seg) => {
    if (seg.type === 'chip') {
      const hint = seg.hint || lookupRuHint(seg.word, hintMap);
      return [hint ? { type: 'chip', word: seg.word, hint } : { type: 'chip', word: seg.word }];
    }

    if (seg.type === 'text' && seg.value && CYRILLIC_TEST_RE.test(seg.value)) {
      return segmentsFromMixedUtterance(seg.value, hintMap);
    }

    return [seg];
  });
}

function resolveUserMessageSegments(record) {
  if (record?.segments?.length) {
    return normalizeUserSegments(record.segments);
  }

  if (typeof record?.text === 'string') {
    return segmentsFromMixedUtterance(record.text);
  }

  return [{ type: 'text', value: '' }];
}

/**
 * 음성인식 모델 결과 → 세그먼트 (연동 시 이 함수만 수정)
 *
 * 기대 형식 예:
 * { segments: [{ type: 'text', value: '...' }, { type: 'chip', word: 'я', hint: '나는' }] }
 * 또는 { text: '...', highlights: [{ start, end, word, hint }] }
 *
 * @param {unknown} modelResult
 * @returns {MessageSegment[]}
 */
function segmentsFromModelResult(modelResult) {
  if (modelResult && Array.isArray(modelResult.segments)) {
    return normalizeUserSegments(modelResult.segments);
  }

  if (typeof modelResult === 'string') {
    return segmentsFromMixedUtterance(modelResult);
  }

  if (modelResult && typeof modelResult.text === 'string') {
    return segmentsFromMixedUtterance(modelResult.text);
  }

  return [{ type: 'text', value: '' }];
}

function createAssistantMessageElement(text) {
  const message = document.createElement('section');
  message.className = 'message assistant';
  message.innerHTML = `<p>${escapeHtml(text)}</p>`;
  return message;
}

function createMessageElement(record) {
  if (record.role === 'user') {
    return createUserMessageElement(resolveUserMessageSegments(record));
  }
  return createAssistantMessageElement(record.text || '');
}

function scrollChatToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function formatChatDateLabel(isoString) {
  const date = new Date(isoString);
  const datePart = new Intl.DateTimeFormat('ko-KR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }).format(date);
  const weekday = new Intl.DateTimeFormat('ko-KR', { weekday: 'short' })
    .format(date)
    .replace('.', '');
  return `${datePart} (${weekday})`;
}

function getMessageDayKey(isoString) {
  const date = new Date(isoString);
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function ensureMessageTimestamps(messages, conversation) {
  const fallback = conversation?.createdAt || new Date().toISOString();
  let changed = false;

  const next = messages.map((msg) => {
    if (msg.createdAt) return msg;
    changed = true;
    return { ...msg, createdAt: fallback };
  });

  return { messages: next, changed };
}

function createChatDateRowElement(isoString) {
  const row = document.createElement('div');
  row.className = 'chat-date-row';

  const lineStart = document.createElement('span');
  lineStart.className = 'chat-date-line';
  lineStart.setAttribute('aria-hidden', 'true');

  const time = document.createElement('time');
  time.className = 'chat-date';
  time.dateTime = getMessageDayKey(isoString);
  time.textContent = formatChatDateLabel(isoString);

  const lineEnd = document.createElement('span');
  lineEnd.className = 'chat-date-line';
  lineEnd.setAttribute('aria-hidden', 'true');

  row.append(lineStart, time, lineEnd);
  return row;
}

function insertChatDateRowIfNeeded(isoString, insertBeforeNode) {
  if (!chatArea || !isoString) return;

  const dayKey = getMessageDayKey(isoString);
  const existingRows = chatArea.querySelectorAll('.chat-date-row time');
  const alreadyShown = Array.from(existingRows).some((el) => el.dateTime === dayKey);
  if (alreadyShown) return;

  const row = createChatDateRowElement(isoString);
  if (insertBeforeNode) {
    chatArea.insertBefore(row, insertBeforeNode);
  } else {
    chatArea.appendChild(row);
  }
}

function clearChatTranscript() {
  if (!chatArea) return;
  chatArea.querySelectorAll('.chat-date-row').forEach((node) => node.remove());
  chatArea.querySelectorAll('.message:not(.recognizing)').forEach((node) => node.remove());
}

function loadChatHistory() {
  const conversation = getConversation(conversationId);
  if (!conversation) {
    window.location.replace('index.html');
    return [];
  }
  return structuredClone(conversation.messages);
}

function getChatDisplayTitle() {
  const conversation = getConversation(conversationId);
  const title = conversation?.title?.trim();
  return title || '새 대화';
}

function updateChatTitle() {
  const title = getChatDisplayTitle();
  if (chatTitleEl) {
    chatTitleEl.textContent = title;
  }
  document.title = `${title} — 이중언어 학습 시스템`;
}

function saveChatHistory() {
  updateConversationMessages(conversationId, chatMessages);
  updateChatTitle();
}

function renderChatHistory() {
  clearChatTranscript();

  let lastDayKey = null;
  chatMessages.forEach((record) => {
    const createdAt = record.createdAt || new Date().toISOString();
    const dayKey = getMessageDayKey(createdAt);

    if (dayKey !== lastDayKey) {
      chatArea.insertBefore(createChatDateRowElement(createdAt), recognizingBox);
      lastDayKey = dayKey;
    }

    chatArea.insertBefore(createMessageElement(record), recognizingBox);
  });

  scrollChatToBottom();
}

function createMessageId() {
  return createId('msg');
}

function appendChatMessage(record) {
  if (!chatArea) return;

  const recordWithTime = {
    ...record,
    createdAt: record.createdAt || new Date().toISOString(),
  };
  const prevMessage = chatMessages[chatMessages.length - 1];
  const prevDayKey = prevMessage?.createdAt
    ? getMessageDayKey(prevMessage.createdAt)
    : null;
  const nextDayKey = getMessageDayKey(recordWithTime.createdAt);

  chatMessages.push(recordWithTime);
  saveChatHistory();

  const insertBeforeNode =
    recognizingBox && recognizingBox.parentElement === chatArea
      ? recognizingBox
      : null;

  if (nextDayKey !== prevDayKey) {
    insertChatDateRowIfNeeded(recordWithTime.createdAt, insertBeforeNode);
  }

  const element = createMessageElement(recordWithTime);
  if (insertBeforeNode) {
    chatArea.insertBefore(element, insertBeforeNode);
  } else {
    chatArea.appendChild(element);
  }
  scrollChatToBottom();
}

/**
 * @param {MessageSegment[]} segments
 * @returns {HTMLElement}
 */
function createUserMessageElement(segments) {
  const message = document.createElement('section');
  message.className = 'message user';
  const p = document.createElement('p');
  p.innerHTML = renderUserMessageHtml(segments);
  message.appendChild(p);
  return message;
}

/** @param {MessageSegment[]} segments */
function addUserMessage(segments) {
  const normalized = normalizeUserSegments(segments);

  normalized.forEach((seg) => {
    if (seg.type === 'chip') {
      addWordToWordbook(seg.word, seg.hint || '');
    }
  });

  appendChatMessage({
    id: createMessageId(),
    role: 'user',
    segments: normalized,
  });
}

function addAssistantMessage(text) {
  appendChatMessage({
    id: createMessageId(),
    role: 'assistant',
    text,
  });
}

/** @deprecated assistant 전용 — 사용자 말풍선은 addUserMessage 사용 */
function addMessage(text, type) {
  if (type === 'user') {
    addUserMessage(segmentsFromModelResult(text));
    return;
  }
  addAssistantMessage(text);
}

function initChatHistory() {
  const conversation = getConversation(conversationId);
  const loaded = loadChatHistory();
  const { messages, changed } = ensureMessageTimestamps(loaded, conversation);

  chatMessages = messages.map((record) => {
    if (record.role !== 'user') return record;
    const segments = resolveUserMessageSegments(record);
    return { ...record, segments };
  });

  if (changed) {
    saveChatHistory();
  }

  renderChatHistory();
}

if (conversationId) {
  initChatHistory();
  updateChatTitle();
}

if (
  conversationId &&
  !USE_MOCK_VOICE_SCENARIO &&
  ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)
) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = 'ko-KR';
  recognition.interimResults = true;
  recognition.continuous = false;

  recognition.onresult = (event) => {
    const text = Array.from(event.results)
      .map((result) => result[0].transcript)
      .join('');
    statusText.textContent = `인식 중: ${text}`;

    if (event.results[0].isFinal) {
      // TODO: const modelResult = await yourAsrApi(audio);
      // addUserMessage(segmentsFromModelResult(modelResult));
      addUserMessage(segmentsFromModelResult(text));
      setTimeout(
        () => addAssistantMessage('좋아요! 해당 표현을 러시아어 단어와 함께 정리해볼게요.'),
        450
      );
    }
  };

  recognition.onend = stopRecording;
} else if (conversationId && statusText && !USE_MOCK_VOICE_SCENARIO) {
  statusText.textContent = '이 브라우저는 음성 인식을 지원하지 않아요.';
}

if (conversationId) {
  micBtn.addEventListener('click', () => {
    isRecording ? stopRecording() : startRecording();
  });
}

function setMicIcon(name) {
  micBtn.innerHTML = `<i data-lucide="${name}" class="mic-icon"></i>`;
  if (window.lucide) lucide.createIcons();
}

function formatRecordTime(ms) {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

function startRecordTimer() {
  if (!recordTime) return;
  recordStartTime = Date.now();
  recordTime.textContent = '00:00';
  recordTimerInterval = setInterval(() => {
    recordTime.textContent = formatRecordTime(Date.now() - recordStartTime);
  }, 200);
}

function stopRecordTimer() {
  clearInterval(recordTimerInterval);
  recordTimerInterval = null;
}

function showRecordPopup() {
  if (recordPopup) recordPopup.hidden = false;
  if (bottombar) bottombar.classList.add('recording-active');
}

function hideRecordPopup() {
  if (recordPopup) recordPopup.hidden = true;
  if (bottombar) bottombar.classList.remove('recording-active');
}

function clearMockScenarioTimer() {
  if (mockScenarioTimer) {
    clearTimeout(mockScenarioTimer);
    mockScenarioTimer = null;
  }
}

function finishMockRecognition() {
  clearMockScenarioTimer();
  if (!isRecording) return;

  addUserMessage(segmentsFromMixedUtterance(MOCK_USER_UTTERANCE));
  stopRecording();
}

function scheduleMockRecognition() {
  clearMockScenarioTimer();
  mockScenarioTimer = setTimeout(finishMockRecognition, MOCK_RECOGNITION_DELAY_MS);
}

function stopAudioVisualizer() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
  analyser = null;
  if (audioBars) {
    audioBars.classList.remove('fallback-animate');
    audioBars.querySelectorAll('span').forEach((bar) => {
      bar.style.height = '8px';
    });
  }
}

async function startAudioVisualizer() {
  if (!audioBars || !navigator.mediaDevices?.getUserMedia) {
    audioBars?.classList.add('fallback-animate');
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(micStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 128;
    analyser.smoothingTimeConstant = 0.75;
    source.connect(analyser);

    const bars = audioBars.querySelectorAll('span');
    const bufferLength = analyser.frequencyBinCount;
    const data = new Uint8Array(bufferLength);

    const draw = () => {
      analyser.getByteFrequencyData(data);
      bars.forEach((bar, index) => {
        const dataIndex = Math.floor((index / bars.length) * bufferLength);
        const value = data[dataIndex] / 255;
        const height = 8 + value * 30;
        bar.style.height = `${height}px`;
      });
      animationFrameId = requestAnimationFrame(draw);
    };

    audioBars.classList.remove('fallback-animate');
    draw();
  } catch {
    audioBars.classList.add('fallback-animate');
  }
}

// 녹음 파일을 API로 보내기
async function sendAudioToAsrApi(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, `recording_${Date.now()}.webm`);

  const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`ASR API error: ${response.status}`);
  }

  return response.json();
}

async function startRecording() {
  isRecording = true;
  document.body.classList.add('recording');
  micBtn.classList.add('is-recording');
  micBtn.setAttribute('aria-label', '녹음 중');
  setMicIcon('audio-lines');
  micText.textContent = '녹음 중...';
  statusText.textContent = '음성 녹음 중';
  showRecordPopup();
  startRecordTimer();

  if (USE_MOCK_VOICE_SCENARIO) {
    scheduleMockRecognition();
    await startAudioVisualizer();
    return;
  }

  try {
    recordedChunks = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(recordedChunks, { type: 'audio/webm' });

      statusText.textContent = '모델 추론 중...';

      try {
        const result = await sendAudioToAsrApi(audioBlob);

        addUserMessage(segmentsFromModelResult(result.transcription));

        if (result.feedback) {
          addAssistantMessage(result.feedback);
        }

        statusText.textContent = '음성 인식 대기 중';
      } catch (error) {
        console.error(error);
        statusText.textContent = '음성 인식 실패';
        addAssistantMessage('음성 인식 처리 중 오류가 발생했습니다.');
      } finally {
        stream.getTracks().forEach((track) => track.stop());
      }
    };

    await startAudioVisualizer();
    mediaRecorder.start();
  } catch (error) {
    console.error(error);
    statusText.textContent = '마이크 접근 실패';
    stopRecording();
  }
}

function stopRecording() {
  clearMockScenarioTimer();
  isRecording = false;
  document.body.classList.remove('recording');
  micBtn.classList.remove('is-recording');
  micBtn.setAttribute('aria-label', '녹음 시작');
  setMicIcon('mic');
  micText.textContent = '녹음 시작';
  hideRecordPopup();
  stopRecordTimer();
  stopAudioVisualizer();

  if (!USE_MOCK_VOICE_SCENARIO && mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    return;
  }

  if (recognition && recognition.stop) {
    recognition.stop();
  }

  if (statusText.textContent === '음성 녹음 중' || statusText.textContent === '음성 인식 중') {
    statusText.textContent = '음성 인식 대기 중';
  }
}

if (conversationId) {
  collapseBtn?.addEventListener('click', () => {
    wordCard.classList.toggle('collapsed');
    collapseBtn.textContent = wordCard.classList.contains('collapsed') ? '⌃' : '⌄';
  });

  closeBtn?.addEventListener('click', () => {
    wordCard.style.display = 'none';
  });

  document.querySelectorAll('.sound').forEach((btn) => {
    btn.addEventListener('click', () => {
      const word = btn.parentElement.querySelector('.ru').textContent;
      const utter = new SpeechSynthesisUtterance(word);
      utter.lang = 'ru-RU';
      speechSynthesis.speak(utter);
    });
  });

  if (window.lucide) {
    lucide.createIcons();
  }
}
