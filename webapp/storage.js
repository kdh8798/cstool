const CONVERSATIONS_STORAGE_KEY = 'bilingual-conversations-v1';
const LEGACY_HISTORY_KEY = 'bilingual-chat-history';
const DEFAULT_CONVERSATION_TITLE = '새 대화';
const WORDBOOK_STORAGE_KEY = 'bilingual-wordbook-v1';

const DEMO_USER_SEGMENTS = [
  { type: 'text', value: '안녕하세요, ' },
  { type: 'chip', word: 'я', hint: '나는' },
  { type: 'text', value: ' ' },
  { type: 'chip', word: 'хочу', hint: '원하다' },
  { type: 'text', value: ' 배우고 싶어요' },
];

const WELCOME_MESSAGE = {
  id: 'msg-welcome',
  role: 'assistant',
  text: '안녕하세요! 무엇을 배우고 싶으신가요?',
};

const DEMO_CONVERSATION_MESSAGES = [
  WELCOME_MESSAGE,
  {
    id: 'msg-2',
    role: 'user',
    segments: DEMO_USER_SEGMENTS,
  },
  {
    id: 'msg-3',
    role: 'assistant',
    text: '좋아요! 해당 표현을 러시아어 단어와 함께 정리해볼게요.',
  },
];

function createId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function normalizeConversationTitles(store) {
  let changed = false;

  store.conversations.forEach((conversation) => {
    if (!conversation.customTitle) {
      if (conversation.title !== DEFAULT_CONVERSATION_TITLE) {
        conversation.title = DEFAULT_CONVERSATION_TITLE;
        changed = true;
      }
    }
  });

  return changed;
}

function loadStore() {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.conversations)) {
        if (normalizeConversationTitles(parsed)) {
          saveStore(parsed);
        }
        return parsed;
      }
    }
  } catch {
    /* ignore */
  }

  const store = { conversations: [] };
  migrateLegacyHistory(store);
  if (store.conversations.length === 0) {
    store.conversations.push(createConversationRecord(DEMO_CONVERSATION_MESSAGES));
  }
  saveStore(store);
  return store;
}

function saveStore(store) {
  localStorage.setItem(CONVERSATIONS_STORAGE_KEY, JSON.stringify(store));
}

function migrateLegacyHistory(store) {
  try {
    const raw = localStorage.getItem(LEGACY_HISTORY_KEY);
    if (!raw) return;

    const messages = JSON.parse(raw);
    if (!Array.isArray(messages) || messages.length === 0) return;

    store.conversations.push(createConversationRecord(messages));
    localStorage.removeItem(LEGACY_HISTORY_KEY);
  } catch {
    /* ignore */
  }
}

function messageToPlainText(message) {
  if (message.role === 'assistant') {
    return message.text || '';
  }

  if (!Array.isArray(message.segments)) return '';

  return message.segments
    .map((seg) => {
      if (seg.type === 'chip') {
        const hint = seg.hint ? ` (${seg.hint})` : '';
        return `${seg.word}${hint}`;
      }
      return seg.value || '';
    })
    .join('');
}

function deriveTitle() {
  return DEFAULT_CONVERSATION_TITLE;
}

function derivePreview(messages) {
  if (!messages.length) return '대화를 시작해 보세요';

  const last = messages[messages.length - 1];
  const plain = messageToPlainText(last).trim();
  if (!plain) return '대화를 시작해 보세요';
  return plain.length > 48 ? `${plain.slice(0, 48)}…` : plain;
}

function createConversationRecord(messages) {
  const now = new Date().toISOString();
  const record = {
    id: createId('conv'),
    messages: structuredClone(messages),
    createdAt: now,
    updatedAt: now,
  };
  record.title = deriveTitle();
  record.preview = derivePreview(record.messages);
  return record;
}

function refreshConversationMeta(conversation) {
  if (!conversation.customTitle) {
    conversation.title = deriveTitle();
  }
  conversation.preview = derivePreview(conversation.messages);
  conversation.updatedAt = new Date().toISOString();
}

function renameConversation(conversationId, title) {
  const store = loadStore();
  const conversation = store.conversations.find((c) => c.id === conversationId);
  if (!conversation) return null;

  const trimmed = title.trim();
  if (trimmed) {
    conversation.title = trimmed.length > 40 ? `${trimmed.slice(0, 40)}…` : trimmed;
    conversation.customTitle = true;
  } else {
    conversation.customTitle = false;
    conversation.title = deriveTitle();
  }

  conversation.updatedAt = new Date().toISOString();
  saveStore(store);
  return conversation;
}

function getAllConversations() {
  const store = loadStore();
  return [...store.conversations].sort(
    (a, b) => new Date(b.updatedAt) - new Date(a.updatedAt)
  );
}

function getConversation(conversationId) {
  const store = loadStore();
  return store.conversations.find((c) => c.id === conversationId) || null;
}

function createConversation(initialMessages) {
  const store = loadStore();
  const messages = initialMessages
    ? structuredClone(initialMessages)
    : [structuredClone(WELCOME_MESSAGE)];
  const conversation = createConversationRecord(messages);
  store.conversations.push(conversation);
  saveStore(store);
  return conversation;
}

function updateConversationMessages(conversationId, messages) {
  const store = loadStore();
  const conversation = store.conversations.find((c) => c.id === conversationId);
  if (!conversation) return null;

  conversation.messages = structuredClone(messages);
  refreshConversationMeta(conversation);
  saveStore(store);
  return conversation;
}

function deleteConversation(conversationId) {
  const store = loadStore();
  const index = store.conversations.findIndex((c) => c.id === conversationId);
  if (index === -1) return false;

  store.conversations.splice(index, 1);
  saveStore(store);
  return true;
}

function formatConversationDate(isoString) {
  const date = new Date(isoString);
  const now = new Date();
  const isToday =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();

  if (isToday) {
    return new Intl.DateTimeFormat('ko-KR', {
      hour: 'numeric',
      minute: '2-digit',
    }).format(date);
  }

  return new Intl.DateTimeFormat('ko-KR', {
    month: 'long',
    day: 'numeric',
  }).format(date);
}

function loadWordbook() {
  try {
    const raw = localStorage.getItem(WORDBOOK_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveWordbook(words) {
  localStorage.setItem(WORDBOOK_STORAGE_KEY, JSON.stringify(words));
}

function addWordToWordbook(word, meaning = '') {
  const normalized = String(word || '').trim();
  if (!normalized) return;

  const words = loadWordbook();
  const exists = words.some((item) => item.word === normalized);

  if (exists) return;

  words.push({
    id: createId('word'),
    word: normalized,
    meaning,
    createdAt: new Date().toISOString(),
  });

  saveWordbook(words);
}

function deleteWordFromWordbook(wordId) {
  const words = loadWordbook().filter((item) => item.id !== wordId);
  saveWordbook(words);
}
