const HOME_PREVIEW_LIMIT_MOBILE = 2;
const HOME_PREVIEW_LIMIT_DESKTOP = 3;
const DESKTOP_MEDIA = window.matchMedia('(min-width: 761px)');

const conversationList = document.querySelector('#conversationList');
const emptyHistory = document.querySelector('#emptyHistory');
const newChatBtn = document.querySelector('#newChatBtn');

function getHomePreviewLimit() {
  return DESKTOP_MEDIA.matches ? HOME_PREVIEW_LIMIT_DESKTOP : HOME_PREVIEW_LIMIT_MOBILE;
}

function renderHomeConversations() {
  const conversations = getAllConversations();
  const preview = conversations.slice(0, getHomePreviewLimit());

  renderConversationList(conversationList, preview, emptyHistory, renderHomeConversations, {
    showDelete: false,
  });
}

newChatBtn?.addEventListener('click', () => {
  const conversation = createConversation();
  openChat(conversation.id);
});

renderHomeConversations();
DESKTOP_MEDIA.addEventListener('change', renderHomeConversations);

function initHomeIcons() {
  if (!window.lucide) return;
  lucide.createIcons();
  applyBrandIconGradient();
  requestAnimationFrame(applyBrandIconGradient);
}

initHomeIcons();

function scheduleHomeSquircle() {
  window.applySquircleFallback?.();
  requestAnimationFrame(() => {
    window.applySquircleFallback?.();
    requestAnimationFrame(() => window.applySquircleFallback?.());
  });
}

scheduleHomeSquircle();
window.addEventListener('load', scheduleHomeSquircle, { once: true });
