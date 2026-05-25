const conversationList = document.querySelector('#conversationList');
const emptyHistory = document.querySelector('#emptyHistory');

function renderAllConversations() {
  renderConversationList(
    conversationList,
    getAllConversations(),
    emptyHistory,
    renderAllConversations,
    { showMoreMenu: true, showArrow: false }
  );
}

renderAllConversations();

if (window.lucide) {
  lucide.createIcons();
}
