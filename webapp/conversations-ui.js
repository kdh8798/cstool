function openChat(conversationId) {
  window.location.href = `/chat.html?id=${encodeURIComponent(conversationId)}`;
}

function handleDeleteConversation(conversationId, event, onDeleted) {
  event.preventDefault();
  event.stopPropagation();

  if (!confirm('이 대화를 삭제할까요?')) return;

  deleteConversation(conversationId);
  onDeleted?.();
}

function startConversationRename(conv, titleRow, onRenamed) {
  const titleEl = titleRow.querySelector('.conversation-item__title');
  const renameBtn = titleRow.querySelector('.conversation-rename');
  if (!titleEl || titleRow.querySelector('.conversation-item__title-input')) return;

  const previousTitle = conv.title;
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'conversation-item__title-input';
  input.value = previousTitle;
  input.maxLength = 40;
  input.setAttribute('aria-label', '대화 제목 수정');

  titleEl.style.visibility = 'hidden';
  if (renameBtn) renameBtn.style.visibility = 'hidden';
  titleRow.appendChild(input);
  input.focus();
  input.select();

  let finished = false;
  const finish = (commit) => {
    if (finished) return;
    finished = true;

    const nextTitle = commit ? input.value : previousTitle;
    input.remove();

    if (commit) {
      const updated = renameConversation(conv.id, nextTitle);
      if (updated) {
        conv.title = updated.title;
        conv.customTitle = updated.customTitle;
      }
      titleEl.textContent = conv.title;
      onRenamed?.();
    } else {
      titleEl.textContent = previousTitle;
    }

    titleEl.style.visibility = '';
    if (renameBtn) renameBtn.style.visibility = '';
  };

  input.addEventListener('click', (event) => event.stopPropagation());
  input.addEventListener('keydown', (event) => {
    event.stopPropagation();
    if (event.key === 'Enter') {
      event.preventDefault();
      finish(true);
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      finish(false);
    }
  });
  input.addEventListener('blur', () => finish(true));
}

let activeConversationMenu = null;
let activeMenuCard = null;
let activeMenuMoreBtn = null;
const MENU_GAP = 2;

function getMenuScrollRoot() {
  return (
    document.querySelector('.home-main--scroll') ||
    document.querySelector('.home-page .home-main')
  );
}

function positionConversationMenu(menu, moreBtn) {
  const rect = moreBtn.getBoundingClientRect();
  const iconCenterY = rect.top + rect.height / 2;
  const placeBelow = iconCenterY < window.innerHeight / 2;

  menu.classList.add('conversation-menu--fixed');
  menu.classList.add(
    placeBelow ? 'conversation-menu--below' : 'conversation-menu--above'
  );

  document.body.appendChild(menu);

  const menuRect = menu.getBoundingClientRect();
  let top = placeBelow ? rect.bottom + MENU_GAP : rect.top - MENU_GAP - menuRect.height;
  let left = rect.right - menuRect.width;

  const edge = 8;
  left = Math.max(edge, Math.min(left, window.innerWidth - menuRect.width - edge));
  top = Math.max(edge, Math.min(top, window.innerHeight - menuRect.height - edge));

  menu.style.top = `${top}px`;
  menu.style.left = `${left}px`;
}

function closeConversationMenu() {
  if (activeConversationMenu) {
    activeMenuCard?.classList.remove('conversation-card--menu-open');
    activeMenuCard
      ?.closest('.conversation-list__item')
      ?.classList.remove('conversation-list__item--menu-open');
    activeMenuMoreBtn?.setAttribute('aria-expanded', 'false');
    activeConversationMenu.remove();
    activeConversationMenu = null;
    activeMenuCard = null;
    activeMenuMoreBtn = null;
  }

  window.removeEventListener('scroll', closeConversationMenu, true);
  getMenuScrollRoot()?.removeEventListener('scroll', closeConversationMenu);
}

document.addEventListener('click', (event) => {
  if (
    event.target.closest('.conversation-menu') ||
    event.target.closest('.conversation-more')
  ) {
    return;
  }
  closeConversationMenu();
});

function openConversationMenu(conv, card, titleRow, onUpdated, moreBtn) {
  closeConversationMenu();

  const menu = document.createElement('div');
  menu.className = 'conversation-menu';
  menu.dataset.convId = conv.id;
  menu.setAttribute('role', 'menu');
  menu.innerHTML = `
    <button type="button" class="conversation-menu__item" data-action="rename">
      <i data-lucide="pencil" class="conversation-menu__icon" aria-hidden="true"></i>
      <span>이름 바꾸기</span>
    </button>
    <button type="button" class="conversation-menu__item conversation-menu__item--danger" data-action="delete">
      <i data-lucide="trash-2" class="conversation-menu__icon" aria-hidden="true"></i>
      <span>삭제</span>
    </button>
  `;

  menu.addEventListener('click', (event) => event.stopPropagation());

  menu.querySelector('[data-action="rename"]')?.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    closeConversationMenu();
    startConversationRename(conv, titleRow, onUpdated);
  });

  menu.querySelector('[data-action="delete"]')?.addEventListener('click', (event) => {
    handleDeleteConversation(conv.id, event, onUpdated);
    closeConversationMenu();
  });

  if (moreBtn) {
    positionConversationMenu(menu, moreBtn);
  } else {
    menu.classList.add('conversation-menu--fixed', 'conversation-menu--below');
    document.body.appendChild(menu);
  }

  if (window.lucide) {
    lucide.createIcons();
  }

  card.classList.add('conversation-card--menu-open');
  card.closest('.conversation-list__item')?.classList.add('conversation-list__item--menu-open');

  activeConversationMenu = menu;
  activeMenuCard = card;
  activeMenuMoreBtn = moreBtn || null;
  moreBtn?.setAttribute('aria-expanded', 'true');

  window.addEventListener('scroll', closeConversationMenu, true);
  getMenuScrollRoot()?.addEventListener('scroll', closeConversationMenu, { passive: true });
}

function attachMoreButton(conv, card, titleRow, onUpdated) {
  const anchor = document.createElement('div');
  anchor.className = 'conversation-more-anchor';

  const moreBtn = document.createElement('button');
  moreBtn.type = 'button';
  moreBtn.className = 'conversation-more';
  moreBtn.setAttribute('aria-label', '더보기');
  moreBtn.setAttribute('aria-haspopup', 'menu');
  moreBtn.setAttribute('aria-expanded', 'false');
  moreBtn.innerHTML = '<i data-lucide="ellipsis" aria-hidden="true"></i>';
  moreBtn.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (activeConversationMenu?.dataset.convId === conv.id) {
      closeConversationMenu();
      return;
    }
    openConversationMenu(conv, card, titleRow, onUpdated, moreBtn);
  });
  anchor.appendChild(moreBtn);
  card.appendChild(anchor);
}

function attachRenameButton(conv, titleRow, onRenamed) {
  const renameBtn = document.createElement('button');
  renameBtn.type = 'button';
  renameBtn.className = 'conversation-rename';
  renameBtn.setAttribute('aria-label', '제목 수정');
  renameBtn.innerHTML = '<i data-lucide="pencil" aria-hidden="true"></i>';
  renameBtn.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    startConversationRename(conv, titleRow, onRenamed);
  });
  titleRow.appendChild(renameBtn);
}

function createConversationListItem(conv, onDeleted, options = {}) {
  const {
    showDelete = false,
    showRename = false,
    showMoreMenu = false,
    showArrow = !(showDelete || showMoreMenu),
  } = options;
  const item = document.createElement('li');
  item.className = 'conversation-list__item';

  const card = document.createElement('div');
  card.className =
    showDelete || showMoreMenu
      ? 'conversation-card'
      : 'conversation-card conversation-card--no-delete';

  const openBtn = document.createElement('button');
  openBtn.type = 'button';
  openBtn.className = 'conversation-item';
  const titleRow = document.createElement('span');
  titleRow.className = 'conversation-item__title-row';

  const titleEl = document.createElement('strong');
  titleEl.className = 'conversation-item__title';
  titleEl.textContent = conv.title;
  titleRow.appendChild(titleEl);
  if (showRename) {
    attachRenameButton(conv, titleRow, onDeleted);
  }

  const previewEl = document.createElement('span');
  previewEl.className = 'conversation-item__preview';
  previewEl.textContent = conv.preview;

  const main = document.createElement('span');
  main.className = 'conversation-item__main';
  main.append(titleRow, previewEl);

  openBtn.appendChild(main);

  if (showArrow) {
    const arrow = document.createElement('span');
    arrow.className = 'conversation-item__arrow';
    arrow.setAttribute('aria-hidden', 'true');
    arrow.innerHTML = '<i data-lucide="arrow-up-right"></i>';
    openBtn.appendChild(arrow);
  }
  openBtn.addEventListener('click', () => openChat(conv.id));

  card.appendChild(openBtn);

  if (showMoreMenu) {
    attachMoreButton(conv, card, titleRow, onDeleted);
  } else if (showDelete) {
    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'conversation-delete';
    deleteBtn.setAttribute('aria-label', '대화 삭제');
    deleteBtn.innerHTML = '<i data-lucide="trash-2"></i>';
    deleteBtn.addEventListener('click', (event) =>
      handleDeleteConversation(conv.id, event, onDeleted)
    );
    card.appendChild(deleteBtn);
  }
  item.appendChild(card);
  return item;
}

function renderConversationList(container, conversations, emptyEl, onDeleted, options = {}) {
  if (!container) return;

  container.innerHTML = '';

  if (!conversations.length) {
    if (emptyEl) emptyEl.hidden = false;
    return;
  }

  if (emptyEl) emptyEl.hidden = true;

  conversations.forEach((conv) => {
    container.appendChild(createConversationListItem(conv, onDeleted, options));
  });

  if (window.lucide) {
    lucide.createIcons();
  }

  window.applySquircleFallback?.(container);
}
