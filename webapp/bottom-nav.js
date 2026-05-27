(function initBottomNav() {
  if (!document.body.classList.contains('tab-shell')) return;

  const app = document.querySelector('.app');
  if (!app || app.querySelector('.bottom-nav')) return;

  const active = document.body.dataset.nav || '';
  const items = [
    { nav: 'home', href: '/index.html', icon: 'house', label: '홈' },
    { nav: 'history', href: '/history.html', icon: 'clock-3', label: '기록' },
    { nav: 'settings', href: '/settings.html', icon: 'settings', label: '설정' },
  ];

  const nav = document.createElement('nav');
  nav.className = 'bottom-nav';
  nav.setAttribute('aria-label', '주요 메뉴');
  nav.innerHTML = items
    .map(({ nav: id, href, icon, label }) => {
      const isActive = id === active;
      return `<a href="${href}" class="bottom-nav__item${isActive ? ' is-active' : ''}" data-nav="${id}"${
        isActive ? ' aria-current="page"' : ''
      }><i data-lucide="${icon}" class="bottom-nav__icon" aria-hidden="true"></i><span class="bottom-nav__label">${label}</span></a>`;
    })
    .join('');

  app.appendChild(nav);

  if (window.lucide) {
    lucide.createIcons();
  }
})();
