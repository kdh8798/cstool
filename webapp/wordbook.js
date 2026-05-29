const wordbookList = document.querySelector('#wordbookList');
const wordbookCount = document.querySelector('.wordbook-panel__count');

function speakRussian(word) {
  const utter = new SpeechSynthesisUtterance(word);
  utter.lang = 'ru-RU';
  speechSynthesis.speak(utter);
}

function formatConfidence(confidence) {
  if (confidence === null || confidence === undefined || confidence === '') {
    return '';
  }

  const percent = Math.round(Number(confidence) * 100);
  if (Number.isNaN(percent)) return '';

  return ` · 신뢰도 ${percent}%`;
}

function renderWordbook() {
  const words = loadWordbook();

  if (wordbookCount) {
    wordbookCount.textContent = words.length;
  }

  if (!wordbookList) return;

  if (!words.length) {
    wordbookList.innerHTML = `
      <article class="word-item">
        <h3>저장된 단어가 없어요</h3>
        <small>대화에서 러시아어 표현이 감지되면 자동으로 저장됩니다.</small>
      </article>
    `;
    return;
  }

  wordbookList.innerHTML = words
    .map((item) => `
      <article class="word-item" data-word-id="${item.id}">
        <button type="button" class="sound" aria-label="발음 듣기">🔊</button>
        <button type="button" class="word-delete" aria-label="단어 삭제">×</button>
        <h3>${item.meaning || '뜻 미등록'}</h3>
        <p class="ru">${item.word}</p>
        <small>${new Date(item.createdAt).toLocaleDateString('ko-KR')}${formatConfidence(item.confidence)}</small>
      </article>
    `)
    .join('');

  wordbookList.querySelectorAll('.sound').forEach((btn) => {
    btn.addEventListener('click', () => {
      const word = btn.parentElement.querySelector('.ru')?.textContent?.trim();
      if (word) speakRussian(word);
    });
  });

  wordbookList.querySelectorAll('.word-delete').forEach((btn) => {
    btn.addEventListener('click', () => {
      const item = btn.closest('.word-item');
      deleteWordFromWordbook(item.dataset.wordId);
      renderWordbook();
    });
  });
}

renderWordbook();

if (window.lucide) {
  lucide.createIcons();
}
