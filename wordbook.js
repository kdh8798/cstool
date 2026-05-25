document.querySelectorAll('.sound').forEach((btn) => {
  btn.addEventListener('click', () => {
    const word = btn.parentElement.querySelector('.ru')?.textContent?.trim();
    if (!word) return;
    const utter = new SpeechSynthesisUtterance(word);
    utter.lang = 'ru-RU';
    speechSynthesis.speak(utter);
  });
});

if (window.lucide) {
  lucide.createIcons();
}
