(() => {
  const key = 'djepa-tour-guidance-seen';
  const hints = [...document.querySelectorAll('[data-tour-guide]')];
  let seen = false;
  try { seen = sessionStorage.getItem(key) === 'yes'; } catch (_) {}
  const hide = () => {
    hints.forEach(hint => { hint.hidden = true; });
    document.querySelectorAll('.guide-target').forEach(target => target.classList.remove('guide-target'));
  };
  const finish = () => {
    hide();
    try { sessionStorage.setItem(key, 'yes'); } catch (_) {}
  };
  if (!seen) hints.forEach(hint => {
    hint.hidden = false;
    document.getElementById(hint.dataset.guideTarget)?.classList.add('guide-target');
  });
  hints.forEach(hint => hint.querySelector('[data-guide-dismiss]')?.addEventListener('click', hide));
  // Capture before asynchronous audio loading: dismiss even if autoplay fails.
  for (const id of ['play', 'narration-play']) {
    document.getElementById(id)?.addEventListener('click', finish, {capture: true});
  }
  // Space and operation-specific playback also start the tour.
  const observer = new MutationObserver(() => {
    if (document.body.classList.contains('running') || document.body.classList.contains('narrating')) finish();
  });
  observer.observe(document.body, {attributes: true, attributeFilter: ['class']});
  document.getElementById('interactive-explainer')?.addEventListener('click', hide);
})();
