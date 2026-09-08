/* No analytics or external runtime dependencies. */
const themeButton = document.querySelector('#theme-toggle');
const systemTheme = matchMedia('(prefers-color-scheme: dark)');
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeButton.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
  themeButton.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
  document.querySelector('#site-icon').href = `static/images/branding/jepa-icon-${theme}.svg`;
}
setTheme(document.documentElement.dataset.theme || 'light');
themeButton.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  setTheme(next);
  try { localStorage.setItem('djepa-theme', next); } catch (_) { /* Session-only preference. */ }
});
systemTheme.addEventListener('change', event => {
  try { if (localStorage.getItem('djepa-theme')) return; } catch (_) { /* Follow system. */ }
  setTheme(event.matches ? 'dark' : 'light');
});

function selectDemo(task) {
  document.querySelectorAll('[data-demo]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.demo === task));
  });
  document.querySelectorAll('.demo-panel').forEach(panel => {
    panel.hidden = panel.id !== `demo-${task}`;
    if (panel.hidden) panel.querySelector('video').pause();
  });
}
document.querySelectorAll('[data-demo]').forEach(button => {
  button.addEventListener('click', () => selectDemo(button.dataset.demo));
});
selectDemo('pusht');

document.querySelector('#copy-command').addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#reproduce-command').textContent);
    status.textContent = 'Copied.';
  } catch (_) { status.textContent = 'Select the command text to copy it manually.'; }
});

// Author-requested artwork slots remain visible until actual files are supplied.
fetch('site-content.json').then(response => {
  if (!response.ok) throw new Error('Content unavailable');
  return response.json();
}).then(content => {
  for (const [name, figure] of Object.entries(content.figures)) {
    if (!figure.src) continue;
    const slot = document.querySelector(`[data-figure="${name}"]`);
    if (!slot) continue;
    const image = new Image();
    image.alt = figure.alt;
    image.onload = () => {
      slot.replaceChildren(image);
      slot.classList.add('artwork-ready');
      slot.removeAttribute('role');
      slot.removeAttribute('aria-label');
    };
    image.src = figure.src;
  }
}).catch(() => { /* Static content, links and explicit artwork placeholders still work. */ });
