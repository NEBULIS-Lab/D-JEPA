/* No analytics or external runtime dependencies. */
const themeButton = document.querySelector('#theme-toggle');
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeButton.querySelector('.theme-label').textContent = theme === 'dark' ? 'Dark' : 'Light';
  themeButton.setAttribute('aria-pressed', String(theme === 'dark'));
  themeButton.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
  document.querySelector('#site-icon').href = `static/images/branding/jepa-icon-${theme}.svg`;
}
setTheme(document.documentElement.dataset.theme || 'dark');
themeButton.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  setTheme(next);
  try { localStorage.setItem('djepa-theme', next); } catch (_) { /* Session-only preference. */ }
});

// Task tabs progressively enhance the gallery; all clips remain available without JS.
const demoTabs = [...document.querySelectorAll('.demo-tabs [role="tab"]')];
function selectDemo(tab, focus = false) {
  demoTabs.forEach(item => {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    const panel = document.getElementById(item.getAttribute('aria-controls'));
    panel.classList.toggle('is-active', selected);
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', item.id);
    if (!selected) panel.querySelector('video').pause();
  });
  if (focus) tab.focus();
}
if (demoTabs.length) {
  selectDemo(demoTabs[0]);
  document.documentElement.classList.add('tabs-enabled');
  demoTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectDemo(tab));
    tab.addEventListener('keydown', event => {
      const keys = { ArrowRight: (index + 1) % demoTabs.length, ArrowLeft: (index - 1 + demoTabs.length) % demoTabs.length, Home: 0, End: demoTabs.length - 1 };
      if (event.key in keys) { event.preventDefault(); selectDemo(demoTabs[keys[event.key]], true); }
    });
  });
}

// Playback starts only when requested, and only one clip plays at a time.
document.querySelectorAll('video').forEach(video => {
  const stage = document.createElement('div');
  stage.className = 'video-stage';
  video.before(stage);
  stage.append(video);
  const playButton = document.createElement('button');
  playButton.type = 'button';
  playButton.className = 'video-play';
  playButton.setAttribute('aria-label', 'Play ' + (video.getAttribute('aria-label') || 'video'));
  playButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3.5v17l14-8.5Z"/></svg><span>Play video</span>';
  stage.append(playButton);
  playButton.addEventListener('click', async () => {
    playButton.disabled = true;
    try { await video.play(); }
    catch (_) { playButton.hidden = false; }
    finally { playButton.disabled = false; }
  });
  video.addEventListener('play', () => {
    playButton.hidden = true;
    document.querySelectorAll('video').forEach(other => {
      if (other !== video) other.pause();
    });
  });
  video.addEventListener('pause', () => { playButton.hidden = false; });
  video.addEventListener('ended', () => { playButton.hidden = false; });
});

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
if ('IntersectionObserver' in window) {
  const reveal = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      if (!reducedMotion.matches) entry.target.classList.add('reveal-in');
      reveal.unobserve(entry.target);
    });
  }, { threshold: .08 });
  document.querySelectorAll('.section-heading, .method-card, .demo-panel').forEach(el => reveal.observe(el));
  const sections = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      document.querySelectorAll('.topbar nav a').forEach(link => {
        if (link.hash === '#' + entry.target.id) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-12% 0px -65% 0px' });
  document.querySelectorAll('main section[id]').forEach(section => sections.observe(section));
}

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
