/* Recorded timelines and metric definitions stay in the current page. */
(() => {
  const dialog = document.createElement('dialog');
  dialog.className = 'evidence-dialog';
  dialog.id = 'evidence-dialog';
  dialog.setAttribute('aria-labelledby', 'evidence-title');
  const header = document.createElement('header');
  const title = document.createElement('h2');
  title.id = 'evidence-title';
  const close = document.createElement('button');
  close.className = 'evidence-close';
  close.type = 'button';
  close.textContent = '×';
  close.setAttribute('aria-label', 'Close viewer');
  const body = document.createElement('div');
  body.className = 'evidence-body';
  header.append(title, close);
  dialog.append(header, body);
  document.body.append(dialog);
  let opener, request = 0;
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  function open(link, heading) {
    opener = link;
    request += 1;
    title.textContent = heading;
    body.replaceChildren();
    if (!dialog.open) dialog.showModal();
    dialog.scrollTop = 0;
    close.focus({preventScroll:true});
    return request;
  }
  close.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const r = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => { request += 1; opener?.focus({preventScroll:true}); });
  document.querySelectorAll('a[href]').forEach(link => {
    const timeline = link.textContent.includes('View the timeline');
    const metrics = link.textContent.includes('Metric definitions and source identities');
    if (!timeline && !metrics) return;
    link.setAttribute('aria-haspopup', 'dialog');
    link.setAttribute('aria-controls', 'evidence-dialog');
    link.addEventListener('click', async event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      if (timeline) {
        const heading = link.closest('.demo-panel')?.querySelector('h3')?.textContent || 'Recorded execution timeline';
        open(link, heading);
        const img = document.createElement('img');
        img.src = link.href;
        img.alt = heading + ' — matched execution frames';
        img.addEventListener('error', () => { if (dialog.open) body.replaceChildren(element('p', 'This timeline could not be loaded. Please close the viewer and try again.')); });
        body.append(img);
        return;
      }
      const id = open(link, 'Distance preferences and executed outcomes');
      body.append(element('p', 'Loading metric definitions…'));
      try {
        const response = await fetch(link.href);
        if (!response.ok) throw new Error('Evidence unavailable');
        const data = await response.json();
        if (id !== request || !dialog.open) return;
        body.replaceChildren(element('p', 'One recorded decision illustrates the conflict; three aggregate diagnostics measure it across the audit population. Lower predicted distance is the native planner’s preference.'));
        const grid = element('div', '', 'evidence-grid');
        const cards = [
          ['Recorded pair', 'A failing candidate has predicted goal RMS distance 0.2027; a successful alternative has distance 0.2167. Both use the same LeWM predictive space, start and goal.', data.populations.a],
          ['Distance gap', 'Gap = (d₊ − d₋) / (d₊ + d₋), using the smallest predicted RMS distance among successful candidates (d₊) and failing candidates (d₋). Positive values favor failure.', 'Points show all 96 audit starts per model; the central bar marks the median and interquartile interval.'],
          ['Success–failure pair inversions', data.definitions.pair_error + '.', data.definitions.c_counts + '. Eligibility is assessed separately at each shortlist size.'],
          ['Rank correlation', 'Spearman correlation compares predicted goal costs with costs obtained from simulator-realized observations in the same latent space. Within-start coefficients are averaged over starts; pooled coefficients combine the same decisions across starts.', 'All 63 candidates, top 16 and top four; the audit contains 96 starts.']
        ];
        cards.forEach(([heading, text, note]) => {
          const card = element('section', '', 'evidence-card');
          card.append(element('h3', heading), element('p', text), element('small', note));
          grid.append(card);
        });
        body.append(grid);
        const provenance = element('details', '', 'evidence-provenance');
        provenance.append(element('summary', 'Source identities and exact definitions'));
        const list = document.createElement('ul');
        data.sources.forEach(source => {
          const item = document.createElement('li');
          item.append(element('span', source.id));
          Object.entries(source).filter(([key]) => key !== 'id').forEach(([key,value]) =>
            item.append(element('code', key + ': ' + (typeof value === 'object' ? JSON.stringify(value) : value))));
          list.append(item);
        });
        const download = element('a', 'Download the machine-readable definitions');
        download.href = link.href;
        download.download = 'decision_local_ranking.json';
        provenance.append(list, download);
        body.append(provenance);
      } catch (_) {
        if (id === request && dialog.open) body.replaceChildren(element('p', 'The definitions could not be loaded. Please close the viewer and try again.'));
      }
    });
  });
})();
