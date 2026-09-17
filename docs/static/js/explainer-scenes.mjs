// Recorded state replay and clearly separated schematic motion. No physics inference.
export function sampleState(candidate, progress) {
  const last = candidate.states.length - 1;
  const position = Math.max(0, Math.min(1, progress)) * last;
  const index = Math.min(last - 1, Math.floor(position)), fraction = position - index;
  return candidate.states[index].map((value, axis) => {
    let delta = candidate.states[index + 1][axis] - value;
    if (axis === 4) delta = Math.atan2(Math.sin(delta), Math.cos(delta));
    return value + fraction * delta;
  });
}

export function sceneMarkup(record, candidateIndex, x, y, size, suffix = '') {
  const candidate = record.candidates[candidateIndex];
  const [vx, vy, extent] = record.viewport;
  const scale = size / extent;
  const points = record.geometry.tee.map(p => p.join(',')).join(' ');
  const goal = record.goal;
  const id = `clip-${candidate.id}-${suffix}`;
  const shape = (fill, stroke) => `<polygon points="${points}" fill="${fill}" stroke="${stroke}" stroke-width="1.2" stroke-linejoin="round"/>`;
  return `<defs><clipPath id="${id}"><rect x="${x}" y="${y}" width="${size}" height="${size}" rx="10"/></clipPath></defs>
    <rect x="${x}" y="${y}" width="${size}" height="${size}" rx="10" fill="var(--scene-bg)" stroke="var(--line)"/>
    <g clip-path="url(#${id})"><g transform="translate(${x} ${y}) scale(${scale}) translate(${-vx} ${-vy})">
      <g transform="translate(${goal[2]} ${goal[3]}) rotate(${goal[4] * 180 / Math.PI})">${shape('#95EE95', '#95EE95')}</g>
      <circle cx="${goal[0]}" cy="${goal[1]}" r="${record.geometry.pusher_radius}" fill="#95EE95" opacity=".6"/>
      <path data-pusher-trail="${candidateIndex}" fill="none" stroke="#4A75E8" stroke-opacity=".3" stroke-width="2" stroke-dasharray="3 5"/>
      <g data-body="${candidateIndex}">${shape('#92A4B7', '#8091A2')}</g>
      <circle data-pusher="${candidateIndex}" r="${record.geometry.pusher_radius}" fill="#4A75E8"/>
    </g></g>`;
}

export function updateScenes(svg, record, progress) {
  record.candidates.forEach((candidate, i) => {
    const s = sampleState(candidate, progress);
    svg.querySelectorAll(`[data-body="${i}"]`).forEach(node => node.setAttribute('transform', `translate(${s[2]} ${s[3]}) rotate(${s[4] * 180 / Math.PI})`));
    svg.querySelectorAll(`[data-pusher="${i}"]`).forEach(node => { node.setAttribute('cx', s[0]); node.setAttribute('cy', s[1]); });
    const states = candidate.states.slice(0, Math.floor(progress * (candidate.states.length - 1)) + 1);
    const path = states.map((s, j) => `${j ? 'L' : 'M'}${s[0]} ${s[1]}`).join(' ');
    svg.querySelectorAll(`[data-pusher-trail="${i}"]`).forEach(node => node.setAttribute('d', path));
  });
}
