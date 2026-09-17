/** Small deterministic teaching example. These are NOT checkpoint outputs or trial data. */
export const CANDIDATES = ['A', 'B', 'C', 'D', 'E', 'F'];
export const COSTS = {
  lewm: [0.21, 0.25, 0.19, 0.63, 0.78, 0.46],
  temporal: [0.20, 0.18, 0.30, 0.49, 0.71, 0.55],
  jepa: [0.22, 0.17, 0.25, 0.50, 0.85, 0.61],
  dino: [0.23, 0.28, 0.19, 0.51, 0.79, 0.64],
};
// Illustrative outputs of a bounded relational correction, not an executed neural model.
const CORRECTION_DIRECTION = [-0.92, 0.36, -0.05, 0.28, -0.32, 0.05];
export function order(values) {
  return values.map((value, id) => ({ value, id })).sort((a, b) => a.value - b.value || a.id - b.id).map(x => x.id);
}
export function ranks(values) {
  const sorted = order(values);
  return values.map((_, i) => sorted.indexOf(i) / (values.length - 1));
}
export function example({ sources = 2, bound = 0.2 } = {}) {
  const sourceRanks = Object.fromEntries(Object.entries(COSTS).map(([key, values]) => [key, ranks(values)]));
  // Two sources: calibrated ordinal fusion. Four sources: its own JEPA-WM base.
  const base = sources === 4 ? [...sourceRanks.jepa] : CANDIDATES.map((_, i) => 0.42 * sourceRanks.lewm[i] + 0.58 * sourceRanks.temporal[i]);
  const delta = CORRECTION_DIRECTION.map(x => x * bound);
  const scores = base.map((x, i) => x + delta[i]);
  const baseWinner = order(base)[0];
  const relationalWinner = order(scores)[0];
  const advantage = base[baseWinner] - scores[relationalWinner];
  const threshold = -0.03; // Schematic calibrated threshold for the teaching example.
  const admitted = advantage > threshold;
  const winner = admitted ? relationalWinner : baseWinner;
  const sorted = order(scores);
  const finalOrder = [winner, ...sorted.filter(i => i !== winner)];
  const ordinal = CANDIDATES.map((_, i) => finalOrder.indexOf(i) + 1);
  return { sourceRanks, base, delta, scores, baseWinner, relationalWinner, advantage, threshold, admitted, winner, finalOrder, ordinal };
}
export function attention(head = 0, sources = 2) {
  const rs = Object.values(COSTS).slice(0, sources).map(ranks);
  return CANDIDATES.map((_, i) => {
    const logits = CANDIDATES.map((__, j) => {
      const difference = rs.reduce((sum, r) => sum + Math.abs(r[i] - r[j]), 0);
      return Math.exp(-difference * (1.3 + head * 0.35) + 0.5 * Math.cos((i + 1) * (j + 1) + head));
    });
    const sum = logits.reduce((a, b) => a + b, 0);
    return logits.map(x => x / sum);
  });
}
export function descriptor(candidate, source, dimension) {
  return (Math.sin((candidate + 1) * 1.7 + dimension * 0.63 + source * 2.1) + 1) / 2;
}
export function realizedPoint(candidate, rank) {
  const radius = rank / (CANDIDATES.length + 1);
  const theta = [-1.0, -2.3, 0.3, 1.1, 2.2, 2.85][candidate];
  // A unit-RMS direction in two dimensions has Euclidean norm sqrt(2).
  return [Math.SQRT2 * radius * Math.cos(theta), Math.SQRT2 * radius * Math.sin(theta)];
}
export function nativeCost(point) { return point.reduce((sum, x) => sum + x * x, 0) / point.length; }
export function transportPoint(candidate, time, progress = 1) {
  const source = [0.13 * time + 0.03 * candidate, 0.20 * Math.sin(time * 0.7 + candidate * 0.5)];
  const angle = time * 0.8 + candidate;
  const beta = 0.1 * Math.tanh(Math.sin(time * 1.4 + candidate));
  return { source, beta, refined: [source[0] + progress * beta * Math.cos(angle), source[1] + progress * beta * Math.sin(angle)] };
}
