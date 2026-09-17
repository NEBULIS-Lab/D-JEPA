// Rounded within-start Spearman values from the published decision-local diagnostic.
// Source: docs/static/images/decision_local_ranking.svg and the project-page caption.
// Tile layout is explanatory, not per-candidate outcome data.
export const DIAGNOSTIC = Object.freeze([
  Object.freeze({ model: 'LeWM', all: .90, shortlist: .11 }),
  Object.freeze({ model: 'TD-JEPA', all: .80, shortlist: .13 }),
]);
export function problemPhase(progress) {
  const ease = x => { x=Math.max(0,Math.min(1,x)); return x*x*(3-2*x); };
  return {
    motion: .32 * ease(progress / .65),
    focus: ease((progress-.18)/.32),
    reveal: ease((progress-.53)/.27),
    step: progress < .25 ? 0 : progress < .58 ? 1 : 2,
  };
}
