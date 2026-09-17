// Pure timing model: audio timestamps drive both captions and visual holds.
import {modelMapCue} from './explainer-model-map.mjs';
export function locateNarration(segments, seconds) {
  let remaining=Math.max(0,seconds);
  for(let index=0;index<segments.length;index++){
    if(remaining<segments[index].duration||index===segments.length-1)
      return {index,time:Math.min(remaining,segments[index].duration)};
    remaining-=segments[index].duration;
  }
  return {index:0,time:0};
}
export function narrationFrame(segment,time) {
  const t=Math.max(0,Math.min(segment.duration,time));
  const sentences=segment.sentences;
  let sentence=sentences[0];
  for(const s of sentences){if(t>=s.start)sentence=s;else break;}
  const points=segment.keyframes;
  let left=points[0],right=points[0];
  for(let i=1;i<points.length;i++){
    right=points[i];if(t<right.time)break;left=right;
  }
  const span=right.time-left.time;
  const ratio=span>0?Math.max(0,Math.min(1,(t-left.time)/span)):0;
  const elapsed=left.elapsed+(right.elapsed-left.elapsed)*ratio;
  return {stage:segment.stage,elapsed,sources:left.sources||2,
    lifting:segment.lifting||'ordinal',candidate:0,
    predictor:!!left.predictor,sentence,architecture:modelMapCue(segment,t),
    rate:span>0?(right.elapsed-left.elapsed)/span:0};
}
