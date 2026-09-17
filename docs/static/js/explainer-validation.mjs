export const VALIDATION_TASKS = Object.freeze([
  {id:'reacher', title:'Reacher', domain:'Articulated control', detail:'TD-JEPA / D-JEPA · recorded matched execution', duration:5.6},
  {id:'granular', title:'Granular manipulation', domain:'Deformable dynamics', detail:'DINO-WM / D-JEPA · recorded particle manipulation', duration:32.7},
  {id:'robotwin', title:'Bimanual grasping', domain:'Robotic manipulation', detail:'Native VLA / D-JEPA · original published comparison, 2× playback', duration:16.37},
  {id:'driving', title:'Autonomous driving', domain:'Trajectory selection', detail:'Drive-JEPA / D-JEPA · recorded camera context and simulated candidate trajectories', duration:12},
  {id:'shape', title:'Unseen object shapes', domain:'Geometric change', detail:'Native / rank fusion / D-JEPA · recorded matched execution', duration:8.04},
]);
const clamp=x=>Math.max(0,Math.min(1,x));
const ease=x=>{x=clamp(x);return x*x*(3-2*x);};
export function validationFrame(seconds) {
  return {
    // Preserve the original eight-second PushT playback, including its end hold.
    pushProgress:ease((seconds/8-.08)/.8),
    shrink:ease((seconds-8)/1.8),
    tasks:VALIDATION_TASKS.map((task,i)=>{
      const entry=9.85+i*.38, reveal=ease((seconds-entry)/.8);
      return {reveal, time:Math.max(0,Math.min(task.duration-.04,seconds-entry-.8)), running:seconds>entry+.8&&seconds<entry+.8+task.duration-.04};
    }),
  };
}

// Videos share the tour clock. Hidden windows never keep playing in the background.
const pending=new WeakSet();
export function pauseValidationVideos(root=document) {
  root.querySelectorAll('[data-validation-video]').forEach(v=>v.pause());
}
export function releaseValidationVideos(root) {
  root.querySelectorAll('[data-validation-video]').forEach(v=>{
    v.pause();if(v.dataset.blobUrl)URL.revokeObjectURL(v.dataset.blobUrl);
  });
}
export function syncValidationVideos(root,seconds,playing) {
  const frame=validationFrame(seconds);
  root.querySelectorAll('[data-validation-video]').forEach(v=>{
    const i=Number(v.dataset.validationVideo), f=frame.tasks[i];
    const shouldPlay=playing&&f.running&&f.reveal>.9;
    v.dataset.wantsPlayback=String(shouldPlay);
    v.dataset.targetTime=f.time;
    v.muted=true;
    if(f.reveal>0&&!v.dataset.loadStarted){
      v.dataset.loadStarted='true';
      v.onloadedmetadata=()=>{v.currentTime=Math.min(Number(v.dataset.targetTime),Math.max(0,v.duration-.04));};
      v.onloadeddata=()=>{const target=Math.min(Number(v.dataset.targetTime),Math.max(0,v.duration-.04));if(Math.abs(v.currentTime-target)>.03)v.currentTime=target;};
      // Small previews are fully buffered, allowing deterministic scrubbing even
      // on a simple local HTTP server without byte-range support.
      fetch(v.dataset.src).then(r=>{if(!r.ok)throw Error('Preview unavailable');return r.blob();}).then(blob=>{
        if(!v.isConnected)return;
        v.dataset.blobUrl=URL.createObjectURL(blob);v.src=v.dataset.blobUrl;v.load();
      }).catch(()=>{v.dataset.playbackBlocked='true';});
    }
    v.playbackRate=Number(root.dataset.playbackSpeed||1);
    if(v.readyState>=1){
      const time=Math.min(f.time,Math.max(0,v.duration-.04));
      if(Math.abs(v.currentTime-time)>(playing?.4:.03))v.currentTime=time;
    }
    if(!shouldPlay){v.pause();return;}
    if(v.paused&&v.readyState>=2&&!pending.has(v)){
      pending.add(v);
      v.play().then(()=>{
        if(v.dataset.wantsPlayback!=='true'||!v.isConnected)v.pause();
      }).catch(()=>{v.dataset.playbackBlocked='true';}).finally(()=>pending.delete(v));
    }
  });
}
