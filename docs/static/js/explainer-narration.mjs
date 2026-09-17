import { locateNarration, narrationFrame } from './explainer-narration-model.mjs';

export function createNarration(adapter) {
  const $=s=>document.querySelector(s),audio=new Audio();
  audio.preload='auto';audio.id='narration-audio';audio.setAttribute('aria-hidden','true');
  document.body.append(audio);
  let manifest=null,index=0,active=false,running=false,waiting=false,raf=0,generation=0;
  let loadPromise=null,ended=false,loading=false,lastSentence='',rate=1,intent=0,failed=false;
  const blobs=new Map(),requests=new Map();
  const total=()=>manifest?.segments.reduce((sum,s)=>sum+s.duration,0)||0;
  const offset=()=>manifest?.segments.slice(0,index).reduce((sum,s)=>sum+s.duration,0)||0;
  const stamp=t=>Math.floor(t/60).toString().padStart(2,'0')+':'+Math.floor(t%60).toString().padStart(2,'0');
  const status=message=>{$('#narration-status').textContent=message;};
  function mode(value){
    active=value;document.documentElement.classList.toggle('narrating',value);
    $('#narration-caption').hidden=!value;$('#narration-mute').hidden=!value;
    $('#narration-exit').hidden=!value;$('#narration-play').setAttribute('aria-pressed',String(value));
    if(value&&document.fullscreenElement?.id==='explorer')document.exitFullscreen().catch(()=>{});
    if(!value){$('#narration-play-label').textContent='Narrated tour';status('');}
  }
  function controls(){
    $('#narration-play-label').textContent=loading?'Loading…':running?'Pause narration':ended?'Replay narration':'Resume narration';
    $('#narration-play').setAttribute('aria-label',running?'Pause narrated tour':ended?'Replay narrated tour':'Play narrated tour');
    if(!manifest||!active)return;
    $('#tour-seek').max=total();$('#tour-seek').value=offset()+audio.currentTime;
    $('#tour-seek').setAttribute('aria-label','Narrated tour timeline');
    $('#tour-seek').style.setProperty('--fill',100*(offset()+audio.currentTime)/total()+'%');
    $('#tour-time').value=stamp(offset()+audio.currentTime)+' / '+stamp(total());
    $('#timeline-label').textContent=(waiting?'Buffering':ended?'Complete':running?'Narrated tour':'Narration paused')+' · '+(index+1)+' / '+manifest.segments.length;
    $('#previous').disabled=index===0;$('#next').disabled=index===manifest.segments.length-1;
    $('#previous').setAttribute('aria-label','Previous narration segment');$('#next').setAttribute('aria-label','Next narration segment');
  }
  function draw(){
    if(!active||!manifest||loading)return;
    const segment=manifest.segments[index],f=narrationFrame(segment,audio.currentTime);
    adapter.frame(f,{running:running&&!waiting,rate:f.rate*rate});
    if(f.sentence.text!==lastSentence){
      $('#narration-subtitle').textContent=f.sentence.text;lastSentence=f.sentence.text;
    }
    controls();
  }
  function tick(){if(!running||!active)return;draw();raf=requestAnimationFrame(tick);}
  function pause(){
    if(!active)return;
    intent++;
    running=false;audio.pause();cancelAnimationFrame(raf);adapter.pause();controls();
  }
  function exit(){
    if(!active)return;
    pause();generation++;loading=false;mode(false);lastSentence='';
    $('#tour-seek').setAttribute('aria-label','Guided tour timeline');
    $('#previous').setAttribute('aria-label','Previous stage');$('#next').setAttribute('aria-label','Next stage');
    adapter.exit();
  }
  async function load(){
    if(!loadPromise)loadPromise=fetch('static/data/explainer-narration.json').then(r=>{
      if(!r.ok)throw Error('Narration timing is unavailable.');return r.json();
    }).then(data=>{manifest=data;return data;}).catch(e=>{loadPromise=null;throw e;});
    return loadPromise;
  }
  async function asset(i){
    if(blobs.has(i))return blobs.get(i);
    if(!requests.has(i))requests.set(i,fetch(manifest.segments[i].audio).then(r=>{
      if(!r.ok)throw Error('Audio '+(i+1)+' could not be loaded.');return r.blob();
    }).then(blob=>{const url=URL.createObjectURL(blob);blobs.set(i,url);return url;})
      .catch(e=>{requests.delete(i);throw e;}));
    return requests.get(i);
  }
  function fail(error){pause();loading=false;waiting=false;failed=true;status(error.message+' Press Narrated tour to retry.');controls();}
  async function setSegment(next,time=0,autoplay=false){
    const token=++generation;pause();index=next;ended=false;failed=false;loading=true;status('Loading audio…');controls();
    const playIntent=intent;
    try{
      const url=await asset(index);if(token!==generation||!active)return;
      if(audio.src!==url){
        await new Promise((resolve,reject)=>{
          const done=()=>{cleanup();resolve();},bad=()=>{cleanup();reject(Error('This audio cannot be decoded.'));};
          const cleanup=()=>{audio.removeEventListener('loadedmetadata',done);audio.removeEventListener('error',bad);};
          audio.addEventListener('loadedmetadata',done,{once:true});audio.addEventListener('error',bad,{once:true});
          audio.src=url;audio.load();
        });
      }
      if(token!==generation||!active)return;
      audio.currentTime=Math.min(time,Math.max(0,audio.duration-.005));audio.playbackRate=rate;
      loading=false;waiting=false;status('');lastSentence='';draw();
      if(index+1<manifest.segments.length)asset(index+1).catch(()=>{});
      if(autoplay&&playIntent===intent)await resume();
    }catch(e){if(token===generation&&active)fail(e);}
  }
  async function resume(){
    if(loading)return;
    const request=++intent;
    try{
      await audio.play();if(!active||request!==intent){audio.pause();return;}
      running=true;waiting=false;status('');cancelAnimationFrame(raf);draw();raf=requestAnimationFrame(tick);
    }catch(e){if(active&&request===intent)fail(Error('Audio playback needs another click.'));}
  }
  async function toggle(){
    if(running){pause();return;}
    if(loading)return;
    if(!active){
      adapter.stop();mode(true);loading=true;status('Loading narration…');controls();
      try{await load();loading=false;if(active)await setSegment(0,0,true);}catch(e){fail(e);}
    }else if(ended)await setSegment(0,0,true);
    else if(failed&&manifest)await setSegment(index,0,true);
    else if(!manifest||!audio.src){exit();await toggle();}
    else await resume();
  }
  async function seek(seconds){
    if(!manifest)return;const position=locateNarration(manifest.segments,seconds);
    const wasRunning=running;
    await setSegment(position.index,position.time,wasRunning);
  }
  async function step(delta){if(manifest)await setSegment(Math.max(0,Math.min(manifest.segments.length-1,index+delta)),0,running);}
  audio.addEventListener('ended',()=>{
    if(!active)return;
    if(index+1<manifest.segments.length)setSegment(index+1,0,true);
    else{ended=true;running=false;cancelAnimationFrame(raf);draw();adapter.pause();controls();}
  });
  audio.addEventListener('waiting',()=>{if(active&&running){waiting=true;adapter.pause();controls();}});
  audio.addEventListener('playing',()=>{waiting=false;});
  audio.addEventListener('error',()=>{if(active&&!loading)fail(Error('Audio playback failed.'));});
  $('#narration-play').onclick=toggle;
  $('#narration-exit').onclick=exit;
  $('#narration-mute').onclick=()=>{
    audio.muted=!audio.muted;$('#narration-mute').setAttribute('aria-pressed',String(audio.muted));
    $('#narration-mute').setAttribute('aria-label',audio.muted?'Unmute narration':'Mute narration');
    $('#narration-mute-label').textContent=audio.muted?'Muted':'Sound';
  };
  window.addEventListener('pagehide',()=>{pause();});
  return {get active(){return active;},get running(){return running&&!waiting;},toggle,pause,exit,seek,step,refreshControls:controls,
    setRate(value){rate=value;audio.playbackRate=rate;},get audio(){return audio;}};
}
