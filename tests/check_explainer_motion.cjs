/* Optional real-browser checks: install playwright, then run with Node.
 * Set EXPLAINER_SHOTS to save screenshots; --narrated also plays all eight recordings.
 * Browser rendering is explicitly CPU-only. No research model is initialized.
 */
const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../docs');
const shots=process.env.EXPLAINER_SHOTS;
if(shots)fs.mkdirSync(shots,{recursive:true});
const mime={'.html':'text/html','.css':'text/css','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.svg':'image/svg+xml','.jpg':'image/jpeg','.mp3':'audio/mpeg','.mp4':'video/mp4'};
const server=http.createServer((req,res)=>{
 const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
 if(!file.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 fs.readFile(file,(err,data)=>{
  if(err){res.writeHead(404).end();return;}
  const headers={'Content-Type':mime[path.extname(file)]||'application/octet-stream'};
  const range=req.headers.range?.match(/bytes=(\d+)-(\d*)/);
  if(range){const start=Number(range[1]),end=range[2]?Math.min(Number(range[2]),data.length-1):data.length-1;
   res.writeHead(206,{...headers,'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${data.length}`,'Content-Length':end-start+1});res.end(data.subarray(start,end+1));
  }else{res.writeHead(200,headers);res.end(data);}
 });
});
const errors=[];
async function progress(page,p){await page.locator('#step-progress').evaluate((n,p)=>{n.value=p;n.dispatchEvent(new Event('input',{bubbles:true}));},p);}
async function shot(page,name){if(shots)await page.screenshot({path:path.join(shots,name+'.png'),fullPage:true});}
async function fresh(browser,base,theme,mobile=false,reducedMotion='no-preference'){
 const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1366,height:768},isMobile:mobile,hasTouch:mobile,reducedMotion});
 await context.addInitScript(t=>{localStorage.setItem('djepa-theme',t);sessionStorage.setItem('djepa-tour-guide-dismissed','true');},theme);
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base+'/explainer.html');await page.waitForSelector('#pair-orbit');
 return {context,page};
}
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({headless:true,args:['--disable-gpu','--disable-gpu-compositing','--disable-webgl','--use-gl=disabled']});
 const base=`http://127.0.0.1:${server.address().port}`;
 try{
  for(const theme of ['dark','light']){
   const {context,page}=await fresh(browser,base,theme);
   await page.locator('#chapters [data-stage="1"]').click();
   for(const [phase,name] of [[.15,'descriptors'],[.48,'ranks'],[.70,'assembly'],[1,'tokens']]){
    await progress(page,phase);await shot(page,theme+'-'+name);
    const focus=await page.locator('[data-computation-zone]').evaluateAll(ns=>ns.map(n=>Number(n.style.opacity)));
    assert.ok(focus.every(v=>v>=.6&&v<=1));
    assert.equal(focus[phase<.28?0:phase<.57?1:2],1);
   }
   await page.locator('[data-sources="4"]').click();await progress(page,.7);await shot(page,theme+'-four-sources');
   await page.locator('[data-sources="2"]').click();
   // A natural chapter boundary must carry the selected teaching candidate,
   // freeze on pause, then disappear completely when the viewer scrubs.
   await page.locator('#chapters [data-stage="1"]').click();await progress(page,.985);await page.locator('#play').click();
   await page.waitForSelector('[data-carry-overlay]');await page.locator('#play').click();
   const frozen=await page.locator('[data-carry-overlay]').innerHTML();await page.waitForTimeout(180);
   assert.equal(await page.locator('[data-carry-overlay]').innerHTML(),frozen);
   await shot(page,theme+'-carry');
   await progress(page,.01);assert.equal(await page.locator('[data-carry-overlay]').count(),0,'manual scrub must cancel a chapter carry');
   assert.ok(await page.locator('#detail-visual [data-token]').evaluateAll(ns=>ns.every(n=>n.style.visibility!=='hidden')));
   await progress(page,.46);await shot(page,theme+'-relations');
   const frozenScene=await page.locator('#detail-visual').innerHTML();await page.waitForTimeout(160);
   assert.equal(await page.locator('#detail-visual').innerHTML(),frozenScene);
   // Rewinding recovers identical computed geometry, not an accumulated tween state.
   await progress(page,.9);await progress(page,.46);
   assert.equal(await page.locator('#detail-visual').innerHTML(),frozenScene);
   await page.locator('.matrix-cell[data-row="3"][data-col="1"]').click();
   assert.equal(await page.locator('#candidate-controls [data-candidate="3"]').getAttribute('aria-pressed'),'true');
   await page.locator('#candidate-controls [data-candidate="0"]').click();
   await progress(page,.96);await shot(page,theme+'-correction');
   await page.locator('#chapters [data-stage="3"]').click();await progress(page,.9);await shot(page,theme+'-decision');
   await page.locator('#chapters [data-stage="4"]').click();
   await progress(page,.3);const before=await page.locator('[data-lift-point="0"]').getAttribute('cx');
   await progress(page,.55);assert.notEqual(await page.locator('[data-lift-point="0"]').getAttribute('cx'),before);
   await shot(page,theme+'-lifting');await progress(page,.79);await shot(page,theme+'-landing');
   await progress(page,1);await shot(page,theme+'-realized');
   const finalCosts=await page.locator('[data-native-cost]').allTextContents();
   assert.deepEqual(finalCosts,['0.020','0.082','0.184','0.327','0.510','0.735']);
   await page.locator('#candidate-controls [data-candidate="4"]').click();
   assert.deepEqual(await page.locator('[data-native-cost]').allTextContents(),finalCosts,'presentation selection must not change native costs');
   const compare=page.locator('#compare-before');await compare.focus();await page.keyboard.down('Space');
   assert.notDeepEqual(await page.locator('[data-native-cost]').allTextContents(),finalCosts);
   await page.keyboard.up('Space');assert.deepEqual(await page.locator('[data-native-cost]').allTextContents(),finalCosts);
   await page.locator('[data-lifting="transport"]').click();await progress(page,.6);await shot(page,theme+'-transport');
   await page.locator('#chapters [data-stage="5"]').click();await progress(page,0);
   assert.ok(await page.locator('[data-validation-card]').evaluateAll(ns=>ns.every(n=>getComputedStyle(n).display==='none')));
   await progress(page,12/26);await page.waitForTimeout(1200);await shot(page,theme+'-wall');
   await progress(page,0);
   assert.ok(await page.locator('[data-validation-video]').evaluateAll(ns=>ns.every(n=>getComputedStyle(n).display==='none'&&n.paused)));
   assert.ok(await page.locator('.playback').evaluate(n=>n.getBoundingClientRect().bottom<=innerHeight+1));
   await context.close();
   const mobile=await fresh(browser,base,theme,true);
   for(const stage of [1,2,4,5]){
    await mobile.page.locator('#chapters [data-stage="'+stage+'"]').click();await progress(mobile.page,.6);
    assert.ok(await mobile.page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'phone must not overflow horizontally');
    await shot(mobile.page,theme+'-phone-'+stage);
   }
   await mobile.context.close();
  }
  const reduced=await fresh(browser,base,'dark',false,'reduce');
  await reduced.page.locator('#chapters [data-stage="1"]').click();await progress(reduced.page,.985);await reduced.page.locator('#play').click();
  await reduced.page.waitForFunction(()=>document.documentElement.dataset.stage==='2');
  assert.equal(await reduced.page.locator('[data-carry-overlay]').count(),0);await reduced.context.close();
  console.log('PASS: candidate carries, pause/scrub, reversible geometry, hidden media, both themes, portrait and reduced motion');
  if(process.argv.includes('--narrated')){
   const {context,page}=await fresh(browser,base,'dark');
   const manifest=JSON.parse(fs.readFileSync(path.join(root,'static/data/explainer-narration.json')));
   await page.evaluate(()=>{window.samples=[];window.sampleTimer=setInterval(()=>{
    const a=document.querySelector('#narration-audio');if(!a||a.paused)return;
    window.samples.push({time:a.currentTime,duration:a.duration,stage:Number(document.documentElement.dataset.stage),caption:document.querySelector('#narration-subtitle').textContent,
     map:document.querySelector('#model-map').hidden?null:document.querySelector('#model-map').dataset.mode,timeline:document.querySelector('#timeline-label').textContent});
   },180);});
   await page.locator('#narration-play').click();
   const heartbeat=setInterval(()=>console.log('Narrated tour: normal-speed playback in progress'),30000);
   try{await page.waitForFunction(()=>document.querySelector('#narration-play-label').textContent==='Replay narration',null,{timeout:260000});}
   finally{clearInterval(heartbeat);}
   const samples=await page.evaluate(()=>{clearInterval(window.sampleTimer);return window.samples;});
   const segments=new Set(),captions=new Set();
   for(const s of samples){
    const id=Number(s.timeline.match(/(\d) \/ 8/)?.[1]);if(!id)continue;
    const segment=manifest.segments[id-1];segments.add(id);captions.add(s.caption);
    assert.equal(s.stage,segment.stage);assert.ok(Math.abs(s.duration-segment.duration)<.1);
    let sentence=segment.sentences[0];for(const row of segment.sentences)if(s.time>=row.start)sentence=row;
    if(!segment.sentences.some(row=>Math.abs(row.start-s.time)<.22))assert.equal(s.caption,sentence.text);
    const cues=segment.architecture||[],cue=cues.find(c=>s.time>=c.start&&s.time<c.end);
    if(!cues.some(c=>Math.abs(c.start-s.time)<.22||Math.abs(c.end-s.time)<.22))assert.equal(s.map,cue?.mode||null);
   }
   assert.equal(segments.size,8);assert.equal(captions.size,36);
   assert.equal(await page.locator('#narration-audio').evaluate(a=>a.paused),true);
   await shot(page,'narrated-complete');await context.close();
   console.log('PASS: eight audio segments, 36 sentence captions and architecture cues at normal speed');
  }
  assert.deepEqual(errors,[]);console.log('PASS: no browser errors');
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
