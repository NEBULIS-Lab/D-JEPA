/* Optional browser regression check: npm install --no-save playwright.
 * Record desktop baselines before editing: node tests/check_mobile_layout.cjs --record
 * Then run without --record. Baselines stay outside the repository in /tmp.
 * Catches desktop visual changes, inaccessible touch controls and page overflow.
 * --artwork-update permits intentional homepage artwork changes only.
 */
const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../docs');
const out=process.env.LAYOUT_BASELINE_DIR||'/tmp/djepa-mobile-layout';
fs.mkdirSync(out,{recursive:true});
const record=process.argv.includes('--record'),failures=[];
const artworkUpdate=process.argv.includes('--artwork-update');
const types={'.html':'text/html','.css':'text/css','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.mp4':'video/mp4','.mp3':'audio/mpeg','.pdf':'application/pdf'};
const server=http.createServer((req,res)=>{
  const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
  if(!file.startsWith(root+path.sep)){res.writeHead(403).end();return;}
  fs.readFile(file,(err,data)=>{
    if(err){res.writeHead(404).end();return;}
    const headers={'Content-Type':types[path.extname(file)]||'application/octet-stream'};
    const range=req.headers.range?.match(/bytes=(\d+)-(\d*)/);
    if(range){const start=Number(range[1]),end=range[2]?Math.min(Number(range[2]),data.length-1):data.length-1;
      res.writeHead(206,{...headers,'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${data.length}`,'Content-Length':end-start+1});res.end(data.subarray(start,end+1));
    }else{res.writeHead(200,headers);res.end(data);}
  });
});
function check(ok,message){if(!ok)failures.push(message);}
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const base=`http://127.0.0.1:${server.address().port}`;
 const browser=await chromium.launch({headless:true,args:['--disable-gpu','--disable-gpu-compositing','--disable-webgl','--use-gl=disabled']});
 try{
  // Desktop and narrow portrait desktop must remain byte-identical.
  for(const theme of ['dark','light'])for(const route of ['index','explainer'])for(const [width,height] of [[1440,1000],[600,900]]){
    const context=await browser.newContext({viewport:{width,height},reducedMotion:'reduce'});
    await context.addInitScript(t=>localStorage.setItem('djepa-theme',t),theme);
    const page=await context.newPage();
    await page.goto(`${base}/${route}.html`);
    await page.waitForFunction(r=>r==='index'?document.documentElement.classList.contains('tabs-enabled'):document.querySelector('#detail-visual')?.childElementCount>0,route);
    await page.evaluate(async()=>{document.querySelectorAll('img').forEach(x=>x.loading='eager');await document.fonts.ready;await Promise.all([...document.images].map(x=>x.decode().catch(()=>{})));document.querySelectorAll('video').forEach(v=>v.pause());});
    const shot=await page.screenshot({fullPage:true,animations:'disabled',mask:route==='index'?[page.locator('video')]:[]});
    const file=path.join(out,`${route}-${theme}-${width}.png`);
    if(record)fs.writeFileSync(file,shot);else if(!(artworkUpdate&&route==='index')){
      const equal=fs.existsSync(file)&&fs.readFileSync(file).equals(shot);
      check(equal,`Desktop screenshot changed: ${route}/${theme}/${width}`);
      if(!equal)fs.writeFileSync(file.replace('.png','-after.png'),shot);
    }
    await context.close();
  }
  for(const [width,height] of [[320,740],[390,844],[430,932]])for(const theme of ['dark','light']){
    const context=await browser.newContext({viewport:{width,height},isMobile:true,hasTouch:true,deviceScaleFactor:1,reducedMotion:'reduce'});
    await context.addInitScript(t=>localStorage.setItem('djepa-theme',t),theme);
    const page=await context.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base+'/index.html');
    await page.waitForFunction(()=>document.documentElement.classList.contains('tabs-enabled'));
    check(await page.locator('.paper-artwork img').count()===3,'Three final paper drawings are not embedded');
    if(await page.locator('.paper-artwork img').count()===3){
      await page.locator('.paper-artwork img').evaluateAll(async xs=>{for(const x of xs){x.loading='eager';await x.decode();}});
      check(await page.locator('.paper-artwork img').evaluateAll(xs=>xs.every(x=>x.naturalWidth>=2500)), 'Artwork previews are not high resolution');
      check(await page.locator('.paper-artwork a').evaluateAll(xs=>xs.length===3&&xs.every(x=>x.href.endsWith('.pdf'))),'Original artwork PDF links missing');
    }
    check(await page.locator('.demo-tabs').evaluate(x=>x.scrollWidth<=x.clientWidth),'Phone video tabs still require sideways scrolling');
    for(const tab of await page.locator('.demo-tabs button').all()){
      await tab.click();
      check(await tab.getAttribute('aria-selected')==='true','A video task tab cannot be selected');
    }
    await page.locator('.demo-tabs button').first().click();
    check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Homepage overflow ${width}`);
    check(await page.locator('.problem-copy p').evaluate(x=>getComputedStyle(x).textAlign!=='justify'),`Phone prose remains justified ${width}`);
    check(await page.locator('.topbar nav a').evaluateAll(xs=>xs.every(x=>x.getBoundingClientRect().height>=44)),`Header touch targets below 44px ${width}`);
    if(await page.locator('.hero-authors').count())check(await page.locator('.hero-authors').evaluate(x=>x.scrollWidth<=x.clientWidth),`Authors require sideways scroll ${width}`);
    await page.getByText('Metric definitions and source identities').click();
    await page.waitForSelector('.evidence-dialog[open]');
    check(await page.locator('.evidence-close').evaluate(x=>x.getBoundingClientRect().height>=44),`Dialog close target too small ${width}`);
    await page.locator('.evidence-close').click();
    await page.locator('#demo-pusht a').filter({hasText:'View the timeline'}).click();
    await page.waitForFunction(()=>document.querySelector('.evidence-body img')?.naturalWidth>0);
    await page.locator('.evidence-close').click();
    if(width===390&&theme==='dark'){
      await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(out,`mobile-home-${record?'before':'after'}.png`)});
      // Viewport screenshots preserve touch/orientation emulation between pages.
      await page.locator('.demo-tabs').scrollIntoViewIfNeeded();
      await page.screenshot({path:path.join(out,'mobile-video-tasks.png')});
      if(await page.locator('.paper-artwork').count()===3){
        await page.locator('[data-figure="alignment"]').scrollIntoViewIfNeeded();
        await page.screenshot({path:path.join(out,'mobile-paper-artwork.png')});
      }
    }
    await page.goto(base+'/explainer.html');
    await page.waitForFunction(()=>document.querySelector('#detail-visual')?.childElementCount>0);
    check(await page.locator('#chapters').evaluate(x=>x.scrollWidth<=x.clientWidth),`Chapters require sideways scroll ${width}`);
    for(let i=0;i<6;i++){
      await page.locator(`button[data-stage="${i}"]`).click();
      check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Explainer page overflow ${width}/stage ${i}`);
      check(await page.locator('#detail-description').evaluate(x=>getComputedStyle(x).textAlign!=='justify'),`Inspector remains justified ${width}`);
      const canvas=await page.locator('#diagram-viewport').boundingBox();
      check(canvas.width<=width&&canvas.height>=200,`Unusable diagram viewport ${width}/stage ${i}`);
    }
    for(const id of ['play','narration-play','previous','next'])check(await page.locator('#'+id).evaluate(x=>x.getBoundingClientRect().height>=44),`Playback target too small ${width}/${id}`);
    await page.locator('button[data-stage="0"]').click();
    if(width===390&&theme==='dark'){
      await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(out,`mobile-explainer-${record?'before':'after'}.png`)});
    }
    if(!record&&width===390){
      await page.locator('#play').click();
      await page.waitForTimeout(350);
      check(await page.locator('#play-label').textContent()!=='Play tour','Tour did not start');
      check(await page.locator('[data-tour-guide]:visible').count()===0,'Guide obscures playback');
      await page.locator('#play').click();
      await page.locator('#narration-play').click();
      await page.waitForFunction(()=>document.querySelector('#narration-audio')?.currentTime>.2&&document.querySelector('#narration-subtitle')?.textContent.length>0);
      check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Narration overflows');
      await page.locator('#narration-mute').click();
      check(await page.locator('#narration-audio').evaluate(x=>x.muted),'Mute failed');
      await page.locator('#narration-exit').click();
    }
    check(!errors.length,`Page errors: ${errors.join('; ')}`);
    await context.close();
  }
 }finally{await browser.close();server.close();}
 fs.writeFileSync(path.join(out,record?'baseline-report.json':'report.json'),JSON.stringify({failures},null,2));
 assert.equal(failures.length,0,failures.join('\n'));
 console.log('PASS: '+(artworkUpdate?'paper artwork and unchanged desktop explainer':'unchanged desktop screenshots')+'; phone portrait layout, controls, dialogs and tours.');
})().catch(e=>{console.error(e.message);server.close();process.exitCode=1;});
