"use strict";
const $ = id => document.getElementById(id);
const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num = (value, digits=2) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
let state = null, paused = false, experiment = null, captureA = null, captureB = null;
const stats = entries => `<div class="stats">${entries.map(([label,value])=>`<div class="stat"><small>${escape(label)}</small><strong>${escape(value)}</strong></div>`).join("")}</div>`;
function chart(target, series, {percent=false, parity=false, xlabel="Trial", maxY=null}={}) {
  const valid = series.flatMap(s=>s.points).filter(p=>p.y!==null && Number.isFinite(p.y));
  if (!valid.length) { $(target).innerHTML='<div class="empty">No measurements available for this chart.</div>'; return; }
  const w=1000,h=190,left=48,right=20,top=15,bottom=32;
  const maxX=Math.max(1,...valid.map(p=>p.x)), maxValue=maxY??Math.max(parity?1.2:0,...valid.map(p=>p.y))*1.12;
  const scaleX=x=>left+x/maxX*(w-left-right), scaleY=y=>h-bottom-y/maxValue*(h-top-bottom);
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escape(xlabel)} time series">`;
  for(let i=0;i<=4;i++){const y=maxValue*i/4;svg+=`<line x1="${left}" x2="${w-right}" y1="${scaleY(y)}" y2="${scaleY(y)}" stroke="#e4e9e4"/><text x="${left-10}" y="${scaleY(y)+4}" text-anchor="end" fill="#74817c" font-size="10">${num(y,percent?0:1)}${percent?"%":""}</text>`;}
  if(parity) svg+=`<line x1="${left}" x2="${w-right}" y1="${scaleY(1)}" y2="${scaleY(1)}" stroke="#a6b2aa" stroke-dasharray="4 4"/>`;
  for(let i=0;i<=4;i++){let x=maxX*i/4;svg+=`<text x="${scaleX(x)}" y="${h-10}" text-anchor="middle" fill="#74817c" font-size="10">${num(x,maxX>20?0:1)}</text>`;}
  series.forEach(s=>{let path="",previous=null; s.points.forEach(p=>{if(p.y===null||!Number.isFinite(p.y)){previous=null;return;}const gap=previous&&p.maxGap&&p.x-previous.x>p.maxGap; path+=`${!previous||gap?"M":"L"}${scaleX(p.x).toFixed(2)},${scaleY(p.y).toFixed(2)} `;previous=p;});svg+=`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linejoin="round"/>`; if(s.points.length<40)s.points.forEach(p=>{if(p.y!==null&&Number.isFinite(p.y))svg+=`<circle cx="${scaleX(p.x)}" cy="${scaleY(p.y)}" r="3.5" fill="${s.color}"><title>${escape(p.label??s.name)}: ${num(p.y)}</title></circle>`;});});
  $(target).innerHTML=svg+"</svg>";
}
function options(id, items, selected, label) {
  const signature=JSON.stringify(items.map(i=>[i.id??i.run_id,label(i)]));
  if($(id).dataset.signature!==signature){$(id).innerHTML=items.map(i=>`<option value="${escape(i.id??i.run_id)}">${escape(label(i))}</option>`).join("");$(id).dataset.signature=signature;}
  if(selected&&items.some(i=>(i.id??i.run_id)===selected))$(id).value=selected;
}
function benchmarkCard(role,b,latency,throughput) {
  return `<div class="deploy-head"><span class="deploy-name">${role}</span><span class="badge">${escape(b.hardware)}</span></div><div class="deploy-note">Same paired replay · ${b.trial_count} trials</div><div class="hero-metric">${num(latency)} <span>s</span></div><div class="metric-label">Median inference latency</div>${stats([["Output throughput",`${num(throughput,1)} tok/s`],["Output agreement",b.correctness==="reported_pass"?"Reported pass":"Unverified"],["GPU idle", "Not recorded"]])}`;
}
function renderBenchmarks(){
  const runs=state.benchmarks;
  options("experiment",runs,experiment,b=>b.name.replace(/[-_]/g," "));
  const b=runs.find(b=>b.id===experiment)??runs.at(-1); experiment=b?.id??null; if(b)$("experiment").value=b.id;
  if(!b){$("outcome").innerHTML='<div class="details"><b>Waiting for benchmark evidence</b>Add paired result files to the watched directory, or try the demo.</div>';$("baseline").innerHTML='<div class="empty">No paired baseline yet</div>';$("candidate").innerHTML='<div class="empty">No paired candidate yet</div>';$("scope").textContent="Live GPU telemetry alone does not establish an optimization gain.";}
  else{
    const pass=b.correctness==="reported_pass"; $("outcome").classList.toggle("unverified",!pass);
    $("outcome").innerHTML=`<div class="gain">${num(b.speedup)}×</div><div class="details"><b>${pass?"Observed paired latency speedup":"Timing result · correctness unverified"}</b>${num(b.latency_reduction_fraction*100,1)}% less total inference time across ${b.trial_count} paired trials${b.data_kind==="synthetic"?" · ILLUSTRATIVE DATA":""}</div>`;
    $("baseline").innerHTML=benchmarkCard("Baseline",b,b.reference_median_s,b.reference_tokens_per_second);
    $("candidate").innerHTML=benchmarkCard("Candidate",b,b.candidate_median_s,b.candidate_tokens_per_second);
    $("scope").textContent=`Scope: ${b.metadata.scope??"Producer did not supply workload scope."} · Throughput unit: ${b.token_unit}. Producer correctness claims are not independently attested.`;
  }
  chart("latency-chart", b?[{name:"Baseline",color:"#556bc5",points:b.trials.map(r=>({x:r.trial,y:r.reference_s,label:r.case}))},{name:"Candidate",color:"#087f73",points:b.trials.map(r=>({x:r.trial,y:r.candidate_s,label:r.case}))}]:[]);
  chart("history-chart",[{name:"Paired speedup",color:"#087f73",points:runs.map((r,i)=>({x:i+1,y:r.speedup,label:r.name}))}],{parity:true,xlabel:"Experiment"});
  $("experiments").innerHTML=runs.length?runs.map(r=>`<tr><td><strong>${escape(r.name.replace(/[-_]/g," "))}</strong>${r.data_kind==="synthetic"?" · illustrative":""}</td><td class="${r.speedup>=1?"positive":"negative"}">${num(r.speedup)}×</td><td>${num(r.latency_reduction_fraction*100,1)}%</td><td>${r.trial_count}</td><td>${r.correctness==="reported_pass"?"Reported pass":"Unverified"}</td></tr>`).join(""):'<tr><td colspan="5">No experiment artifacts yet.</td></tr>';
}
function telemetryCard(report,role){
  if(!report)return '<div class="empty">No capture selected</div>';
  const covered=report.gpus.reduce((s,g)=>s+g.covered_seconds,0), window=report.gpus.reduce((s,g)=>s+g.window_seconds,0);
  const idle=report.gpus.filter(g=>g.estimated_engine_idle_seconds!==null).reduce((s,g)=>s+g.estimated_engine_idle_seconds,0);
  const fraction=covered?idle/covered:null, tokens=report.serving.generation_tokens.rate_per_second;
  return `<div class="deploy-head"><span class="deploy-name">${escape(report.deployment)}</span><span class="badge">${role} · ${escape(report.engine)}</span></div><div class="deploy-note">${escape(report.allocation)} GPU allocation · ${escape(report.status)}</div><div class="hero-metric">${num(fraction===null?null:fraction*100,1)}<span> %</span></div><div class="metric-label">Estimated engine inactivity on covered intervals</div>${stats([["GPU idle estimate",covered?`${num(idle,1)} GPU-s`:"Unknown"],["Metric coverage",`${num(window?covered/window*100:0,0)}%`],["Output tokens/s",num(tokens,1)]])}${report.gpus.map(g=>{const active=g.estimated_engine_idle_fraction===null?null:1-g.estimated_engine_idle_fraction;return `<div class="gpu-row"><span>${escape(g.gpu)}</span><div class="gpu-bar" aria-label="Mean observed engine activity">${Array.from({length:24},(_,i)=>`<i class="${active!==null&&i<active*24?"active":""}"></i>`).join("")}</div><span>${num(active===null?null:active*100,0)}% active</span></div>`;}).join("")}<div class="findings">${report.findings.map(f=>escape(f.kind.replaceAll("_"," "))).join(" · ")||"No diagnostic finding in this capture"}</div>`;
}
function captureSeries(report,color){
  if(!report)return {name:"",color,points:[]}; const grouped=new Map();
  report.timeline.forEach(r=>{const key=r.wall_time_ns;const prev=grouped.get(key)??{sum:0,count:0,dt:r.interval_seconds};prev.sum+=r.engine_active_fraction;prev.count++;grouped.set(key,prev);});
  const rows=[...grouped].sort((a,b)=>a[0]-b[0]), start=rows[0]?.[0]??0;
  return {name:report.deployment,color,points:rows.map(([t,r])=>({x:(t-start)/1e9,y:r.count===report.gpus.length?r.sum/r.count*100:null,maxGap:r.dt*1.5}))};
}
function renderCaptures(){
  const reports=state.reports; $("captures").hidden=!reports.length;
  if(!reports.length)return;
  const a=reports.find(r=>r.run_id===captureA)??reports.at(-2)??reports[0];
  const b=reports.find(r=>r.run_id===captureB)??reports.at(-1);
  captureA=a.run_id;captureB=b.run_id;
  const label=r=>`${r.deployment} · ${r.run_id.slice(0,12)} · ${r.status}`;
  options("capture-a",reports,captureA,label);options("capture-b",reports,captureB,label);
  $("telemetry-a").innerHTML=telemetryCard(a,"Baseline");$("telemetry-b").innerHTML=telemetryCard(b,"Candidate");
  chart("activity-chart",[captureSeries(a,"#556bc5"),captureSeries(b,"#087f73")],{percent:true,maxY:100,xlabel:"Elapsed seconds"});
}
function renderHost(){
  const local=state.local,last=local.at(-1),start=local[0]?.wall_time_ns??0;
  $("host-status").textContent=state.monitor_enabled?(last?.status??"STARTING"):"NOT ENABLED";
  $("host-summary").innerHTML=last?`<div><strong>${num(last.gpu_activity_percent,0)}%</strong><span>GPU activity</span></div><div><strong>${num(last.gpu_memory_bytes===null?null:last.gpu_memory_bytes/2**30,1)} GB</strong><span>GPU memory in use</span></div><div class="host-description">${escape(last.scope)}<br>${escape(last.measurement)}</div>`:'<div class="host-description">Start with --monitor-local on Apple Silicon to record available host counters.<br>NVIDIA deployments use DCGM capture data above.</div>';
  chart("host-chart",[{name:"Apple GPU",color:"#087f73",points:local.map(r=>({x:(r.wall_time_ns-start)/1e9,y:r.gpu_activity_percent,maxGap:6}))}],{percent:true,maxY:100,xlabel:"Elapsed seconds"});
}
function render(){
  const demo=state.mode==="synthetic_demo";$("mode").textContent=demo?"ILLUSTRATIVE DEMO":"LOCAL EVIDENCE";$("mode").classList.toggle("demo",demo);
  $("mode-description").textContent=demo?"Synthetic examples show how comparisons work. These are not measured gains.":"Watching benchmark artifacts and retained captures. Updates every two seconds.";
  $("notice").textContent=state.issues.length?`${state.issues.length} data source issue(s): ${state.issues.map(i=>i.kind.replaceAll("_"," ")).join(", ")}. Available evidence is still shown.`:"";
  renderBenchmarks();renderCaptures();renderHost();$("updated").textContent=`Updated ${new Date(state.updated_ns/1e6).toLocaleTimeString()}`;
}
async function refresh(){if(paused)return;try{const response=await fetch("/api/state");if(!response.ok)throw new Error("HTTP error");state=await response.json();render();$("connection").textContent="Connected";}catch{$("connection").textContent="Connection unavailable";$("notice").textContent="The dashboard cannot reach its local collector. Last available evidence remains visible.";}}
$("pause").onclick=()=>{paused=!paused;$("pause").textContent=paused?"Resume updates":"Pause updates";$("connection").textContent=paused?"Paused":"Connecting";if(!paused)refresh();};
$("experiment").onchange=()=>{experiment=$("experiment").value;renderBenchmarks();};$("capture-a").onchange=()=>{captureA=$("capture-a").value;renderCaptures();};$("capture-b").onchange=()=>{captureB=$("capture-b").value;renderCaptures();};
$("download").onclick=()=>{if(!state)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="obsrv-evidence.json";a.click();URL.revokeObjectURL(url);};
refresh();setInterval(refresh,2000);
