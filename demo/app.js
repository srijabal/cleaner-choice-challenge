'use strict';
let bundle,savedRuns,db;const $=id=>document.getElementById(id);
const labels={current_only:'Current only',with_history:'With history',starter:'Original starter'};
function put(row){return new Promise((resolve,reject)=>{const t=db.transaction('runs','readwrite');t.objectStore('runs').put(row);t.oncomplete=()=>resolve();t.onerror=()=>reject(t.error);});}
function all(){return new Promise((resolve,reject)=>{const q=db.transaction('runs').objectStore('runs').getAll();q.onsuccess=()=>resolve(q.result);q.onerror=()=>reject(q.error);});}
async function refreshLocal(){const rows=await all();$('local-run').replaceChildren(new Option('Select a run',''));rows.forEach(r=>$('local-run').add(new Option(r.id+' · '+r.status,r.id)));}
function input(){
  const sample=$('example').value==='new'?null:bundle.examples[Number($('example').value)];
  return {household:{household_id:sample?.household.household_id||'new',budget:$('budget').value,usual_brand:$('usual').value,income_band:$('income').value,bathrooms:$('bathrooms').value,cleaning_spend_usd:$('spend').value},
    visit:{entrant_price:Number($('price').value),entrant_message:$('message').value,channel:$('channel').value,standard_price:3.99,budget_price:2.99,scented_price:4.29},history:sample?.history||[]};
}
function render(result){
  const root=$('result');root.replaceChildren();const title=document.createElement('h3');title.textContent='Top choice: '+result.top_choice;root.append(title);
  Object.entries(result.probabilities).sort((a,b)=>b[1]-a[1]).forEach(([name,p])=>{const row=document.createElement('div');const text=document.createElement('div');text.className='prob mono';text.textContent=name+' · '+(100*p).toFixed(1)+'%';const bar=document.createElement('div');bar.className='bar';bar.style.width=(100*p)+'%';row.append(text,bar);root.append(row);});
  const explanation=document.createElement('p');explanation.textContent='Score contributions for '+result.top_choice+' versus '+result.runner_up+'. Positive terms favor the top choice; negative terms oppose it. These are log-odds contributions, not probability points or causal explanations.';root.append(explanation);
  const table=document.createElement('table');const head=document.createElement('tr');for(const text of ['Feature','Contribution']){const th=document.createElement('th');th.textContent=text;head.append(th);}table.append(head);
  result.contributions.forEach(t=>{const tr=document.createElement('tr');for(const v of [t.feature,(t.contribution>=0?'+':'')+t.contribution.toFixed(3)]){const td=document.createElement('td');td.textContent=v;tr.append(td);}table.append(tr);});root.append(table);
  const note=document.createElement('p');note.className='muted';note.textContent='Trust limit: the selected learned model had worse check-set log loss than the original starter. Small probability differences are not reliable business evidence.';root.append(note);
}
$('example').onchange=()=>{
  if($('example').value==='new'){$('history-note').textContent='New household: no purchase history.';return;}
  const s=bundle.examples[Number($('example').value)],h=s.household,v=s.visit;
  for(const [id,value] of Object.entries({price:v.entrant_price,message:v.entrant_message,channel:v.channel,budget:h.budget,usual:h.usual_brand,income:h.income_band,bathrooms:h.bathrooms,spend:h.cleaning_spend_usd}))$(id).value=value;
  $('history-note').textContent=s.household.household_id+' · visit '+v.visit_number+' · earlier choices: '+(s.history.map(r=>r.choice).join(', ')||'none recorded')+'. Current-only ignores this history.';
};
$('prediction-form').onsubmit=async event=>{
  event.preventDefault();if(!$('expectation').value.trim()){$('expectation').setCustomValidity('Write an expectation first.');$('expectation').reportValidity();return;}
  $('run-button').disabled=true;const data=input(),model=$('model').value==='with_history'?bundle.history_model:bundle.model;
  const row={id:'browser-'+crypto.randomUUID(),created_at:new Date().toISOString(),status:'planned',expectation:$('expectation').value.trim(),inputs:data,model_version:{source_run:bundle.source_run,model},data_version:bundle.data_version};
  let logged=false;
  try{await put(row);logged=true;row.outputs=ChoiceModel.predict(model,data.household,data.visit,data.history);row.status='completed';row.finished_at=new Date().toISOString();await put(row);render(row.outputs);$('status').textContent='Saved '+row.id;await refreshLocal();}
  catch(e){row.status='failed';row.error=String(e);if(logged){try{await put(row);}catch{}}$('status').textContent='Could not complete and save run: '+e.message;$('status').className='error';}
  finally{$('run-button').disabled=false;}
};
$('expectation').oninput=()=>$('expectation').setCustomValidity('');
$('saved-run').onchange=()=>{const r=savedRuns.find(r=>r.id===Number($('saved-run').value));if(r){const output=r.outputs?{...r.outputs}:null;if(output){delete output.predictions;delete output.fold_models;delete output.models;delete output.household_folds;} $('run-detail').textContent=JSON.stringify({...r,outputs:output},null,2)+'\n\nFull per-visit predictions and coefficients are in the downloads.';}};
$('local-run').onchange=async()=>{const r=(await all()).find(r=>r.id===$('local-run').value);if(r)$('run-detail').textContent=JSON.stringify(r,null,2);};
$('export-local').onclick=async()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(await all(),null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='browser-runs.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
(async()=>{
 try{
  const responses=await Promise.all([fetch('model.json'),fetch('runs.json')]);if(responses.some(r=>!r.ok))throw Error('Could not load saved artifacts');[bundle,savedRuns]=await Promise.all(responses.map(r=>r.json()));
  for(const [k,m]of Object.entries(bundle.metrics)){const tr=document.createElement('tr');for(const v of [labels[k],m.log_loss.toFixed(4),m.buy_log_loss.toFixed(4),(100*m.accuracy).toFixed(1)+'%']){const td=document.createElement('td');td.textContent=v;tr.append(td);}$('metrics').append(tr);}
  bundle.examples.forEach((s,i)=>$('example').add(new Option(s.household.household_id+' · visit '+s.visit.visit_number,i)));
  savedRuns.forEach(r=>$('saved-run').add(new Option('Run '+r.id+' · '+r.kind+' · '+r.status,r.id)));$('saved-run').onchange();
  db=await new Promise((resolve,reject)=>{const request=indexedDB.open('cleaner-choice-experiments',1);request.onupgradeneeded=()=>request.result.createObjectStore('runs',{keyPath:'id'});request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});
  await refreshLocal();$('status').textContent='Ready. Your expectation is saved before prediction.';$('run-button').disabled=false;
 }catch(e){$('status').textContent='Could not initialize demo: '+e.message;$('status').className='error';}
})();
