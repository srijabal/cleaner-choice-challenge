(function(root){
'use strict';
const products=['entrant','standard','budget','scented'];
const choices=['budget','entrant','none','scented','standard'];
function features(h,v,history,useHistory){
  const f={};
  for(const p of products) f['price:'+p]=Number(v[p+'_price'])-4;
  for(const m of ['cleaning','low_odor','environment','missing']) f['message:'+m]=Number((v.entrant_message||'missing')===m);
  f['channel:online']=Number(v.channel==='online');
  for(const p of [...products,'missing']) f['usual:'+p]=Number((h.usual_brand||'missing')===p);
  for(const b of ['tight','flexible']) f['budget:'+b]=Number(h.budget===b);
  for(const b of ['under_50k','over_100k']) f['income:'+b]=Number(h.income_band===b);
  f.bathrooms=Number(h.bathrooms)-1;f.monthly_spend=Number(h.cleaning_spend_usd||0)/20;
  f['monthly_spend:missing']=Number(h.cleaning_spend_usd==='');
  if(useHistory){
    f['history:available']=Number(history.length>0);
    for(const p of choices) f['previous:'+p]=Number(history.length>0&&history[history.length-1].choice===p);
    f['history:ever_entrant']=Number(history.some(r=>r.choice==='entrant'));
    let gap=0;for(let i=history.length-1;i>=0;i--){if(history[i].choice!=='none')break;gap++;}
    f['history:visits_since_purchase']=gap/6;
  }return f;
}
function predict(model,h,v,history=[]){
  const f=features(h,v,history,model.history),x=model.feature_names.map(n=>f[n]);
  if(x.some(n=>!Number.isFinite(n)))throw Error('Inputs must contain finite numbers.');
  const scores=model.coefficients.map((ws,i)=>model.intercepts[i]+ws.reduce((s,w,j)=>s+w*x[j],0));
  const peak=Math.max(...scores),weights=scores.map(s=>Math.exp(s-peak)),total=weights.reduce((a,b)=>a+b,0);
  const probs=Object.fromEntries(model.classes.map((p,i)=>[p,weights[i]/total]));
  const [top,runner]=scores.map((_,i)=>i).sort((a,b)=>scores[b]-scores[a]);
  const terms=[{feature:'intercept',value:1,contribution:model.intercepts[top]-model.intercepts[runner]}];
  model.feature_names.forEach((n,j)=>{if(x[j]!==0)terms.push({feature:n,value:x[j],contribution:(model.coefficients[top][j]-model.coefficients[runner][j])*x[j]});});
  return {probabilities:probs,top_choice:model.classes[top],runner_up:model.classes[runner],log_odds_margin:scores[top]-scores[runner],contributions:terms.sort((a,b)=>Math.abs(b.contribution)-Math.abs(a.contribution))};
}
root.ChoiceModel={features,predict};
if(typeof module!=='undefined')module.exports=root.ChoiceModel;
})(typeof globalThis!=='undefined'?globalThis:this);
