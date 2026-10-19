(function(root){
'use strict';
function utilityRows(model,h,v,history){
  return model.classes.map(choice=>{
    const f=Object.fromEntries(model.feature_names.map(n=>[n,0]));
    if(choice!=='none'){
      const price=Number(v[choice+'_price']);
      if(!Number.isFinite(price)||price<=0)throw Error('Prices must be positive finite numbers.');
      f['product:'+choice]=1;f.price_penalty=-price;
      f.tight_budget_price_penalty=h.budget==='tight'?-price:0;
      f.usual_brand=Number(h.usual_brand===choice);
    }
    if(choice==='entrant'){
      for(const message of ['cleaning','low_odor','environment'])f['entrant_message:'+message]=Number(v.entrant_message===message);
      f['entrant_channel:online']=Number(v.channel==='online');
    }
    if(model.history&&history.length){
      const last=history[history.length-1].choice;
      f['history:repeat_product']=Number(choice!=='none'&&choice===last);
      f['history:purchase_before_none']=Number(choice==='none'&&last!=='none');
    }
    return model.feature_names.map(n=>f[n]);
  });
}
function utilityPredict(model,h,v,history){
  const matrix=utilityRows(model,h,v,history);
  const scores=matrix.map(x=>x.reduce((s,v,j)=>s+v*model.weights[j],0));
  const peak=Math.max(...scores),weights=scores.map(s=>Math.exp(s-peak)),total=weights.reduce((a,b)=>a+b,0);
  const probabilities=Object.fromEntries(model.classes.map((p,i)=>[p,weights[i]/total]));
  const [top,runner]=scores.map((_,i)=>i).sort((a,b)=>scores[b]-scores[a]);
  const contributions=model.feature_names.map((feature,j)=>({feature,value:matrix[top][j]-matrix[runner][j],coefficient:model.weights[j],contribution:(matrix[top][j]-matrix[runner][j])*model.weights[j]})).filter(t=>t.value!==0).sort((a,b)=>Math.abs(b.contribution)-Math.abs(a.contribution));
  return {probabilities,top_choice:model.classes[top],runner_up:model.classes[runner],log_odds_margin:scores[top]-scores[runner],contributions};
}
function predict(model,h,v,history=[]){
  if(model.type!=='shared_utility')throw Error('Unsupported model type.');
  return utilityPredict(model,h,v,history);
}
root.ChoiceModel={predict};
if(typeof module!=='undefined')module.exports=root.ChoiceModel;
})(typeof globalThis!=='undefined'?globalThis:this);
