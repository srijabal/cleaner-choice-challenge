"""Exploratory model scenarios, not causal estimates of message or channel effects."""
from collections import defaultdict
import itertools
import traceback
from choice_model import examples, predict
from baseline import predict as starter_predict
from run_baseline import offers_for
from experiments import register, begin, finish, read_run


def run_scenarios():
    run_id=register('scenario_sensitivity',
        'The cheapest entrant price may maximize modeled purchase share, but message/channel rankings may disagree across starter and learned models. I do not expect this to establish a real-world optimum or profitability. Use only observed price/message/channel levels, fixed competitor prices, and first visits to avoid replaying incompatible counterfactual histories.',
        {'model_run':3,'model':'current_only','prices':[3.49,4.49,5.49],
         'messages':['cleaning','low_odor','environment'],'channels':['retail','online'],
         'competitor_prices':{'standard':3.99,'budget':2.99,'scented':4.29},
         'population':'200 build households at first visit','purpose':'sensitivity only; no retraining or new check evaluation'})
    begin(run_id)
    try:
        base=[r for r in examples('visits.csv') if int(r['visit']['visit_number'])==1]
        visits=examples('visits.csv')
        trained=read_run(3)['outputs']['models']['current_only']
        fold_models=[r['models']['current_only'] for r in read_run(2)['outputs']['fold_models']]
        scenarios=[]
        for price,message,channel in itertools.product([3.49,4.49,5.49],['cleaning','low_odor','environment'],['retail','online']):
            predictions=defaultdict(list)
            for r in base:
                v={**r['visit'],'entrant_price':price,'entrant_message':message,'channel':channel,
                   'standard_price':3.99,'budget_price':2.99,'scented_price':4.29}
                predictions['current_only'].append(predict(trained,r['household'],v,[])['probabilities']['entrant'])
                predictions['starter'].append(starter_predict(r['household'],offers_for(v),v,[])['probabilities']['entrant'])
                for i,m in enumerate(fold_models):predictions[f'fold_{i}'].append(predict(m,r['household'],v,[])['probabilities']['entrant'])
            support=[r for r in visits if float(r['visit']['entrant_price'])==price and r['visit']['entrant_message']==message and r['visit']['channel']==channel]
            scenarios.append({'price':price,'message':message,'channel':channel,
                'predicted_share':{k:sum(v)/len(v) for k,v in predictions.items()},
                'observed_visits_in_cell':len(support),'observed_households_in_cell':len({r['visit']['household_id'] for r in support}),
                'observed_share_in_cell':sum(r['visit']['choice']=='entrant' for r in support)/len(support) if support else None})
        winners={k:max(scenarios,key=lambda s:s['predicted_share'][k]) for k in scenarios[0]['predicted_share']}
        finish(run_id,{'scenarios':scenarios,'winners':winners,'warning':'Observed cells are associations with varying competitors and repeated households. Scenario predictions hold competitors fixed. Neither is an identified treatment effect. No costs or margins supplied.'})
        print('Scenario run',run_id)
        for key,s in winners.items():print(key,s['price'],s['message'],s['channel'],round(s['predicted_share'][key],4))
    except Exception:
        finish(run_id,{'error':traceback.format_exc()},'failed');raise

if __name__=='__main__':run_scenarios()
