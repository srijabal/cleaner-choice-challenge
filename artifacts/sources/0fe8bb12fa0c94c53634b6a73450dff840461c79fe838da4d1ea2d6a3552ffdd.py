"""Small multinomial logistic model; features only use information before choice."""
import math
from collections import defaultdict
from run_baseline import read_rows, offers_for

CHOICES = ['budget', 'entrant', 'none', 'scented', 'standard']
PRODUCTS = ['entrant', 'standard', 'budget', 'scented']
CURRENT_FEATURES = ([f'price:{p}' for p in PRODUCTS]
    + ['message:cleaning', 'message:low_odor', 'message:environment', 'message:missing', 'channel:online']
    + [f'usual:{p}' for p in PRODUCTS] + ['usual:missing', 'budget:tight', 'budget:flexible',
       'income:under_50k', 'income:over_100k', 'bathrooms', 'monthly_spend', 'monthly_spend:missing'])
HISTORY_FEATURES = ['history:available'] + [f'previous:{p}' for p in CHOICES] + ['history:ever_entrant', 'history:visits_since_purchase']


def features(household, visit, history, use_history=False):
    values = {f'price:{p}': float(visit[f'{p}_price']) - 4.0 for p in PRODUCTS}
    message = visit.get('entrant_message') or 'missing'
    values.update({f'message:{m}': float(message == m) for m in ['cleaning','low_odor','environment','missing']})
    values['channel:online'] = float(visit['channel'] == 'online')
    usual = household.get('usual_brand') or 'missing'
    values.update({f'usual:{p}': float(usual == p) for p in PRODUCTS + ['missing']})
    values.update({f'budget:{b}': float(household['budget'] == b) for b in ['tight','flexible']})
    values.update({f'income:{b}': float(household['income_band'] == b) for b in ['under_50k','over_100k']})
    values['bathrooms'] = float(household['bathrooms']) - 1
    values['monthly_spend'] = float(household['cleaning_spend_usd'] or 0) / 20
    values['monthly_spend:missing'] = float(not household['cleaning_spend_usd'])
    if use_history:
        values['history:available'] = float(bool(history))
        values.update({f'previous:{p}': float(bool(history) and history[-1]['choice'] == p) for p in CHOICES})
        values['history:ever_entrant'] = float(any(r['choice'] == 'entrant' for r in history))
        gap = 0
        for past in reversed(history):
            if past['choice'] != 'none':
                break
            gap += 1
        # Visits, not elapsed days or inferred bottle inventory. Zero without history.
        values['history:visits_since_purchase'] = gap / 6
    names = CURRENT_FEATURES + (HISTORY_FEATURES if use_history else [])
    return [values[n] for n in names]


def examples(filename):
    households = {r['household_id']: r for r in read_rows('households.csv')}
    visits = sorted(read_rows(filename), key=lambda r: (r['household_id'], int(r['visit_number'])))
    histories = defaultdict(list)
    result = []
    for visit in visits:
        hid = visit['household_id']
        result.append({'household': households[hid], 'visit': visit, 'history': list(histories[hid])})
        histories[hid].append(visit)
    return result


def fit(rows, use_history=False, C=1.0):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=C, solver='lbfgs', max_iter=2000, random_state=42)
    model.fit([features(r['household'],r['visit'],r['history'],use_history) for r in rows],
              [r['visit']['choice'] for r in rows])
    if max(model.n_iter_) >= 2000:
        raise RuntimeError('Model did not converge; do not report this fit as successful.')
    return {'type':'multinomial_logistic', 'version':'logistic-v1', 'history':use_history,
            'feature_names': CURRENT_FEATURES + (HISTORY_FEATURES if use_history else []),
            'classes':model.classes_.tolist(), 'coefficients':model.coef_.tolist(),
            'intercepts':model.intercept_.tolist(), 'C':C, 'sklearn_version':sklearn.__version__,
            'iterations':int(max(model.n_iter_))}


def predict(model, household, visit, history):
    x = features(household, visit, history, model['history'])
    scores = [b + sum(w*v for w,v in zip(ws,x)) for b,ws in zip(model['intercepts'], model['coefficients'])]
    weights = [math.exp(s-max(scores)) for s in scores]
    probs = {k:w/sum(weights) for k,w in zip(model['classes'],weights)}
    top, runner = sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)[:2]
    terms = [{'feature':'intercept', 'value':1, 'contribution':model['intercepts'][top]-model['intercepts'][runner]}]
    terms += [{'feature':n,'value':v,'contribution':(model['coefficients'][top][j]-model['coefficients'][runner][j])*v}
              for j,(n,v) in enumerate(zip(model['feature_names'],x)) if v != 0]
    return {'probabilities':probs,'top_choice':model['classes'][top], 'runner_up':model['classes'][runner],
            'log_odds_margin':scores[top]-scores[runner],
            'contributions':sorted(terms,key=lambda t:abs(t['contribution']),reverse=True),
            'explanation_limit':'These terms explain the model score, not the household’s true motivation.'}


def score_rows(model, rows):
    from baseline import predict as starter_predict
    records=[]
    for r in rows:
        h,v,hist=r['household'],r['visit'],r['history']
        result = starter_predict(h,offers_for(v),v,hist) if model == 'starter' else predict(model,h,v,hist)
        probs=result['probabilities']; choice=v['choice']
        pnone=probs['none']; buy=choice!='none'
        records.append({'household_id':v['household_id'],'visit_id':v['visit_id'],'visit_number':int(v['visit_number']),
            'choice':choice,'budget':h['budget'],'channel':v['channel'],'probabilities':probs,
            'loss':-math.log(max(probs[choice],1e-15)),
            'buy_loss':-math.log(max(1-pnone if buy else pnone,1e-15)),
            'brand_loss':-math.log(max(probs[choice]/(1-pnone),1e-15)) if buy else None,
            'brier':sum((probs[p]-float(p==choice))**2 for p in CHOICES),
            'correct':max(probs,key=probs.get)==choice})
    return records


def metrics(records):
    def mean(xs): return sum(xs)/len(xs) if xs else None
    out={'n':len(records),'log_loss':mean([r['loss'] for r in records]),
         'buy_log_loss':mean([r['buy_loss'] for r in records]),
         'brand_log_loss_given_purchase':mean([r['brand_loss'] for r in records if r['brand_loss'] is not None]),
         'brier':mean([r['brier'] for r in records]),'accuracy':mean([r['correct'] for r in records]),
         'entrant_observed':mean([r['choice']=='entrant' for r in records]),
         'entrant_predicted':mean([r['probabilities']['entrant'] for r in records])}
    out['entrant_calibration']=[]
    for lo,hi in [(0,.1),(.1,.2),(.2,.3),(.3,.5),(.5,1.000001)]:
        group=[r for r in records if lo<=r['probabilities']['entrant']<hi]
        out['entrant_calibration'].append({'range':[lo,min(hi,1)],'n':len(group),
              'predicted':mean([r['probabilities']['entrant'] for r in group]),
              'observed':mean([r['choice']=='entrant' for r in group])})
    return out
