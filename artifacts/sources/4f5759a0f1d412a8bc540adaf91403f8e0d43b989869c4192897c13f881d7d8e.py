"""A small choice model with shared, non-increasing price effects.

Each cleaner gets a product intercept, a common price penalty, a possible extra
price penalty for tight budgets, and a usual-brand term. Only entrant scores use
its message and channel. No-purchase is the reference score. History is optional.
"""
import math
import numpy as np
from scipy.optimize import minimize
from choice_model import CHOICES, PRODUCTS

BASE_NAMES = [f'product:{p}' for p in PRODUCTS] + [
    'price_penalty', 'tight_budget_price_penalty', 'usual_brand',
    'entrant_message:cleaning', 'entrant_message:low_odor',
    'entrant_message:environment', 'entrant_channel:online',
]
HISTORY_NAMES = ['history:repeat_product', 'history:purchase_before_none']


def feature_matrix(household, visit, history, use_history=False):
    names = BASE_NAMES + (HISTORY_NAMES if use_history else [])
    rows = []
    for choice in CHOICES:
        f = dict.fromkeys(names, 0.0)
        if choice != 'none':
            price = float(visit[f'{choice}_price'])
            if not math.isfinite(price) or price <= 0:
                raise ValueError('Prices must be positive finite numbers.')
            f[f'product:{choice}'] = 1.0
            f['price_penalty'] = -price
            f['tight_budget_price_penalty'] = -price if household['budget'] == 'tight' else 0.0
            f['usual_brand'] = float(household.get('usual_brand') == choice)
        if choice == 'entrant':
            for message in ['cleaning', 'low_odor', 'environment']:
                f[f'entrant_message:{message}'] = float(visit.get('entrant_message') == message)
            f['entrant_channel:online'] = float(visit['channel'] == 'online')
        if use_history and history:
            last = history[-1]['choice']
            f['history:repeat_product'] = float(choice != 'none' and last == choice)
            f['history:purchase_before_none'] = float(choice == 'none' and last != 'none')
        rows.append([f[name] for name in names])
    return np.array(rows, dtype=float)


def fit(rows, use_history=False, penalty=0.01):
    x = np.array([feature_matrix(r['household'], r['visit'], r['history'], use_history) for r in rows])
    labels = np.array([CHOICES.index(r['visit']['choice']) for r in rows])
    names = BASE_NAMES + (HISTORY_NAMES if use_history else [])
    penalized = np.array([not name.startswith('product:') for name in names], dtype=float)

    def objective(weights):
        scores = x @ weights
        peak = scores.max(axis=1, keepdims=True)
        exponents = np.exp(scores - peak)
        probabilities = exponents / exponents.sum(axis=1, keepdims=True)
        log_normalizer = peak[:, 0] + np.log(exponents.sum(axis=1))
        loss = np.mean(log_normalizer - scores[np.arange(len(rows)), labels])
        residual = probabilities.copy()
        residual[np.arange(len(rows)), labels] -= 1.0
        gradient = np.einsum('nc,ncf->f', residual, x) / len(rows)
        return (loss + penalty * np.sum(penalized * weights**2) / 2,
                gradient + penalty * penalized * weights)

    bounds = [(0, None) if 'price_penalty' in name else (None, None) for name in names]
    result = minimize(objective, np.zeros(len(names)), method='L-BFGS-B', jac=True,
                      bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-7})
    if not result.success:
        raise RuntimeError(f'Utility model failed to converge: {result.message}')
    return {'type': 'shared_utility', 'version': 'utility-v1', 'history': use_history,
            'feature_names': names, 'classes': CHOICES, 'weights': result.x.tolist(),
            'penalty': penalty, 'iterations': int(result.nit),
            'training_visits': len(rows), 'parameter_count': len(names),
            'structural_assumptions': [
                'A higher own price cannot increase a cleaner’s modeled purchase probability.',
                'Price sensitivity is shared across cleaners; tight budgets may add a penalty.',
                'Only the entrant’s score responds to its message and online channel.',
                'Product utilities are additive; substitution follows multinomial logit assumptions.',
                'These constraints are modeling assumptions, not established causal findings.']}


def predict(model, household, visit, history):
    matrix = feature_matrix(household, visit, history, model['history'])
    weights = np.asarray(model['weights'])
    scores = matrix @ weights
    exponents = np.exp(scores - scores.max())
    probabilities = exponents / exponents.sum()
    ranking = sorted(range(len(CHOICES)), key=lambda i: float(scores[i]), reverse=True)
    top, runner = ranking[:2]
    terms = []
    for j, name in enumerate(model['feature_names']):
        difference = float(matrix[top, j] - matrix[runner, j])
        if difference:
            terms.append({'feature': name, 'value': difference,
                          'coefficient': float(weights[j]),
                          'contribution': float(weights[j] * difference)})
    return {'probabilities': dict(zip(CHOICES, probabilities.tolist())),
            'top_choice': CHOICES[top], 'runner_up': CHOICES[runner],
            'log_odds_margin': float(scores[top] - scores[runner]),
            'contributions': sorted(terms, key=lambda t: abs(t['contribution']), reverse=True),
            'explanation_limit': 'Contributions explain model scores, not a household’s true motivation.'}
