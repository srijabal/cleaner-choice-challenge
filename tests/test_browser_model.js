/* Check the exported browser predictor against saved Python calculations. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.join(__dirname, '..');
const {predict} = require(path.join(root, 'demo/model.js'));
const bundle = JSON.parse(fs.readFileSync(path.join(root, 'demo/model.json'), 'utf8'));
for (const test of bundle.parity_cases) {
  const {household, visit, history} = test.input;
  const actual = predict(bundle.model, household, visit, history);
  assert.equal(actual.top_choice, test.expected.top_choice);
  for (const choice of bundle.model.classes) {
    assert.ok(Math.abs(actual.probabilities[choice] - test.expected.probabilities[choice]) < 1e-12);
  }
  const sum = actual.contributions.reduce((value, term) => value + term.contribution, 0);
  assert.ok(Math.abs(sum - actual.log_odds_margin) < 1e-12);
  for (const product of ['entrant', 'standard', 'budget', 'scented']) {
    const higher = {...visit, [product + '_price']: Number(visit[product + '_price']) + 1};
    assert.ok(predict(bundle.model, household, higher, history).probabilities[product] <= actual.probabilities[product] + 1e-12);
  }
}
console.log(`Passed Python/browser parity, explanations and price checks for ${bundle.parity_cases.length} cases.`);
