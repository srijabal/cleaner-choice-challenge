'use strict';
const $ = id => document.getElementById(id);
let bundle, db, runs = [], serverAvailable = false;
const names = {
  history_comparison: 'Does purchase history help?', frozen_check: 'Check earlier models on other households',
  scenario_sensitivity: 'Compare price, message and channel', regularization_nested_cv: 'Reduce overfitting',
  frequency_baseline: 'Compare with average choice rates', shared_utility_comparison: 'Try a simpler model',
  final_offer_sensitivity: 'Check the proposed offer and costs', bounded_research_loop: 'Run automatic research',
  research_loop_trial: 'Automatic research trial'
};
function text(parent, tag, content, className) {
  const e = document.createElement(tag); e.textContent = content; if (className) e.className = className; parent.append(e); return e;
}
function show(view) {
  document.querySelectorAll('[data-view]').forEach(button => {button.setAttribute('aria-selected', String(button.dataset.view === view)); button.tabIndex = button.dataset.view === view ? 0 : -1;});
  for (const name of ['offer', 'runs']) $('panel-' + name).hidden = name !== view;
}
document.querySelectorAll('[data-view]').forEach((button, index, list) => {
  button.onclick = () => show(button.dataset.view);
  button.tabIndex = index === 0 ? 0 : -1;
  button.onkeydown = event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? list.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + list.length) % list.length;
    show(list[next].dataset.view); list[next].focus();
  };
});
$('experiment-type').onchange = () => {
  const research = $('experiment-type').value === 'research';
  $('offer-controls').hidden = research; $('research-controls').hidden = !research;
};
function storeRun(run) {
  return new Promise((resolve, reject) => {const t = db.transaction('runs', 'readwrite'); t.objectStore('runs').put(run); t.oncomplete = resolve; t.onerror = () => reject(t.error);});
}
function browserRuns() {
  return new Promise((resolve, reject) => {const q = db.transaction('runs').objectStore('runs').getAll(); q.onsuccess = () => resolve(q.result); q.onerror = () => reject(q.error);});
}
function exportJSON(value, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function offerInput() {
  return {household: {household_id: 'custom', budget: $('budget').value, usual_brand: $('usual').value},
          visit: {entrant_price: Number($('price').value), entrant_message: $('message').value, channel: $('channel').value,
                  standard_price: 3.99, budget_price: 2.99, scented_price: 4.29},
          history: [], assumed_unit_cost: Number($('cost').value)};
}
function calculate(data) {
  const result = ChoiceModel.predict(bundle.model, data.household, data.visit, []);
  const average = bundle.population.reduce((sum, h) => sum + ChoiceModel.predict(bundle.model, h, data.visit, []).probabilities.entrant, 0) / bundle.population.length;
  result.population_estimate = {mean_entrant_probability: average, assumed_unit_cost: data.assumed_unit_cost,
    contribution_per_exposed_shopper: average * (data.visit.entrant_price - data.assumed_unit_cost),
    scope: '200 fictional build households; illustrative contribution excludes other channel costs.'};
  return result;
}
function renderOffer(result, target) {
  target.replaceChildren(); target.hidden = false;
  const p = result.population_estimate;
  text(target, 'h3', 'Result');
  const summary = document.createElement('table');
  for (const [label, value] of [
    ['Entrant purchase probability · 200 profiles', (p.mean_entrant_probability * 100).toFixed(1) + '%'],
    ['Contribution / exposure · assumed cost $' + p.assumed_unit_cost.toFixed(2), '$' + p.contribution_per_exposed_shopper.toFixed(3)]
  ]) {const row = document.createElement('tr'); text(row, 'td', label); text(row, 'td', value, 'num'); summary.append(row);}
  target.append(summary);
  const detail = document.createElement('details'); text(detail, 'summary', 'Household predictions and score contributions');
  const content = document.createElement('div'), table = document.createElement('table');
  Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).forEach(([choice, probability]) => {
    const row = document.createElement('tr'); text(row, 'td', choice === 'none' ? 'Buy nothing' : choice);
    text(row, 'td', (probability * 100).toFixed(1) + '%', 'num'); table.append(row);
  }); content.append(table);
  text(content, 'h3', result.top_choice + ' vs ' + result.runner_up);
  const terms = document.createElement('table'); result.contributions.forEach(t => {
    const row = document.createElement('tr'); text(row, 'td', t.feature); text(row, 'td', t.contribution.toFixed(3), 'num'); terms.append(row);
  }); content.append(terms); detail.append(content); target.append(detail);
}
$('offer-form').onsubmit = async event => {
  event.preventDefault(); const expectation = $('expectation').value.trim();
  if (!expectation) {$('expectation').setCustomValidity('Write what you expect first.'); $('expectation').reportValidity(); return;}
  $('predict-button').disabled = true;
  const run = {id: 'browser-' + crypto.randomUUID(), kind: 'offer_prediction', created_at: new Date().toISOString(),
    status: 'planned', expectation, inputs: offerInput(), model_version: {source_run: bundle.source_run, model: bundle.model}, data_version: bundle.data_version};
  let saved = false;
  try {
    await storeRun(run); saved = true; run.outputs = calculate(run.inputs); run.status = 'completed'; run.finished_at = new Date().toISOString();
    await storeRun(run); renderOffer(run.outputs, $('offer-result')); $('offer-status').textContent = 'Saved as ' + run.id;
  } catch (error) {
    run.status = 'failed'; run.error = String(error); if (saved) {try {await storeRun(run);} catch {}}
    $('offer-status').textContent = 'Could not save the experiment: ' + error.message; $('offer-status').className = 'error';
  } finally {$('predict-button').disabled = false;}
};
$('expectation').oninput = () => $('expectation').setCustomValidity('');
function describe(run) {
  if (run.status === 'failed') return 'Failed — retained';
  if (run.status !== 'completed') return run.status;
  const output = run.outputs || run.summary || {};
  if (run.kind === 'bounded_research_loop') return output.decision === 'keep_existing_model' ? 'Stopped; kept the existing model' : 'Candidate ready for review';
  return {history_comparison: 'No clear benefit from history', frozen_check: 'Starter still competitive',
    scenario_sensitivity: 'An offer candidate, not proof', regularization_nested_cv: 'Smaller effects helped',
    frequency_baseline: 'Inputs help distinguish cleaners', shared_utility_comparison: 'Simpler model selected',
    final_offer_sensitivity: 'Costs change the price choice'}[run.kind] || 'Completed';
}
async function refreshRuns() {
  try {
    const response = await fetch('api/runs', {cache: 'no-store'}); if (!response.ok) throw Error('No local API');
    runs = await response.json(); serverAvailable = true;
  } catch {
    const response = await fetch('runs.json'); if (!response.ok) throw Error('Could not load experiments'); runs = await response.json(); serverAvailable = false;
  }
  const list = $('run-list'); list.replaceChildren();
  for (const run of runs.filter(r => r.kind !== 'research_loop_trial')) {
    const row = document.createElement('tr'), id = document.createElement('td'), button = document.createElement('button');
    button.type = 'button'; button.className = 'link'; button.textContent = '#' + run.id; button.onclick = () => inspectRun(run.id); id.append(button); row.append(id);
    text(row, 'td', names[run.kind] || run.kind); text(row, 'td', describe(run)); list.append(row);
  }
  if (db) for (const run of (await browserRuns()).sort((a, b) => a.created_at.localeCompare(b.created_at)).slice(-5)) {
    const row = document.createElement('tr'), id = document.createElement('td'), button = document.createElement('button');
    button.className = 'link'; button.type = 'button'; button.textContent = run.id.slice(-8); button.onclick = () => inspectBrowserRun(run); id.append(button); row.append(id);
    text(row, 'td', 'Your offer experiment'); text(row, 'td', run.status); list.append(row);
  }
  $('database-download').href = serverAvailable ? 'api/database' : 'runs.sqlite';
  $('research-button').disabled = !serverAvailable;
  if (!serverAvailable) $('research-status').textContent = 'Run locally with python3 app.py to enable this. CLI: python3 research_loop.py';
  return runs;
}
async function inspectRun(id) {
  const target = $('run-detail'); target.replaceChildren(); target.hidden = false;
  try {
    let run;
    if (serverAvailable) {const response = await fetch('api/runs/' + id); if (!response.ok) throw Error('Run not found'); run = await response.json();}
    else run = runs.find(r => r.id === Number(id));
    if (!run) throw Error('Run not found');
    text(target, 'h3', 'Run #' + run.id + ' · ' + (names[run.kind] || run.kind));
    text(target, 'p', describe(run)); text(target, 'p', 'Expected: ' + run.expectation);
    if (run.outputs?.loop_minus_incumbent) {
      text(target, 'p', 'Prediction loss: ' + run.outputs.metrics.log_loss.toFixed(4) + '. Trial IDs: ' + run.outputs.trial_ids.join(', ') + '.');
      text(target, 'p', 'The loop does not change the final model automatically. Reused build data cannot prove a new real-world winner.', 'muted');
    }
    const raw = document.createElement('details'); text(raw, 'summary', 'Full inputs, outputs and versions'); text(raw, 'pre', JSON.stringify(run, null, 2)); target.append(raw);
    const download = document.createElement('button'); download.type = 'button'; download.className = 'link'; download.textContent = 'Download this run'; download.onclick = () => exportJSON(run, 'run-' + run.id + '.json'); target.append(download);
  } catch (error) {text(target, 'p', error.message, 'error');}
}
function inspectBrowserRun(run) {
  const target = $('run-detail'); target.replaceChildren(); target.hidden = false;
  text(target, 'h3', 'Your offer run'); text(target, 'p', 'Expected: ' + run.expectation);
  if (run.outputs?.population_estimate?.contribution_per_exposed_shopper !== undefined) {const result = document.createElement('div'); renderOffer(run.outputs, result); target.append(result);}
  const detail = document.createElement('details'); text(detail, 'summary', 'Full saved run'); text(detail, 'pre', JSON.stringify(run, null, 2)); target.append(detail);
}
$('lookup-form').onsubmit = event => {event.preventDefault(); inspectRun(Number($('lookup-id').value));};
$('export-browser').onclick = async () => exportJSON(await browserRuns(), 'my-offer-runs.json');
let pollTimer;
async function pollResearch() {
  try {
    const response = await fetch('api/research', {cache: 'no-store'}); if (!response.ok) return;
    const state = await response.json();
    if (state.status === 'running') {$('research-button').disabled = true; $('research-status').textContent = 'Running bounded model tests…'; pollTimer = setTimeout(pollResearch, 1200);}
    else if (state.status === 'completed') {
      await refreshRuns(); $('research-status').textContent = 'Finished. ' + (state.summary.decision === 'keep_existing_model' ? 'No clear improvement; kept the existing model.' : 'Candidate ready for review.') + ' Run #' + state.summary.run_id + '.';
      $('research-button').disabled = false;
    } else if (state.status === 'failed') {$('research-status').textContent = 'Research failed: ' + state.error; $('research-button').disabled = false; await refreshRuns();}
    else $('research-status').textContent = 'Ready';
  } catch (error) {$('research-status').textContent = 'Could not check progress. Reopen Experiments to inspect saved runs.'; $('research-button').disabled = false;}
}
$('research-form').onsubmit = async event => {
  event.preventDefault(); $('research-button').disabled = true;
  try {
    const response = await fetch('api/research', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({expectation: $('research-expectation').value.trim()})});
    const body = await response.json(); if (!response.ok) throw Error(body.error); clearTimeout(pollTimer); await pollResearch();
  } catch (error) {$('research-status').textContent = error.message; $('research-button').disabled = false;}
};
document.querySelector('[data-view="runs"]').addEventListener('click', () => {refreshRuns().catch(error => {$('research-status').textContent = error.message;});});
(async () => {
  try {
    const response = await fetch('model.json'); if (!response.ok) throw Error('Could not load model'); bundle = await response.json();
    db = await new Promise((resolve, reject) => {const q = indexedDB.open('cleaner-choice-experiments', 1); q.onupgradeneeded = () => q.result.createObjectStore('runs', {keyPath: 'id'}); q.onsuccess = () => resolve(q.result); q.onerror = () => reject(q.error);});
    $('predict-button').disabled = false; $('offer-status').textContent = 'Ready';
    await refreshRuns(); if (serverAvailable) await pollResearch();
  } catch (error) {$('offer-status').textContent = error.message + '. Refresh to try again.'; $('offer-status').className = 'error';}
})();
