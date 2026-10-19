"""Make review files and a source ZIP without changing Git history."""
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

from experiments import read_run, versions

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'submission'


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare():
    OUT.mkdir(exist_ok=True)
    with closing(sqlite3.connect(ROOT / 'artifacts/runs.sqlite')) as source, closing(sqlite3.connect(OUT / 'runs.sqlite')) as target:
        source.backup(target)
    fields = ['data_split', 'run_id', 'model', 'log_loss', 'accuracy', 'buy_log_loss', 'brier']
    scores = []
    for run_id, split in [(2, 'build_household_cv'), (3, 'earlier_models_check_set'),
                          (5, 'build_nested_cv_reused'), (6, 'build_frequency_comparison_reused'),
                          (7, 'build_utility_comparison_reused')]:
        for model, values in read_run(run_id)['outputs']['metrics'].items():
            scores.append({'data_split': split, 'run_id': run_id, 'model': model,
                           **{key: values[key] for key in ['log_loss', 'accuracy', 'buy_log_loss', 'brier']}})
    with (ROOT / 'artifacts/scores.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scores)
    decision = read_run(8)['outputs']
    dump(ROOT / 'artifacts/decision_summary.json', {key: decision[key] for key in [
        'candidate', 'candidate_status', 'fold_winner_agreement', 'hypothetical_profit_winners', 'decision', 'limits']})
    dump(ROOT / 'artifacts/data_versions.json', versions())
    with closing(sqlite3.connect(ROOT / 'artifacts/runs.sqlite')) as connection:
        connection.row_factory = sqlite3.Row
        runs = [dict(row) for row in connection.execute('SELECT id,kind,status,created_at,finished_at,inputs FROM runs ORDER BY id')]
    with (ROOT / 'artifacts/run_index.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id','kind','parent_run','status','created_at','finished_at'])
        writer.writeheader()
        for row in runs:
            inputs = json.loads(row.pop('inputs'))
            writer.writerow({**row, 'parent_run': inputs.get('parent_run', '')})
    final = (ROOT / 'FINAL.md').read_bytes()
    (OUT / 'FINAL.md').write_bytes(final)
    (OUT / 'submission-message.txt').write_text(
        'Hi Rishika and Saksham,\n\n'
        'I have put together my work on the cleaner choice challenge. I started by testing whether purchase history helped, then tried regularization and a smaller model. I also added a bounded CPU research loop that saves its trials and stops when changes do not improve enough.\n\n'
        'Repository: https://github.com/srijabal/cleaner-choice-challenge\n'
        'Live link: [ADD VERIFIED LIVE LINK AND CONFIRM REVIEWER ACCESS BEFORE SENDING]\n'
        'Run locally: ./run.sh\n'
        'Database: attached runs.sqlite\n'
        'Final note: attached FINAL.md\n\n'
        'The provisional test candidate is $3.49/environment/online. The notes explain why I would not call it a proven launch or profit optimum.\n\n'
        'Warm regards,\nSrija\n')
    # Exclude local environments, Git metadata, credentials and this output folder.
    excluded = {'.git','.venv','node_modules','__pycache__','submission','.agents','.codex','.aws','.openai','.sites-runtime','.sites-checkout'}
    files = []
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in excluded for part in relative.parts) or path.is_symlink() or not path.is_file():
            continue
        if path.name in {'.DS_Store'} or path.suffix in {'.pyc','.pem','.key'} or path.name.startswith('.env') or path.name.endswith(('-wal','-shm','-journal')):
            continue
        files.append(path)
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    archive = OUT / 'cleaner-choice-submission.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(files):
            package.write(path, path.relative_to(ROOT))
        package.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    # Check every archived byte against the manifest and ensure required files exist.
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        for name, sha in manifest.items():
            assert hashlib.sha256(package.read(name)).hexdigest() == sha, name
        for name in ['FIRST.md','FINAL.md','RESULTS.md','run.sh','app.py','artifacts/runs.sqlite','demo/index.html']:
            assert name in package.namelist(), name
    dump(OUT / 'checksums.json', {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [archive, OUT / 'runs.sqlite', OUT / 'FINAL.md']})
    print(json.dumps({'archive': str(archive), 'files': len(files), 'size_bytes': archive.stat().st_size,
                      'final_note_words': len(final.decode().split()), 'saved_runs': len(runs),
                      'pending': ['final commit/push', 'working live link', 'reviewer access to private resources']}, indent=2))


if __name__ == '__main__':
    prepare()
