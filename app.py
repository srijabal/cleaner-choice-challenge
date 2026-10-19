"""Local demo and bounded CPU experiment endpoint: python3 app.py."""
from contextlib import closing
import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'artifacts' / 'runs.sqlite'
job_lock = threading.Lock()
job = {'status': 'idle'}


def execute_research(expectation):
    global job
    try:
        result = subprocess.run([sys.executable, str(ROOT / 'research_loop.py'), '--expectation', expectation],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:] or 'Research process failed.')
        summary = json.loads(result.stdout)
        with job_lock:
            job = {'status': 'completed', 'summary': summary}
    except Exception as error:
        with job_lock:
            job = {'status': 'failed', 'error': str(error)}


def database_rows(run_id=None):
    with closing(sqlite3.connect(DB)) as connection:
        connection.row_factory = sqlite3.Row
        if run_id is None:
            rows = connection.execute('SELECT * FROM runs ORDER BY id').fetchall()
        else:
            rows = connection.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        for key in ['inputs', 'outputs', 'model_version', 'data_version']:
            if item[key]:
                item[key] = json.loads(item[key])
        if run_id is None:
            output = item.get('outputs') or {}
            item = {key: item[key] for key in ['id', 'kind', 'status', 'expectation', 'created_at', 'finished_at', 'inputs']}
            item['summary'] = {key: output[key] for key in ['metrics', 'decision', 'outcome', 'loop_minus_incumbent', 'error'] if key in output}
        results.append(item)
    return results


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'demo'), **kwargs)

    def send_json(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/runs':
            return self.send_json(database_rows())
        if path.startswith('/api/runs/'):
            try:
                rows = database_rows(int(path.rsplit('/', 1)[1]))
            except ValueError:
                return self.send_json({'error': 'Use a numeric run ID.'}, 400)
            return self.send_json(rows[0] if rows else {'error': 'Run not found.'}, 200 if rows else 404)
        if path == '/api/research':
            with job_lock:
                return self.send_json(job.copy())
        if path == '/api/database':
            # Take a consistent snapshot even if a research trial is writing.
            with tempfile.TemporaryDirectory() as folder:
                snapshot = Path(folder) / 'runs.sqlite'
                with closing(sqlite3.connect(DB)) as source, closing(sqlite3.connect(snapshot)) as target:
                    source.backup(target)
                data = snapshot.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/vnd.sqlite3')
            self.send_header('Content-Disposition', 'attachment; filename="runs.sqlite"')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.startswith('/api/'):
            return self.send_json({'error': 'Endpoint not found.'}, 404)
        return super().do_GET()

    def do_POST(self):
        global job
        if urlparse(self.path).path != '/api/research':
            return self.send_json({'error': 'Endpoint not found.'}, 404)
        origin = self.headers.get('Origin')
        allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        if origin and origin not in allowed:
            return self.send_json({'error': 'Start research from the local demo.'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Request must be between 1 and 4096 bytes.')
            body = json.loads(self.rfile.read(length))
            expectation = body.get('expectation', '').strip()
            if len(expectation) < 5 or len(expectation) > 1000:
                raise ValueError('Write an expectation between 5 and 1000 characters.')
        except (ValueError, TypeError, AttributeError) as error:
            return self.send_json({'error': str(error)}, 400)
        with job_lock:
            if job['status'] == 'running':
                return self.send_json({'error': 'A research loop is already running.'}, 409)
            job = {'status': 'running'}
        threading.Thread(target=execute_research, args=(expectation,), daemon=True).start()
        return self.send_json({'status': 'running'}, 202)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Cleaner choice demo: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
