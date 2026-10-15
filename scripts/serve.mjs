/**
 * Development / preview server.
 *
 *   node scripts/serve.mjs [port]
 *   npm run serve
 *
 * It serves two things from one origin:
 *   - the app in public/ (index.html, style.css, app.js, lane.js)
 *   - the shared library in src/ (imported directly by the browser workers, so
 *     the code running in your browser is byte for byte the code the tests and
 *     the Node trainer use)
 *
 * plus a small API for champion checkpoints under runs/.
 *
 * No dependencies: node:http and friends only. Bind address is 0.0.0.0 so the
 * page is reachable from a preview proxy or another machine on the network.
 */
import { createServer } from 'node:http';
import { readFile, readdir, stat, mkdir, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { extname, join, resolve, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const publicDir = join(root, 'public');
const runsDir = join(root, 'runs');
const port = Number(process.argv[2] ?? process.env.PORT ?? 8787);
const host = process.env.HOST ?? '0.0.0.0';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
};

/** Resolve a URL path to a file inside the repo, or null if it escapes. */
function resolvePath(urlPath) {
  const clean = decodeURIComponent(urlPath.split('?')[0]);
  if (clean === '/' || clean === '') return join(publicDir, 'index.html');
  // /src/* and /runs/* are served from the repository, everything else from public/.
  const base = clean.startsWith('/src/') || clean.startsWith('/runs/') ? root : publicDir;
  const rel = clean.startsWith('/src/') || clean.startsWith('/runs/') ? clean.slice(1) : clean.slice(1);
  const full = normalize(join(base, rel));
  if (!full.startsWith(root)) return null;
  return full;
}

async function sendFile(res, path) {
  try {
    const info = await stat(path);
    if (info.isDirectory()) return sendFile(res, join(path, 'index.html'));
    res.writeHead(200, {
      'content-type': MIME[extname(path).toLowerCase()] || 'application/octet-stream',
      'content-length': info.size,
      'cache-control': 'no-cache',
      // The browser workers are module scripts, so nothing else is needed.
      'access-control-allow-origin': '*',
    });
    createReadStream(path).pipe(res);
    return true;
  } catch {
    return false;
  }
}

async function sendJSON(res, status, body) {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'access-control-allow-origin': '*' });
  res.end(text);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host ?? 'localhost'}`);

  // ---- API --------------------------------------------------------------
  if (url.pathname.startsWith('/api/')) {
    try {
      if (url.pathname === '/api/health') {
        return sendJSON(res, 200, { ok: true, node: process.version, platform: process.platform, cpus: (await import('node:os')).cpus().length });
      }
      if (url.pathname === '/api/champions' && req.method === 'GET') {
        const out = [];
        try {
          for (const dir of await readdir(runsDir)) {
            const full = join(runsDir, dir);
            if (!(await stat(full)).isDirectory()) continue;
            for (const file of await readdir(full)) {
              if (!file.endsWith('.json') || file.endsWith('.meta.json')) continue;
              const info = await stat(join(full, file));
              out.push({ name: file, dir, size: info.size, mtime: info.mtimeMs });
            }
          }
        } catch {
          /* runs/ may not exist yet */
        }
        out.sort((a, b) => b.mtime - a.mtime);
        return sendJSON(res, 200, { champions: out.slice(0, 50) });
      }
      if (url.pathname === '/api/champion' && req.method === 'GET') {
        const name = url.searchParams.get('name') ?? 'champion.json';
        const dir = url.searchParams.get('dir') ?? 'champion';
        const path = normalize(join(runsDir, dir, name));
        if (!path.startsWith(runsDir)) return sendJSON(res, 400, { error: 'bad path' });
        try {
          const text = await readFile(path, 'utf8');
          res.writeHead(200, { 'content-type': 'application/json; charset=utf-8' });
          return res.end(text);
        } catch {
          return sendJSON(res, 404, { error: 'not found' });
        }
      }
      if (url.pathname === '/api/champion' && req.method === 'POST') {
        const body = await readBody(req, 12 * 1024 * 1024);
        const parsed = JSON.parse(body);
        const dir = url.searchParams.get('dir') ?? 'uploads';
        const name = url.searchParams.get('name') ?? `policy-${Date.now()}.json`;
        const target = normalize(join(runsDir, dir, name));
        if (!target.startsWith(runsDir)) return sendJSON(res, 400, { error: 'bad path' });
        await mkdir(dirname(target), { recursive: true });
        await writeFile(target, JSON.stringify(parsed));
        return sendJSON(res, 200, { ok: true, path: `runs/${dir}/${name}` });
      }
      return sendJSON(res, 404, { error: 'unknown endpoint' });
    } catch (err) {
      return sendJSON(res, 500, { error: String(err && err.message ? err.message : err) });
    }
  }

  // ---- static -----------------------------------------------------------
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405).end('method not allowed');
    return;
  }
  const path = resolvePath(url.pathname);
  if (!path) {
    res.writeHead(403).end('forbidden');
    return;
  }
  if (await sendFile(res, path)) return;
  // Unknown path: fall back to the app shell for client side routing.
  if (!url.pathname.includes('.')) {
    if (await sendFile(res, join(publicDir, 'index.html'))) return;
  }
  res.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
});

function readBody(req, limit) {
  return new Promise((resolveBody, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error('payload too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolveBody(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

server.listen(port, host, () => {
  console.log(`RL--RUNNING serving on http://${host}:${port}`);
  console.log(`  app:      http://localhost:${port}/`);
  console.log(`  library:  http://localhost:${port}/src/ppo.js (imported directly by the browser workers)`);
  console.log(`  api:      /api/health, /api/champions, /api/champion`);
  console.log('  training does not need this server: node scripts/train.mjs');
});
