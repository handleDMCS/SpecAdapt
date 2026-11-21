import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Keep all source text in the HTML body: notebook URL imports do not follow
// download links or run JavaScript to retrieve the repository content.
const directory = 'build/notebook-site';
const source = readFileSync(join(directory, 'specadapt.md'), 'utf8');
if (!source.includes('src/specadapt/bridge.py') || !source.includes('src/specadapt/decoding.py')) {
  throw new Error('The source bundle is missing the core library. Refusing to publish.');
}
const revision = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const timestamp = new Date().toISOString();
const snapshot = `Repository: SpecAdapt\nCommit: ${revision}\nBuilt at: ${timestamp}\n\n${source}`;
// Leave room for page metadata under Gemini Notebook's 500,000-word limit.
const words = snapshot.match(/\S+/gu)?.length ?? 0;
if (words > 490_000) {
  throw new Error(`Source has ${words} words; narrow repomix.config.jsonc before publishing.`);
}
const escapeHtml = (value) => value.replace(/[&<>"']/gu, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[character]);

writeFileSync(join(directory, 'specadapt.md'), snapshot);
writeFileSync(join(directory, 'index.html'), `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpecAdapt repository source</title>
  <style>
    body { max-width: 100rem; margin: 2rem auto; padding: 0 1.5rem; font: 1rem/1.6 system-ui, sans-serif; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 1rem; background: #f4f5f7; font-size: .85rem; }
  </style>
</head>
<body>
  <h1>SpecAdapt repository source</h1>
  <p>Commit: <code>${escapeHtml(revision)}</code>. Built at: ${escapeHtml(timestamp)}.</p>
  <p>Add this page's URL as a website source in Gemini Notebook.
  Re-import it after updates to use the latest code snapshot.</p>
  <p><a href="specadapt.md" download>Download the same source as Markdown</a></p>
  <main><pre>${escapeHtml(snapshot)}</pre></main>
</body>
</html>
`);
writeFileSync(join(directory, '.nojekyll'), '');
console.log(`Built notebook source: ${words.toLocaleString('en-US')} words, commit ${revision}.`);
