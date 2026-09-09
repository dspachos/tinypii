import { pipeline, env } from '@huggingface/transformers';
import fs from 'fs';

env.allowLocalModels = true;

const TAG_RE = /\[(NAME|CREDIT_CARD|EMAIL|PHONE|ADDRESS)\]/g;
const TAGS = ['NAME', 'CREDIT_CARD', 'EMAIL', 'PHONE', 'ADDRESS'];

function tagCounts(t) {
  const c = {};
  for (const m of (t.match(TAG_RE) || [])) c[m.slice(1, -1)] = (c[m.slice(1, -1)] || 0) + 1;
  return c;
}
function tokens(t) {
  return t.replace(TAG_RE, ' ').replace(/[^0-9A-Za-z]+/g, ' ').split(' ').filter(Boolean);
}
function metrics(pairs) {
  let matchedT = 0, goldT = 0, predT = 0, matchedK = 0, goldK = 0, extraK = 0, exact = 0;
  for (const [pred, gold] of pairs) {
    const pc = tagCounts(pred), gc = tagCounts(gold);
    for (const t of TAGS) matchedT += Math.min(pc[t] || 0, gc[t] || 0);
    goldT += Object.values(gc).reduce((a, b) => a + b, 0);
    predT += Object.values(pc).reduce((a, b) => a + b, 0);
    const pt = tokens(pred), gt = tokens(gold);
    let i = 0, m = 0;
    for (const w of gt) { while (i < pt.length && pt[i] !== w) i++; if (i < pt.length) { m++; i++; } }
    matchedK += m; goldK += gt.length; extraK += pt.length - m;
    exact += Number(pred.trim() === gold.trim());
  }
  const n = Math.max(pairs.length, 1);
  return {
    tag_recall: matchedT / Math.max(goldT, 1),
    tag_precision: matchedT / Math.max(predT, 1),
    copy_fidelity: matchedK / Math.max(goldK, 1),
    exact_match: exact / n,
  };
}

const rows = fs.readFileSync('../data/synthetic.jsonl', 'utf-8').trim().split('\n')
  .map(l => JSON.parse(l));
// same 90/10 split as training: datasets shuffles with seed 42 — approximate
// by taking every 10th row for an independent sample instead (n=200).
const sample = rows.filter((_, i) => i % 20 === 0).slice(0, 200);
const adversarial = JSON.parse(fs.readFileSync('../eval/adversarial.json', 'utf-8'));

console.log(`loading q8 model...`);
const gen = await pipeline('text2text-generation', '../web_llm_model', { dtype: 'q8', device: 'cpu' });

async function run(rows) {
  const pairs = [];
  for (const r of rows) {
    const out = await gen('redact: ' + r.raw_input, { max_new_tokens: 128 });
    pairs.push([out[0].generated_text, r.redacted_target]);
  }
  return pairs;
}

const t0 = Date.now();
const held = await run(sample);
const adv = await run(adversarial);
const total = Date.now() - t0;

console.log(`\n== q8 held-out sample (n=${sample.length}) ==`);
console.log(metrics(held));
console.log(`\n== q8 adversarial (n=${adversarial.length}) ==`);
console.log(metrics(adv));
console.log(`\ntotal gen time: ${(total / 1000).toFixed(1)}s for ${sample.length + adversarial.length} examples`);
