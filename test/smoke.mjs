import { pipeline, env } from '@huggingface/transformers';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
env.allowLocalModels = true;

const cases = [
  "Hi, I am Marcus Vance. Charge 4532-1182-9920-1092 and reach me at contact@acme.org.",
  "ph 555.010.9988 till 6",
  "CALL TOM REED AT 555-010-9988 ASAP",
  "Meeting moved to 3pm, room 402, bring the slides.",
  "Ship to 221B Baker Street, London, tonight please and email sarah.jones@example.com",
];

console.log('loading model (q8)...');
const t0 = Date.now();
const generator = await pipeline('text2text-generation', '../web_llm_model', {
  dtype: 'q8',
  device: 'cpu',
});
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

for (const text of cases) {
  const t1 = Date.now();
  const out = await generator('redact: ' + text, { max_new_tokens: 128 });
  const dt = Date.now() - t1;
  console.log(`\n[${dt}ms] ${text}\n  -> ${out[0].generated_text}`);
}
