import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push(String(e)));

await page.goto('http://127.0.0.1:8123/index.html', { waitUntil: 'domcontentloaded' });
console.log('waiting for model load...');
await page.waitForFunction(() => !document.getElementById('runBtn').disabled, null, { timeout: 180000 });
console.log('model loaded, running redaction...');

await page.fill('#inputText',
  'Dr. Elena Rostova paid with 4111 2222 3333 4444, email e.r@clinic.org, call +44 20 7946 0958.');
await page.click('#runBtn');
await page.waitForFunction(
  () => document.getElementById('status').textContent.includes('Done in'),
  null, { timeout: 180000 });

console.log('OUTPUT: ', await page.textContent('#output'));
console.log('STATUS: ', await page.textContent('#status'));
console.log('DEVICE: ', await page.textContent('#device'));
console.log('GUARD : ', await page.isVisible('#guardWarn') ? 'FLAGGED' : 'ok');
await page.screenshot({ path: 'browser-test.png', fullPage: true });
console.log('ERRORS: ', errors.length ? errors : 'none');
await browser.close();
