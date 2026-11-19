import fs from 'node:fs';
// test/i18n.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

const fr = JSON.parse(fs.readFileSync('public/i18n/fr.json', 'utf-8'));
const en = JSON.parse(fs.readFileSync('public/i18n/en.json', 'utf-8'));

describe('i18n tests', () => {
  it('fr and en dictionaries have the same keys', () => {
    assert.deepStrictEqual(Object.keys(fr), Object.keys(en));
  });

  it('welcome key has correct values', () => {
    assert.strictEqual(fr.welcome, 'Bienvenue');
    assert.strictEqual(en.welcome, 'Welcome');
  });

  it('goodbye key has correct values', () => {
    assert.strictEqual(fr.goodbye, 'Au revoir');
    assert.strictEqual(en.goodbye, 'Goodbye');
  });

  it('error key has correct values', () => {
    assert.strictEqual(fr.error, 'Erreur');
    assert.strictEqual(en.error, 'Error');
  });
});