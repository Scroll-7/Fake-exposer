import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const fr = JSON.parse(fs.readFileSync('public/i18n/fr.json', 'utf-8'));
const en = JSON.parse(fs.readFileSync('public/i18n/en.json', 'utf-8'));

function checkI18n() {
  const keys = Object.keys(fr);
  for (const key of keys) {
    if (!(key in en)) {
      return false;
    }
  }
  return true;
}

describe('i18n', () => {
  it('fr.json and en.json have the same keys', () => {
    assert.ok(checkI18n());
  });
});