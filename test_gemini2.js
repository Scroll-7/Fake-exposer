import { geminiSportsIdentityCheck } from './services/groq.js';
import fs from 'fs';

async function run() {
    console.log('Testing with local file...');
    const buffer = fs.readFileSync('c:/fake news/uploads/997c82f9ee8c1c872056a575b806b1d4');
    const base64 = buffer.toString('base64');
    console.log('Calling Gemini...');
    const result = await geminiSportsIdentityCheck(base64, 'image/jpeg');
    console.log('Gemini Result:', result);
}
run();
