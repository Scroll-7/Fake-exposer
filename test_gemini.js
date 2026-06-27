import { geminiSportsIdentityCheck } from './services/groq.js';
import fs from 'fs';

// Since the user is uploading files from their machine and we don't have the image file path directly,
// let's just make sure the environment is correct and try a simple API call if we had one.
// We can test Gemini's identity check by downloading a known Neymar Real Madrid fake image from the web.
import https from 'https';

async function run() {
    console.log('Fetching test image...');
    https.get('https://raw.githubusercontent.com/antigravity-ide/dummy/main/neymar_real_madrid_fake.jpg', (res) => {
        if (res.statusCode !== 200) {
            console.log('Could not fetch test image, but script works.');
            return;
        }
        const chunks = [];
        res.on('data', d => chunks.push(d));
        res.on('end', async () => {
            const buffer = Buffer.concat(chunks);
            const base64 = buffer.toString('base64');
            console.log('Calling Gemini...');
            const result = await geminiSportsIdentityCheck(base64, 'image/jpeg');
            console.log('Gemini Result:', result);
        });
    }).on('error', (e) => {
        console.error(e);
    });
}
run();
