import fs from 'fs';
import FormData from 'form-data';

async function test() {
    try {
        const formData = new FormData();
        const buffer = fs.readFileSync('test_cat.png');
        formData.append('file', buffer, { filename: 'test.png', contentType: 'image/png' });

        const res = await fetch('http://127.0.0.1:8001/predict_animal', {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders()
        });

        const text = await res.text();
        console.log("Status:", res.status);
        console.log("Body:", text);
    } catch (e) {
        console.error("Fetch Error:", e);
    }
}

test();
