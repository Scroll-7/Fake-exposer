import { GoogleGenerativeAI } from '@google/generative-ai';
import dotenv from 'dotenv';
import fs from 'fs';
dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

export async function extractTextFromImage(imagePath) {
    try {
        const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

        const imageData = fs.readFileSync(imagePath);
        const base64Image = imageData.toString('base64');

        // Detect mime type from file magic bytes
        let mimeType = 'image/jpeg';
        if (imageData[0] === 0x89 && imageData[1] === 0x50) mimeType = 'image/png';
        else if (imageData[0] === 0x47 && imageData[1] === 0x49) mimeType = 'image/gif';
        else if (imageData[0] === 0x52 && imageData[1] === 0x49) mimeType = 'image/webp';

        const result = await model.generateContent([
            {
                inlineData: { data: base64Image, mimeType }
            },
            'Extract and return ONLY the raw text visible in this image. Do not add any commentary or formatting — just the plain text exactly as it appears.'
        ]);

        const text = result.response.text().trim();

        // Clean up uploaded image
        fs.unlink(imagePath, (err) => {
            if (err) console.error('Failed to delete temp image:', err);
        });

        return text;
    } catch (error) {
        console.error('Vision OCR Error:', error);
        fs.unlink(imagePath, () => {});
        throw new Error('Failed to extract text from image.');
    }
}

