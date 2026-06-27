import { createWriteStream, existsSync, mkdirSync } from 'fs';
import { get } from 'https';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const KNOWN_FACES_DIR = join(__dirname, '..', 'known_faces');

// Wikimedia Commons search: find a free-license face photo
async function downloadFace(name) {
    mkdirSync(KNOWN_FACES_DIR, { recursive: true });

    const searchUrl = `https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(name + ' face portrait')}&srlimit=5&format=json&origin=*`;

    const searchRes = await fetch(searchUrl);
    const searchData = await searchRes.json();
    const pages = searchData?.query?.search || [];

    if (pages.length === 0) {
        console.error(`No Wikimedia results for "${name}"`);
        process.exit(1);
    }

    // Try each result until we find an image URL
    for (const page of pages) {
        const imageUrl = `https://commons.wikimedia.org/w/api.php?action=query&titles=${encodeURIComponent(page.title)}&prop=imageinfo&iiprop=url&iiurlwidth=500&format=json&origin=*`;
        const imgRes = await fetch(imageUrl);
        const imgData = await imgRes.json();
        const pagesData = imgData?.query?.pages || {};
        const pageInfo = Object.values(pagesData)[0];

        if (pageInfo?.imageinfo?.[0]?.url) {
            const url = pageInfo.imageinfo[0].url;
            const filename = name.toLowerCase().replace(/\s+/g, '_') + '.jpg';
            const filepath = join(KNOWN_FACES_DIR, filename);

            console.log(`Downloading ${url}...`);
            const response = await fetch(url);
            const buffer = Buffer.from(await response.arrayBuffer());
            createWriteStream(filepath).write(buffer);
            console.log(`Saved to ${filepath}`);
            return;
        }
    }

    console.error(`Could not find an image URL for "${name}"`);
    process.exit(1);
}

const name = process.argv[2];
if (!name) {
    console.error('Usage: node bin/download-face.js "Person Name"');
    console.error('Example: node bin/download-face.js "Leonardo DiCaprio"');
    process.exit(1);
}

downloadFace(name);
