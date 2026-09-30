import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './test',
    testMatch: '**/*.spec.js',
    timeout: 30000,
    retries: 1,
    use: {
        baseURL: 'http://localhost:3001',
        headless: true,
        viewport: { width: 1280, height: 720 },
    },
    webServer: {
        command: 'node server.js',
        port: 3001,
        timeout: 10000,
        reuseExistingServer: true,
    },
});
