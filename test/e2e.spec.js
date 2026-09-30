import { test, expect } from '@playwright/test';

test.describe('Fake Exposer E2E', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
    });

    test('page loads with correct title and heading', async ({ page }) => {
        await expect(page).toHaveTitle(/Fake Exposer/);
        await expect(page.locator('h1.title')).toContainText('Fake Exposer');
        await expect(page.locator('.subtitle')).toBeVisible();
    });

    test('three tabs are visible and text tab is active by default', async ({ page }) => {
        const tabs = page.locator('.tab-btn');
        await expect(tabs).toHaveCount(3);
        await expect(tabs.nth(0)).toContainText('Text');
        await expect(tabs.nth(1)).toContainText('URL');
        await expect(tabs.nth(2)).toContainText('Screenshot');
        await expect(tabs.nth(0)).toHaveClass(/active/);
    });

    test('clicking tabs switches between panes', async ({ page }) => {
        const tabBtns = page.locator('.tab-btn');
        const textPane = page.locator('#tab-text');
        const urlPane = page.locator('#tab-url');
        const imagePane = page.locator('#tab-image');

        await expect(textPane).toHaveClass(/active/);
        await expect(urlPane).not.toHaveClass(/active/);

        await tabBtns.nth(1).click();
        await expect(urlPane).toHaveClass(/active/);
        await expect(textPane).not.toHaveClass(/active/);

        await tabBtns.nth(2).click();
        await expect(imagePane).toHaveClass(/active/);
        await expect(urlPane).not.toHaveClass(/active/);

        await tabBtns.nth(0).click();
        await expect(textPane).toHaveClass(/active/);
    });

    test('results section is hidden initially', async ({ page }) => {
        await expect(page.locator('#results')).toHaveClass(/hidden/);
    });

    test('shows toast for empty text submission', async ({ page }) => {
        await page.locator('#analyze-text-btn').click();
        const toast = page.locator('.toast');
        await expect(toast).toBeVisible({ timeout: 500 });
    });

    test('shows toast for empty URL submission', async ({ page }) => {
        await page.locator('.tab-btn').nth(1).click();
        await page.locator('#analyze-url-btn').click();
        const toast = page.locator('.toast');
        await expect(toast).toBeVisible({ timeout: 500 });
    });

    test('shows toast for invalid URL', async ({ page }) => {
        await page.locator('.tab-btn').nth(1).click();
        await page.locator('#url-input').fill('not-a-url');
        await page.locator('#analyze-url-btn').click();
        const toast = page.locator('.toast');
        await expect(toast).toBeVisible({ timeout: 500 });
    });

    test('enter key submits text', async ({ page }) => {
        const textInput = page.locator('#text-input');
        await textInput.fill('some sample text');
        await textInput.press('Enter');
        // Should trigger the button click -> progress bar appears
        await expect(page.locator('#progress-section')).not.toHaveClass(/hidden/);
    });

    test('enter key submits URL', async ({ page }) => {
        await page.locator('.tab-btn').nth(1).click();
        const urlInput = page.locator('#url-input');
        await urlInput.fill('https://example.com');
        await urlInput.press('Enter');
        await expect(page.locator('#progress-section')).not.toHaveClass(/hidden/);
    });

    test('shift+enter adds newline in textarea does not submit', async ({ page }) => {
        const textInput = page.locator('#text-input');
        await textInput.fill('hello');
        await textInput.press('Shift+Enter');
        await textInput.press('Shift+Enter');
        const val = await textInput.inputValue();
        expect(val).toBe('hello\n\n');
        // Progress should still be hidden since we didn't submit
        await expect(page.locator('#progress-section')).toHaveClass(/hidden/);
    });

    test('auto-detects URL paste and switches to URL tab', async ({ page }) => {
        const textInput = page.locator('#text-input');
        const urlInput = page.locator('#url-input');

        // Fill the textarea, then dispatch a paste event to trigger auto-detect
        await textInput.fill('https://www.bbc.com/news/some-article');
        await page.evaluate(() => {
            document.querySelector('#text-input').dispatchEvent(new Event('paste', { bubbles: true }));
        });
        await page.waitForTimeout(100);

        await expect(page.locator('#tab-url')).toHaveClass(/active/);
        await expect(urlInput).toHaveValue('https://www.bbc.com/news/some-article');
        await expect(textInput).toHaveValue('');
    });

    test('image upload area is visible', async ({ page }) => {
        await page.locator('.tab-btn').nth(2).click();
        await expect(page.locator('#drop-zone')).toBeVisible();
        await expect(page.locator('#file-input')).toBeHidden();
    });

    test('image tab has context field and analyze button', async ({ page }) => {
        await page.locator('.tab-btn').nth(2).click();
        await expect(page.locator('#image-context')).toBeVisible();
        await expect(page.locator('#change-image-btn')).toHaveClass(/hidden/);
        await expect(page.locator('#analyze-image-btn')).toHaveClass(/hidden/);
    });

    test('file input accept attribute includes image types', async ({ page }) => {
        const accept = await page.locator('#file-input').getAttribute('accept');
        expect(accept).toContain('image/*');
    });

    test('all glass panels render', async ({ page }) => {
        const panels = page.locator('.glass-panel');
        const count = await panels.count();
        expect(count).toBeGreaterThanOrEqual(1);
    });
});
