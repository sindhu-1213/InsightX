import { test, expect } from '@playwright/test';
import { Buffer } from 'node:buffer';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const screenshotsDir = path.join(__dirname, 'screenshots');

test.beforeAll(async () => {
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }
});

async function navigateToAnalyze(page) {
  await page.goto('/analyze');
  const skipBtn = page.locator('#skip-splash-btn');
  try {
    if (await skipBtn.isVisible({ timeout: 2000 })) {
      await skipBtn.click();
    }
  } catch {
    // Splash already finished or not present
  }
  await expect(page.locator('#analyze-workbench')).toBeVisible({ timeout: 15000 });
}

test.describe('In-Browser Chest Radiograph Analysis Smoke Tests', () => {
  test('Sample 1 analysis as guest: loads model, infers in browser, and displays Grad-CAM overlay', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    page.on('pageerror', (err) => {
      consoleErrors.push(err.message);
    });

    await navigateToAnalyze(page);

    // Select the first sample card (Normal chest radiograph)
    const firstSampleBtn = page.locator('#sample-gallery-section button', { hasText: 'Try This Sample' }).first();
    await expect(firstSampleBtn).toBeVisible();
    await firstSampleBtn.click();

    // Wait for the diagnostic dashboard verdict to appear
    const verdictCard = page.locator('#verdict-card');
    await expect(verdictCard).toBeVisible({ timeout: 45000 });

    // Assert verdict contains classification finding
    const verdictText = await verdictCard.innerText();
    expect(verdictText).toMatch(/Normal|Abnormal/);
    expect(verdictText).toMatch(/\d+\.\d+%/);

    // Assert "Ran in your browser" privacy confirmation is present
    const privacyBadge = page.locator('#browser-privacy-badge');
    await expect(privacyBadge).toBeVisible();
    await expect(privacyBadge).toContainText('Ran in your browser');

    // Assert Grad-CAM overlay image is rendered with positive dimensions
    const overlayImg = page.locator('img[alt="Grad-CAM Heatmap"]');
    await expect(overlayImg).toBeVisible();
    const naturalWidth = await overlayImg.evaluate((img) => img.naturalWidth);
    const naturalHeight = await overlayImg.evaluate((img) => img.naturalHeight);
    expect(naturalWidth).toBeGreaterThan(0);
    expect(naturalHeight).toBeGreaterThan(0);

    // Save screenshot
    await page.screenshot({ path: path.join(screenshotsDir, '01-sample-analysis.png'), fullPage: true });

    // Ensure zero console or unhandled page errors
    expect(consoleErrors).toEqual([]);
  });

  test('Custom PNG upload: executes client-side analysis and produces overlay', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    page.on('pageerror', (err) => {
      consoleErrors.push(err.message);
    });

    await navigateToAnalyze(page);

    // Upload sample_01_normal.png through hidden file input
    const sampleFilePath = path.resolve(__dirname, '../../samples/sample_01_normal.png');
    expect(fs.existsSync(sampleFilePath)).toBe(true);

    const fileInput = page.locator('#file-upload-input');
    await fileInput.setInputFiles(sampleFilePath);

    // Click Run In-Browser Neural Analysis button
    const runBtn = page.locator('#run-analysis-btn');
    await expect(runBtn).toBeEnabled({ timeout: 15000 });
    await runBtn.click();

    // Wait for verdict
    const verdictCard = page.locator('#verdict-card');
    await expect(verdictCard).toBeVisible({ timeout: 45000 });

    const verdictText = await verdictCard.innerText();
    expect(verdictText).toMatch(/Normal|Abnormal/);
    expect(verdictText).toMatch(/\d+\.\d+%/);

    // Verify privacy badge
    await expect(page.locator('#browser-privacy-badge')).toContainText('Ran in your browser');

    // Verify Grad-CAM overlay image
    const overlayImg = page.locator('img[alt="Grad-CAM Heatmap"]');
    await expect(overlayImg).toBeVisible();
    const naturalWidth = await overlayImg.evaluate((img) => img.naturalWidth);
    expect(naturalWidth).toBeGreaterThan(0);

    // Save screenshot
    await page.screenshot({ path: path.join(screenshotsDir, '02-upload-analysis.png'), fullPage: true });

    // Assert zero console errors
    expect(consoleErrors).toEqual([]);
  });

  test('Corrupt image upload: shows friendly error alert with retry button without breaking app', async ({ page }) => {
    let unhandledPageError = null;
    page.on('pageerror', (err) => {
      unhandledPageError = err;
    });

    await navigateToAnalyze(page);

    // Upload a corrupt pseudo-image (plain text content with .png extension)
    const fileInput = page.locator('#file-upload-input');
    await fileInput.setInputFiles({
      name: 'corrupted_sample.png',
      mimeType: 'image/png',
      buffer: Buffer.from('THIS IS NOT A VALID PNG IMAGE HEADER OR PAYLOAD'),
    });

    // Expect friendly error banner
    const errorAlert = page.locator('#analyze-error-alert');
    await expect(errorAlert).toBeVisible({ timeout: 10000 });

    // Assert friendly message and retry button
    const errorText = await errorAlert.innerText();
    expect(errorText).toMatch(/Invalid or Corrupt Image|could not be decoded/i);

    const retryBtn = errorAlert.locator('button', { hasText: 'Retry' });
    await expect(retryBtn).toBeVisible();

    // Assert page did not crash or blank out
    await expect(page.locator('#analyze-workbench')).toBeVisible();
    expect(unhandledPageError).toBeNull();

    // Save screenshot
    await page.screenshot({ path: path.join(screenshotsDir, '03-corrupt-file-error.png'), fullPage: true });
  });
});
