import {test, expect} from '@playwright/test';

test('sources dialog opens from the credit, closes and hands focus back', async ({page}) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  const link = page.getByRole('button', {name: 'Sources, licenses and disclaimer'});
  const dialog = page.locator('#sources');
  await link.click();
  await expect(dialog).toBeVisible();
  await expect(page.locator('#sources-close')).toBeFocused();
  await expect(dialog).toContainText('not for clinical decisions');
  await expect(dialog.locator('.lic-nc')).toHaveText('CC BY-NC 4.0');
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(link).toBeFocused();
  await link.click();
  await page.locator('#sources-close').click();
  await expect(dialog).toBeHidden();
  expect(errors).toEqual([]);
});
