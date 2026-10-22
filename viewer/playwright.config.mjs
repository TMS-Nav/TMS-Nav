import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'tests/browser', timeout:60000, workers:1,
  use:{baseURL:'http://127.0.0.1:1234', viewport:{width:1440,height:1000},
    launchOptions:{args:['--enable-unsafe-swiftshader'],
      ...(process.env.CONDA_PREFIX ? {env:{...process.env,
        LD_LIBRARY_PATH:process.env.CONDA_PREFIX+'/lib'+(process.env.LD_LIBRARY_PATH ? ':'+process.env.LD_LIBRARY_PATH : '')}} : {})}},
  webServer:[{command:'python ../run_viewer.py --api-only',url:'http://127.0.0.1:8765/api/health',reuseExistingServer:!process.env.CI,timeout:30000},{command:'npm run dev -- --host 127.0.0.1', url:'http://127.0.0.1:1234',
    reuseExistingServer:!process.env.CI, timeout:120000}],
});
