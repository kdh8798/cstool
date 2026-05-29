import { defineConfig } from 'vite';
import fs from 'fs';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 443,
    strictPort: true,

    https: {
      key: fs.readFileSync('./ssl/private.key'),
      cert: fs.readFileSync('./ssl/certificate.crt'),
    },

    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
