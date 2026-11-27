import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  base: './', // relative, so the build works under any GitHub Pages path (/<repo>/)
  plugins: [react()],
})
