import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// Camera access (getUserMedia) only works in a "secure context" — HTTPS,
// or plain http:// on localhost/127.0.0.1. This dev server binds to
// 0.0.0.0 so it's reachable from a phone on the same network, which means
// it's *not* localhost from the phone's point of view, so HTTPS is
// required or the browser blocks the camera outright regardless of any
// permission granted. basicSsl() serves a self-signed cert so that works;
// the browser will show a one-time "not secure" warning to click through.
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: '0.0.0.0', // Forces binding to all network interfaces to bypass 'localhost' Service Workers and allow external access
    allowedHosts: 'all'
  }
})
