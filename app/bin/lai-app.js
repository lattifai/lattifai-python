#!/usr/bin/env node

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import express from 'express';
import open from 'open';
import { Command } from 'commander';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const program = new Command();

program
  .name('lai-app')
  .description('LattifAI Alignment Web Application - Text-Speech Forced Alignment Tool')
  .version('1.0.0')
  .option('-p, --port <number>', 'Port to run the server on', '5173')
  .option('--no-open', 'Do not automatically open browser')
  .option('--backend <url>', 'Backend API URL', 'http://localhost:8001')
  .parse(process.argv);

const options = program.opts();
const port = parseInt(options.port, 10);

const app = express();

// Serve static files from dist directory
const distPath = join(__dirname, '..', 'dist');
app.use(express.static(distPath));

// API proxy to backend (if needed)
app.use('/api', (req, res) => {
  res.status(503).json({
    error: 'Backend server not configured. Please start the backend separately or configure --backend option.'
  });
});

// Fallback to index.html for SPA routing
app.get('*', (req, res) => {
  res.sendFile(join(distPath, 'index.html'));
});

// Start server
app.listen(port, async () => {
  const url = `http://localhost:${port}`;
  console.log(`
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🎯 LattifAI Alignment Web App                      ║
║                                                       ║
║   Server running at: ${url.padEnd(28)} ║
║   Backend API: ${options.backend.padEnd(34)} ║
║                                                       ║
║   Press Ctrl+C to stop the server                    ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
  `);

  if (options.open) {
    try {
      await open(url);
      console.log('✓ Browser opened automatically\n');
    } catch (error) {
      console.error('Failed to open browser:', error.message);
      console.log(`Please open ${url} manually in your browser\n`);
    }
  }
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\nShutting down gracefully...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n\nShutting down gracefully...');
  process.exit(0);
});
