#!/usr/bin/env bun
/**
 * Lightweight headless Chrome CDP script for fetching JS-rendered HTML.
 *
 * Usage:  bun fetch_rendered_html.ts <url> [timeout_ms]
 *
 * Outputs the fully rendered DOM (document.documentElement.outerHTML) to stdout.
 * Called from Python via subprocess for SPA/CSR pages where plain HTTP returns an empty shell.
 *
 * CDP core (CdpConnection, Chrome launcher, page-load helpers) is adapted from
 * baoyu-url-to-markdown (https://github.com/nicepkg/baoyu-skills) — MIT License.
 */

import {
  CdpConnection,
  getFreePort,
  launchChrome,
  waitForChromeDebugPort,
  waitForPageLoad,
  waitForNetworkIdle,
  autoScroll,
  evaluateScript,
  killChrome,
} from "/Users/feiteng/.claude/skills/baoyu-url-to-markdown/scripts/cdp.js";
import { CDP_CONNECT_TIMEOUT_MS } from "/Users/feiteng/.claude/skills/baoyu-url-to-markdown/scripts/constants.js";

const url = process.argv[2];
const timeoutMs = parseInt(process.argv[3] || "30000");

if (!url) {
  console.error("Usage: bun fetch_rendered_html.ts <url> [timeout_ms]");
  process.exit(1);
}

const port = await getFreePort();
const chromeProcess = await launchChrome(url, port, true /* headless */);

try {
  const wsUrl = await waitForChromeDebugPort(port, CDP_CONNECT_TIMEOUT_MS);
  const cdp = await CdpConnection.connect(wsUrl, CDP_CONNECT_TIMEOUT_MS);

  // Find the page target and attach
  const targets = await cdp.send<{ targetInfos: Array<{ targetId: string; type: string; url: string }> }>("Target.getTargets");
  const pageTarget = targets.targetInfos.find((t) => t.type === "page" && t.url.startsWith("http"));
  if (!pageTarget) throw new Error("No page target found");

  const { sessionId } = await cdp.send<{ sessionId: string }>("Target.attachToTarget", {
    targetId: pageTarget.targetId,
    flatten: true,
  });
  await cdp.send("Network.enable", {}, { sessionId });
  await cdp.send("Page.enable", {}, { sessionId });

  // Wait for page load + network idle
  console.error(`Fetching: ${url}`);
  await Promise.race([waitForPageLoad(cdp, sessionId, timeoutMs), new Promise((r) => setTimeout(r, timeoutMs))]);
  await Promise.race([waitForNetworkIdle(cdp, sessionId, 3000), new Promise((r) => setTimeout(r, 5000))]);

  // Extra wait for SPA rendering
  await new Promise((r) => setTimeout(r, 3000));

  // Scroll to trigger lazy loading
  await autoScroll(cdp, sessionId, 10, 300);
  await new Promise((r) => setTimeout(r, 2000));

  // Extract rendered HTML
  const html = await evaluateScript<string>(cdp, sessionId, "document.documentElement.outerHTML");
  process.stdout.write(html);
  console.error(`OK: ${html.length} bytes`);

  cdp.close();
} finally {
  killChrome(chromeProcess);
}
