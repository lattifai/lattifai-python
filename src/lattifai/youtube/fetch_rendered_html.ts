#!/usr/bin/env bun
/**
 * Lightweight headless Chrome CDP script for fetching JS-rendered HTML.
 *
 * Usage:  bun fetch_rendered_html.ts <url> [timeout_ms]
 *
 * Outputs the fully rendered DOM (document.documentElement.outerHTML) to stdout.
 * Called from Python via subprocess for SPA/CSR pages where plain HTTP returns an empty shell.
 *
 * Zero extra dependencies — uses system Chrome via raw CDP WebSocket.
 */

const url = process.argv[2];
const timeoutMs = parseInt(process.argv[3] || "30000");

if (!url) {
  console.error("Usage: bun fetch_rendered_html.ts <url> [timeout_ms]");
  process.exit(1);
}

// Find Chrome executable
function findChrome(): string {
  const envPath = process.env.CHROME_PATH || process.env.URL_CHROME_PATH;
  if (envPath) return envPath;
  const candidates =
    process.platform === "darwin"
      ? ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium"]
      : ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"];
  for (const c of candidates) {
    try {
      if (Bun.file(c).size > 0) return c;
    } catch {}
  }
  throw new Error("Chrome not found. Set CHROME_PATH environment variable.");
}

const chrome = findChrome();

// Get a free port
const server = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { data() {}, open() {}, close() {} } });
const port = server.port;
server.stop();

// Launch Chrome with the target URL
const chromeProc = Bun.spawn(
  [chrome, `--remote-debugging-port=${port}`, "--headless=new", "--disable-gpu", "--no-sandbox", "--disable-software-rasterizer", url],
  { stdout: "ignore", stderr: "ignore" },
);
const cleanup = () => {
  try {
    chromeProc.kill();
  } catch {}
};

try {
  // Wait for CDP page target
  let wsUrl = "";
  for (let i = 0; i < 50; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/json/list`);
      const targets = (await r.json()) as any[];
      const page = targets.find((t: any) => t.type === "page" && t.url.startsWith("http"));
      if (page) {
        wsUrl = page.webSocketDebuggerUrl;
        break;
      }
    } catch {}
    await Bun.sleep(300);
  }
  if (!wsUrl) throw new Error("No page target found");

  // Connect WebSocket
  const ws = new WebSocket(wsUrl);
  let msgId = 0;
  const pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();

  await new Promise<void>((resolve, reject) => {
    ws.onopen = () => resolve();
    ws.onerror = () => reject(new Error("WS connect failed"));
  });
  ws.onmessage = (ev: MessageEvent) => {
    const msg = JSON.parse(String(ev.data));
    if (msg.id !== undefined) {
      const p = pending.get(msg.id);
      if (p) {
        pending.delete(msg.id);
        msg.error ? p.reject(new Error(msg.error.message)) : p.resolve(msg.result);
      }
    }
  };

  const send = (method: string, params: any = {}): Promise<any> => {
    const id = ++msgId;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
  };

  // Wait for readyState === "complete"
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const { result } = await send("Runtime.evaluate", { expression: "document.readyState" });
    if (result.value === "complete") break;
    await Bun.sleep(500);
  }

  // Extra wait for SPA rendering + scroll to trigger lazy loading
  await Bun.sleep(3000);
  await send("Runtime.evaluate", {
    expression: `(async()=>{const h=Math.min(document.body.scrollHeight,15000);for(let i=0;i<h;i+=window.innerHeight){window.scrollTo(0,i);await new Promise(r=>setTimeout(r,200))}window.scrollTo(0,0)})()`,
    awaitPromise: true,
  });
  await Bun.sleep(2000);

  // Extract rendered HTML
  const { result } = await send("Runtime.evaluate", { expression: "document.documentElement.outerHTML" });
  process.stdout.write(result.value as string);
  console.error(`OK: ${(result.value as string).length} bytes`);

  ws.close();
} finally {
  cleanup();
}
