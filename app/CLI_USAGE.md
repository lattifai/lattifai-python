# LattifAI App - CLI Usage Guide

## Installation

### Local Development
```bash
cd app
npm install
npm run build
npm link
```

### From npm (when published)
```bash
npm install -g lai-app
```

## Usage

### Start the application
```bash
lai-app
```

This will:
- Start a local web server on port 5173
- Automatically open your default browser
- Serve the LattifAI Alignment web interface

### Options

```bash
lai-app [options]
```

**Available Options:**

- `-p, --port <number>` - Specify the port (default: 5173)
  ```bash
  lai-app --port 8080
  ```

- `--no-open` - Don't automatically open the browser
  ```bash
  lai-app --no-open
  ```

- `--backend <url>` - Configure backend API URL (default: http://localhost:8001)
  ```bash
  lai-app --backend http://localhost:9000
  ```

- `-h, --help` - Display help information
  ```bash
  lai-app --help
  ```

- `-V, --version` - Show version number
  ```bash
  lai-app --version
  ```

### Examples

**Start on custom port without opening browser:**
```bash
lai-app --port 3000 --no-open
```

**Use with custom backend:**
```bash
lai-app --backend https://api.example.com
```

## Development

### Build the frontend
```bash
npm run build
```

### Run in development mode (with hot reload)
```bash
npm run dev
```

### Run the built CLI locally
```bash
npm start
```

## Uninstall

```bash
npm unlink -g lai-app
```

## Distribution

To distribute the CLI tool, you can:

1. **Publish to npm:**
   ```bash
   npm publish
   ```

2. **Package as tarball:**
   ```bash
   npm pack
   ```
   Then share the `.tgz` file. Users can install with:
   ```bash
   npm install -g lai-app-1.0.0.tgz
   ```

3. **Use npx (no installation required):**
   ```bash
   npx lai-app
   ```
