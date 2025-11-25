# OpenCV Pyodide Bacteria Counter Deployment

This project implements a static webpage that performs client-side image processing using OpenCV and Pyodide. It is designed to run as a standalone web application or embedded within a React Native WebView.

## System Architecture

The system consists of three main layers:

1.  **Frontend (HTML/JS)**:
    *   Handles user interactions (image upload) and results display.
    *   Manages the Pyodide runtime lifecycle.
    *   Implements a communication bridge for React Native integration.
    *   **Key Files**: `index.html`, `main.js`.

2.  **WebAssembly Runtime (Pyodide)**:
    *   Executes the Python code directly in the browser.
    *   Provides a virtual filesystem (FS) to share data between JavaScript and Python.
    *   **Dependencies**: `numpy`, `opencv-python`, `matplotlib`, `tqdm`.

3.  **Python Core (`processor.py`)**:
    *   Wraps the underlying image processing logic.
    *   Handles I/O: reads input images from the Pyodide FS, executes the feature extraction pipeline, and writes results to the FS.
    *   Returns structured JSON data including metrics and Base64-encoded result images.

## React Native Bridge Integration

The application supports a two-way communication protocol compatible with React Native's `react-native-webview`.

### Protocol
*   **Requests (RN -> Web)**:
    *   Format: `JSON.stringify({ fn_name: "function_name", args: ... })`
    *   Method: `webviewRef.current.postMessage(...)`
*   **Responses (Web -> RN)**:
    *   Format: `JSON.stringify({ input: <original_request>, return: <result_data> })`
    *   Method: `window.ReactNativeWebView.postMessage(...)`

### Exposed Functions
The `main.js` exposes the following functions to the `window` object for the bridge:

*   `window.load_image_from_b64(b64_string)`: Decodes a Base64 image string and saves it as `input.png` in the Pyodide virtual filesystem.
*   `window.analyze_bacteria(args)`: Triggers the processing of `input.png`. It returns a status and sends a full payload (metrics + images) back via the bridge.
*   `window.process_base64_image(b64_string)`: A convenience function to load and analyze in a single step.

## Deployment

The project is configured for automatic deployment to GitHub Pages via GitHub Actions.

1.  **Workflow**: `.github/workflows/deploy_pyodide.yml`
2.  **Trigger**: Pushing to the `main` branch.
3.  **Process**:
    *   Checks out the repository.
    *   Uploads the `pyodide_deploy` directory as a build artifact.
    *   Deploys the artifact to the `github-pages` environment.

### Local Testing
Because Pyodide loads external scripts and WASM binaries, it requires a local web server (due to CORS restrictions on `file://` protocol).

To test locally:
```bash
cd pyodide_deploy
python3 -m http.server 8000
```
Access the page at `http://localhost:8000`.
