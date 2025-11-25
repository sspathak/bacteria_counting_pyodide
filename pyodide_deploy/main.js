const statusDiv = document.getElementById('status');
const processBtn = document.getElementById('processBtn');
const fileInput = document.getElementById('fileInput');
const resultsDiv = document.getElementById('results');
const metricsDiv = document.getElementById('metrics');

let pyodide = null;
let processor = null;

async function initPyodide() {
    try {
        statusDiv.innerText = "Loading Pyodide...";
        pyodide = await loadPyodide();
        
        statusDiv.innerText = "Installing dependencies (numpy, opencv, matplotlib, tqdm)...";
        await pyodide.loadPackage(['numpy', 'opencv-python', 'matplotlib']);
        await pyodide.loadPackage('micropip');
        const micropip = pyodide.pyimport("micropip");
        await micropip.install('tqdm');

        statusDiv.innerText = "Loading application code...";
        
        // List of python files to load
        const pythonFiles = [
            'python/extract_feature.py',
            'python/frontier.py',
            'python/bacteria.py',
            'python/bounds.py',
            'python/hyperparameters.py',
            'processor.py'
        ];

        // Fetch and write files to Pyodide FS
        for (const file of pythonFiles) {
            const response = await fetch(file);
            if (!response.ok) throw new Error(`Failed to fetch ${file}`);
            const text = await response.text();
            // Write to root of virtual filesystem for easy import
            const fileName = file.split('/').pop();
            pyodide.FS.writeFile(fileName, text);
        }

        // Import the processor module in Python
        // We just run the file content or import it.
        // Since we wrote it to filesystem, we can import it.
        await pyodide.runPythonAsync(`
            import sys
            sys.path.append('.')
            import processor
        `);

        processor = pyodide.globals.get('processor');
        
        statusDiv.innerText = "Ready. Please upload an image.";
        processBtn.disabled = false;
        
    } catch (err) {
        console.error(err);
        statusDiv.innerText = `Error initializing: ${err.message}`;
    }
}

processBtn.addEventListener('click', async () => {
    if (!fileInput.files.length) {
        alert("Please select a file first.");
        return;
    }
    
    const file = fileInput.files[0];
    statusDiv.innerText = "Processing...";
    processBtn.disabled = true;
    resultsDiv.innerHTML = '';
    metricsDiv.style.display = 'none';

    try {
        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        
        // Call the Python function
        // We need to pass the bytes. Pyodide handles Uint8Array -> bytes conversion automatically in newer versions,
        // but explicitly using to_py() or just passing it often works.
        // Let's rely on Pyodide's automatic conversion for Uint8Array to bytes.
        
        const minSize = parseInt(document.getElementById('minSize').value) || 5;
        const maxSize = parseInt(document.getElementById('maxSize').value) || 70;
        const maxDiameter = parseInt(document.getElementById('maxDiameter').value) || 17;
        const threshold = parseInt(document.getElementById('threshold').value) || 5;

        const jsonResult = processor.process_image_data(uint8Array, minSize, maxSize, maxDiameter, threshold);
        const result = JSON.parse(jsonResult);
        
        if (result.error) {
            throw new Error(result.error + (result.traceback ? "\n" + result.traceback : ""));
        }
        
        displayResults(result);
        sendToRN(result);
        
        statusDiv.innerText = `Processing complete. Found ${result.count} bacteria.`;
        
    } catch (err) {
        console.error(err);
        statusDiv.innerText = `Error processing: ${err.message}`;
    } finally {
        processBtn.disabled = false;
    }
});

function displayResults(result) {
    // Display Metrics
    metricsDiv.innerHTML = `
        <h3>Analysis Results</h3>
        <p><strong>Bacteria Count:</strong> ${result.count}</p>
        <p><strong>Max Shape:</strong> ${JSON.stringify(result.metrics.max_shape)}</p>
    `;
    metricsDiv.style.display = 'block';

    // Display Images
    // Sort keys to show original first, then debug images in some order
    const keys = Object.keys(result.images).sort((a, b) => {
        if (a === 'original') return -1;
        if (b === 'original') return 1;
        return a.localeCompare(b);
    });

    for (const key of keys) {
        const base64Data = result.images[key];
        const card = document.createElement('div');
        card.className = 'image-card';
        
        const img = document.createElement('img');
        img.src = base64Data;
        
        const title = document.createElement('h4');
        title.innerText = formatTitle(key);
        
        card.appendChild(img);
        card.appendChild(title);
        resultsDiv.appendChild(card);
    }
}

function formatTitle(key) {
    // humanize title
    return key.replace(/_/g, ' ').replace('.png', '');
}

// React Native Bridge
function sendToRN(data) {
    const message = {
        type: "bacteria_analysis_result",
        payload: data
    };
    
    if (window.ReactNativeWebView) {
        window.ReactNativeWebView.postMessage(JSON.stringify(message));
        console.log("Sent data to ReactNativeWebView");
    } else {
        console.log("ReactNativeWebView not found. Data:", message);
    }
}


// --- Helper Functions ---
function base64ToUint8Array(base64) {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

// --- Exposed Functions for React Native Bridge ---

window.load_image_from_b64 = async function(b64_string) {
    try {
        console.log("load_image_from_b64 called");
        const bytes = base64ToUint8Array(b64_string);
        pyodide.FS.writeFile('input.png', bytes);
        statusDiv.innerText = "Image loaded from React Native.";
        return JSON.stringify({status: 'success'});
    } catch (e) {
        console.error(e);
        return JSON.stringify({status: 'error', message: e.message});
    }
};

window.analyze_bacteria = async function(args) {
    try {
        console.log("analyze_bacteria called");
        statusDiv.innerText = "Processing (via bridge)...";
        resultsDiv.innerHTML = '';
        metricsDiv.style.display = 'none';

        // Read input.png from FS
        if (!pyodide.FS.analyzePath('input.png').exists) {
            throw new Error("input.png not found. Call load_image_from_b64 first.");
        }
        const data = pyodide.FS.readFile('input.png');
        
        let minSize = 5, maxSize = 70, maxDiameter = 17, threshold = 5;
        
        // Use DOM values as base, but allow args to override
        if (document.getElementById('minSize')) {
             minSize = parseInt(document.getElementById('minSize').value) || 5;
             maxSize = parseInt(document.getElementById('maxSize').value) || 70;
             maxDiameter = parseInt(document.getElementById('maxDiameter').value) || 17;
             threshold = parseInt(document.getElementById('threshold').value) || 5;
        }

        if (args && typeof args === 'object') {
            if (args.minSize !== undefined) minSize = args.minSize;
            if (args.maxSize !== undefined) maxSize = args.maxSize;
            if (args.maxDiameter !== undefined) maxDiameter = args.maxDiameter;
            if (args.threshold !== undefined) threshold = args.threshold;
        }
        
        const jsonResult = processor.process_image_data(data, minSize, maxSize, maxDiameter, threshold);
        const result = JSON.parse(jsonResult);

        if (result.error) {
            throw new Error(result.error + (result.traceback ? "\n" + result.traceback : ""));
        }

        displayResults(result);
        // The caller expects a return string, usually status, but maybe we should return the result?
        // The existing analyze_particles just returns status: success and then sends a message later.
        // But here we can return the result or send it via postMessage.
        // Let's send it via postMessage as per original flow, and return success.
        sendToRN(result);
        
        statusDiv.innerText = `Processing complete. Found ${result.count} bacteria.`;
        return JSON.stringify({status: 'success', count: result.count});

    } catch (e) {
        console.error(e);
        return JSON.stringify({status: 'error', message: e.message});
    }
};

// Generic one-shot function
window.process_base64_image = async function(b64_string) {
    await window.load_image_from_b64(b64_string);
    return await window.analyze_bacteria();
};


// --- React Native Message Listener (Fallback/Bridge) ---
// This handles messages in the format {fn_name: '...', args: ...} 
// sent by the RN app if the specific ImJoy injection is not present or if we want to support it natively.

window.addEventListener('message', async function(event) {
    try {
        // Basic validation
        if (typeof event.data !== 'string') return;
        
        let message;
        try {
            message = JSON.parse(event.data);
        } catch (e) {
            return; // Not a JSON message
        }

        if (!message.fn_name) return;

        const fnName = message.fn_name;
        const args = message.args;

        console.log(`Received bridge call: ${fnName}`);

        if (window[fnName]) {
            const result = await window[fnName](args);
            
            // Send response back in the format expected by the RN app's onMessage handler
            // The existing RN app expects: { input: event.data, return: fn_output_string }
            const response = JSON.stringify({
                input: event.data,
                return: result
            });
            
            if (window.ReactNativeWebView) {
                window.ReactNativeWebView.postMessage(response);
            }
        } else {
            console.log(`Function ${fnName} not found on window.`);
        }

    } catch (e) {
        console.error("Error in bridge listener:", e);
    }
});

// Initialize
initPyodide();
