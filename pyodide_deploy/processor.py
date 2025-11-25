import cv2
import numpy as np
import base64
import json
import os
import sys

# In the Pyodide environment, we will write all python files to the root directory
# so we can import them directly.
from extract_feature import BacteriaGenerator, C, MAX_DIAMETER, SIZE

def process_image_data(image_data_bytes):
    try:
        # Setup paths
        input_filename = "input.png"
        debug_dir = "debug_output"
        
        # Clean previous debug output
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        else:
            for f in os.listdir(debug_dir):
                os.remove(os.path.join(debug_dir, f))
        
        # Write input file
        # image_data_bytes is expected to be a bytes-like object (e.g. from to_py())
        # If it's a JsProxy (passed directly from JS), convert it to python bytes/memoryview
        if hasattr(image_data_bytes, "to_py"):
            image_data_bytes = image_data_bytes.to_py()
            
        with open(input_filename, "wb") as f:
            f.write(image_data_bytes)
        
        # Read image
        img = cv2.imread(input_filename)
        if img is None:
            return json.dumps({"error": "Could not decode image", "count": 0, "images": {}})
            
        # Initialize generator
        # BacteriaGenerator(size_bounds, max_diameter, debug, cover_corners)
        # We enable debug to get the intermediate images
        gen = BacteriaGenerator(size_bounds=SIZE, max_diameter=MAX_DIAMETER, debug=True, cover_corners=True)
        
        # Process
        # generate_bacts(self, img, label, image_name = "current_image.bmp", debug_path = "for_debug")
        bacts, max_shape = gen.generate_bacts(img, label=1, image_name="result.png", debug_path=debug_dir)
        
        # Collect results
        results = {
            "count": len(bacts),
            "images": {},
            "metrics": {
                "max_shape": max_shape
            }
        }
        
        # Add original image for reference (compressed to save space if needed, but keeping png for quality)
        _, buffer = cv2.imencode('.png', img)
        results["images"]["original"] = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
        
        # Read debug images
        # The debug images are written by extract_feature.py into debug_dir
        for filename in os.listdir(debug_dir):
            filepath = os.path.join(debug_dir, filename)
            if os.path.isfile(filepath):
                img_debug = cv2.imread(filepath)
                if img_debug is not None:
                    _, buffer = cv2.imencode('.png', img_debug)
                    b64_str = base64.b64encode(buffer).decode('utf-8')
                    # keys like 'threshold_result.png', 'num_bacts_5_result.png'
                    key = os.path.splitext(filename)[0]
                    results["images"][key] = "data:image/png;base64," + b64_str
        
        return json.dumps(results)
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
