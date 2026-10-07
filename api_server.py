"""
PCOS-BioQuant HTTP API Server & Interactive Clinical Web Dashboard
Serves the web application, REST API endpoints for live scan analysis,
preset loading, and benchmark telemetry.
"""

import os
import sys
import json
import base64
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import cv2
import numpy as np

# Ensure workspace packages are discoverable
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from pcos_bioquant.pipeline import PCOSBioQuantPipeline

PORT = 8080
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        print("[API Server] Initializing PCOSBioQuantPipeline...")
        PIPELINE = PCOSBioQuantPipeline()
        print("[API Server] Pipeline ready.")
    return PIPELINE


class PCOSRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE_ROOT, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(os.path.join(WORKSPACE_ROOT, "webapp", "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        elif path == "/api/presets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            presets_file = os.path.join(WORKSPACE_ROOT, "webapp", "presets", "presets_index.json")
            if os.path.exists(presets_file):
                with open(presets_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"[]")
            return

        elif path == "/api/benchmark":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            bm_file = os.path.join(WORKSPACE_ROOT, "experiments", "benchmark_summary.json")
            if os.path.exists(bm_file):
                with open(bm_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"{}")
            return

        elif path.startswith("/presets/"):
            rel_file = path[len("/presets/"):]
            full_path = os.path.join(WORKSPACE_ROOT, "webapp", "presets", rel_file)
            if os.path.exists(full_path):
                self.send_response(200)
                if full_path.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif full_path.endswith(".jpg") or full_path.endswith(".jpeg"):
                    self.send_header("Content-Type", "image/jpeg")
                else:
                    self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                with open(full_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        elif path.startswith("/webapp/"):
            rel_file = path[len("/webapp/"):]
            full_path = os.path.join(WORKSPACE_ROOT, "webapp", rel_file)
            if os.path.exists(full_path):
                self.send_response(200)
                if full_path.endswith(".css"):
                    self.send_header("Content-Type", "text/css; charset=utf-8")
                elif full_path.endswith(".js"):
                    self.send_header("Content-Type", "application/javascript; charset=utf-8")
                elif full_path.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif full_path.endswith(".jpg"):
                    self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(full_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        elif path.startswith("/experiments/"):
            rel_file = path[len("/experiments/"):]
            full_path = os.path.join(WORKSPACE_ROOT, "experiments", rel_file)
            if os.path.exists(full_path) and full_path.endswith(".png"):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(full_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                image_b64 = data.get("image")
                scan_id = data.get("scan_id", "SCAN_LIVE_001")

                if not image_b64:
                    self.send_error(400, "Missing 'image' parameter in JSON payload")
                    return

                # Decode base64 image
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(image_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    self.send_error(400, "Invalid image data format")
                    return

                # Run PCOS-BioQuant pipeline
                pipeline = get_pipeline()
                res = pipeline.analyze(img_bgr, scan_id=scan_id, generate_figures=False)

                # Encode overlay to base64
                _, overlay_buf = cv2.imencode(".png", res["overlay_bgr"])
                overlay_b64 = "data:image/png;base64," + base64.b64encode(overlay_buf).decode("utf-8")

                # Encode canonical raw to base64
                _, raw_buf = cv2.imencode(".png", res["prep_data"]["canonical_bgr"])
                raw_b64 = "data:image/png;base64," + base64.b64encode(raw_buf).decode("utf-8")

                ovary = res["ovary_data"]
                metrics = res["metrics"]

                response_obj = {
                    "status": "success",
                    "scan_id": scan_id,
                    "record": res["record"],
                    "overlay_b64": overlay_b64,
                    "raw_b64": raw_b64,
                    "capsule": ovary["contour"].tolist() if ovary.get("contour") is not None else [],
                    "centroid": ovary["centroid"],
                    "major_axis_px": ovary.get("major_axis_px"),
                    "minor_axis_px": ovary.get("minor_axis_px"),
                    "follicles": metrics.get("follicles_enriched", []),
                    "is_ovary": ovary.get("is_ovary", True),
                    "organ_type": ovary.get("organ_type", "Ovary")
                }

                response_bytes = json.dumps(response_obj).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)

            except Exception as e:
                import traceback
                traceback.print_exc()
                err_bytes = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(err_bytes)
            return

        self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port=PORT):
    # Pre-initialize pipeline for instant response
    get_pipeline()
    server_address = ("", port)
    httpd = HTTPServer(server_address, PCOSRequestHandler)
    print(f"\n=======================================================")
    print(f" PCOS-BioQuant Web Dashboard Server running at:")
    print(f" http://localhost:{port}/")
    print(f"=======================================================\n")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
