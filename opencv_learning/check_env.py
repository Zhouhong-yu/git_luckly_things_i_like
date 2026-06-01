# -*- coding: utf-8 -*-
# =============================================================================
# Diagnostic script: check Python + OpenCV environment
# Usage: python check_env.py
# =============================================================================
import sys, os

# Windows console encoding fix
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

print("=" * 55)
print("  Python + OpenCV Environment Check")
print("=" * 55)
print(f"Python:    {sys.version.split()[0]}")
print(f"Executable: {sys.executable}")

# ---- 1. Check dependencies ----
errors = []

print("\n[1/5] Checking packages...")
for pkg, import_name in [("numpy", "numpy"), ("cv2", "cv2")]:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {pkg} {ver}")
    except ImportError:
        print(f"  [FAIL] {pkg} not installed. Run: pip install {pkg}")
        errors.append(pkg)

try:
    import matplotlib
    print(f"  [OK] matplotlib {matplotlib.__version__}")
except ImportError:
    print(f"  [WARN] matplotlib not installed (some charts won't work)")

if errors:
    print(f"\nMissing packages. Please run:")
    print(f"  pip install {' '.join(errors)} opencv-python matplotlib")
    sys.exit(1)

import cv2
import numpy as np

# ---- 2. Check paths ----
print("\n[2/5] Checking project paths...")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"  Project root: {PROJECT_ROOT}")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
SAMPLE_PATH = os.path.join(DATA_DIR, "sample.jpg")

has_image = os.path.exists(SAMPLE_PATH)
print(f"  Test image: {'[OK] exists' if has_image else '[WARN] will generate'}")

# ---- 3. Generate test image ----
print("\n[3/5] Generating test image...")
try:
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (200, 180), (255, 0, 0), -1)    # Blue
    cv2.rectangle(img, (220, 50), (370, 180), (0, 255, 0), -1)   # Green
    cv2.rectangle(img, (390, 50), (540, 180), (0, 0, 255), -1)   # Red
    cv2.circle(img, (125, 280), 80, (255, 255, 0), -1)           # Cyan
    cv2.circle(img, (320, 280), 80, (0, 255, 255), 3)            # Yellow outline
    cv2.circle(img, (515, 280), 80, (255, 0, 255), -1)           # Magenta
    triangle = np.array([[100, 380], [200, 460], [20, 460]], dtype=np.int32)
    cv2.fillPoly(img, [triangle.reshape(-1, 1, 2)], (0, 165, 255))
    cv2.putText(img, "OpenCV Test Image", (130, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    print(f"  [OK] Image created: {img.shape}")
except Exception as e:
    print(f"  [FAIL] Image generation failed: {e}")
    sys.exit(1)

# ---- 4. Test OpenCV core functions ----
print("\n[4/5] Testing OpenCV core functions...")
try:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  [OK] BGR->Gray: {gray.shape}")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print(f"  [OK] GaussianBlur")

    edges = cv2.Canny(gray, 50, 150)
    print(f"  [OK] Canny edge detection")

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  [OK] findContours: {len(contours)} contours found")
except Exception as e:
    print(f"  [FAIL] OpenCV test error: {type(e).__name__}: {e}")
    sys.exit(1)

# ---- 5. Save test image ----
print("\n[5/5] Saving test image...")
try:
    cv2.imwrite(SAMPLE_PATH, img)
    print(f"  [OK] Saved to: {SAMPLE_PATH}")
except Exception as e:
    print(f"  [WARN] Save failed: {e}")

# ---- Done ----
print("\n" + "=" * 55)
print("  [OK] Environment check passed!")
print("  Now run: python main.py")
print("=" * 55)
