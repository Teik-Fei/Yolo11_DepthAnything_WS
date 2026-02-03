# 1. Import the internal check module BEFORE importing YOLO
import ultralytics.utils.checks

# 2. Define a dummy function that does NOTHING
def dummy_check_requirements(requirements, exclude=(), install=True, cmds=''):
    print(f" >> [BLOCKED] Ultralytics tried to auto-install: {requirements}")
    return True

# 3. Overwrite the real function with our dummy one
ultralytics.utils.checks.check_requirements = dummy_check_requirements

# 4. Now it is safe to import YOLO
from ultralytics import YOLO

print("Loading model...")
model = YOLO("sauvc_5.pt")

print("Starting Export (Auto-Install Disabled)...")
# We only use simplify=False. The validation step will fail gracefully 
# because onnxruntime is missing, but it WON'T crash/download.
success = model.export(
    format="onnx", 
    dynamic=True, 
    simplify=False
)

print(f"Export finished: {success}")
