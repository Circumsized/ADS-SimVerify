import sys
import traceback
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

print("Starting Rigorous Diagnostic Probe...", flush=True)

try:
    print("1. Attempting to import isaac_dora_node...", flush=True)
    import isaac_dora_node
    print("Import succeeded!", flush=True)
    
    print("2. Testing CLIDDEngine instantiation with ONNX...", flush=True)
    engine = isaac_dora_node.CLIDDEngine(os.path.join(REPO_ROOT, "model", "xfeat_640x640.onnx"))
    print("CLIDDEngine instantiation succeeded!", flush=True)

    print("3. Executing main() inside guarded try-except block...", flush=True)
    isaac_dora_node.main()
    
except Exception as e:
    print("\nCRITICAL PROBE FAILURE", flush=True)
    print("========================================================", flush=True)
    print(f"Exception Type: {type(e).__name__}", flush=True)
    print(f"Exception Message: {e}", flush=True)
    print("Traceback:", flush=True)
    traceback.print_exc(file=sys.stdout)
    print("========================================================", flush=True)
    sys.exit(1)
