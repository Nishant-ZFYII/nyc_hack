"""Inspect how nat builds input_schema from a function."""
import sys, inspect
sys.path.insert(0, ".")

from nat.builder.function_info import FunctionInfo
import agent.register  # ensures our wrappers exist

# Take one of our wrapped tools and see what schema nat infers
from agent.register import _traced

async def sample(query: str, lat: float = 0, lon: float = 0, case_id: str = "") -> str:
    """Test tool."""
    return "ok"

wrapped = _traced("sample", sample)
print("=== wrapped signature ===", inspect.signature(wrapped))
print("=== wrapped annotations ===", wrapped.__annotations__)

info = FunctionInfo.from_fn(wrapped)
print()
print("=== FunctionInfo ===")
for attr in ("input_schema", "single_output_schema", "streaming_output_schema", "description"):
    v = getattr(info, attr, "<none>")
    print(f"  {attr}: {v}")

print()
print("=== input_schema fields ===")
if info.input_schema:
    print(info.input_schema.model_json_schema())

print()
print("=== Unwrapped (raw sample) ===")
info2 = FunctionInfo.from_fn(sample)
print("  input_schema:", info2.input_schema.model_json_schema() if info2.input_schema else None)
