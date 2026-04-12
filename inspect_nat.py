"""Inspect OutputArgsSchema and how nat validates tool I/O."""
import sys, inspect
sys.path.insert(0, ".")

from nat.builder.function_info import FunctionInfo


# Simulate a real tool with return type str
async def tool_str(query: str, lat: float = 0) -> str:
    """doc"""
    return "hello world"

info = FunctionInfo.from_fn(tool_str)
print("=== tool that returns str → single_output_schema ===")
print(info.single_output_schema.model_json_schema() if info.single_output_schema else None)

# And without return annotation
async def tool_noret(query: str):
    """doc"""
    return "hello"

info2 = FunctionInfo.from_fn(tool_noret)
print()
print("=== tool with NO return annotation → single_output_schema ===")
print(info2.single_output_schema.model_json_schema() if info2.single_output_schema else None)

# Check if return annotation is lost by functools.wraps
import functools
@functools.wraps(tool_str)
async def wrapped(*args, **kwargs):
    return await tool_str(*args, **kwargs)

print()
print("=== wrapped with functools.wraps signature ===", inspect.signature(wrapped))
print("=== wrapped annotations ===", wrapped.__annotations__)
info3 = FunctionInfo.from_fn(wrapped)
print("=== wrapped single_output_schema ===")
print(info3.single_output_schema.model_json_schema() if info3.single_output_schema else None)
