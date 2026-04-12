"""Inspect nat's FunctionGroup.add_function signature + how tools are registered."""
from nat.builder.function import FunctionGroup
import inspect

print("=== FunctionGroup.add_function signature ===")
print(inspect.signature(FunctionGroup.add_function))
print()
print("=== add_function source ===")
print(inspect.getsource(FunctionGroup.add_function)[:3000])
print()
print("=== FunctionGroup other methods ===")
for m in dir(FunctionGroup):
    if not m.startswith("_") and callable(getattr(FunctionGroup, m, None)):
        print(" -", m)
