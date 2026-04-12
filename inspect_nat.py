"""Inspect nat's middleware API for hooking tool calls."""
import sys, inspect
sys.path.insert(0, ".")

from nat.builder.function import FunctionGroup

print("=== configure_middleware signature ===")
print(inspect.signature(FunctionGroup.configure_middleware))
print()
print("=== configure_middleware source ===")
print(inspect.getsource(FunctionGroup.configure_middleware))
print()

# Look for middleware base class / interface
try:
    from nat.builder import middleware as mw_mod
    print("=== nat.builder.middleware ===")
    for name in dir(mw_mod):
        if not name.startswith("_"):
            print(" -", name)
except ImportError as e:
    print("no nat.builder.middleware:", e)

# Maybe middleware is elsewhere
import pkgutil, nat
print()
print("=== nat submodules with 'middleware' ===")
for m in pkgutil.walk_packages(nat.__path__, nat.__name__ + "."):
    if "middleware" in m.name.lower():
        print(" -", m.name)
