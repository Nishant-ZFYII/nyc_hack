"""Inspect Middleware base class + logging middleware as template."""
import inspect
from nat.middleware.middleware import Middleware

print("=== Middleware class source ===")
print(inspect.getsource(Middleware)[:3000])
print()

from nat.middleware.logging import logging_middleware
print("=== logging_middleware module ===")
print(inspect.getsource(logging_middleware)[:3500])
