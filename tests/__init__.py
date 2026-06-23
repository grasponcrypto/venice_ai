"""Test suite for the Venice AI Home Assistant integration.

Implements TEST-1 (unit tests) and TEST-2 (integration-style tests with a
mocked Venice AI client). These tests focus on the pure domain logic that can
be exercised without a running Home Assistant instance:

- ``VeniceAIMetrics`` telemetry counters (LOW-4)
- HTTP error categorization in ``client.py``
- The ``venice_api`` service layer, including streaming accumulation and
  tool-call fragment reassembly (ARCH-1/ARCH-2, MED-3)

Run with::

    pytest tests/ -v
"""
