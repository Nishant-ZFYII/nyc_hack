"""Sanity tests for the pure-Python scenario engine.

Run with: pytest tests/test_scenarios.py
"""
import math

import pytest

from pipeline import scenarios


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


@pytest.mark.parametrize("name,expected_phase", [
    ("cold_emergency", "cold_emergency"),
    ("migrant_bus", "migrant_bus"),
    ("reset", "reset"),
])
def test_scenario_shapes(name, expected_phase):
    result = scenarios.run(name)
    assert result["phase"] == expected_phase
    assert "demand" in result
    assert "sites" in result
    assert "arcs" in result
    assert "stats" in result


def test_cold_emergency_populates_arcs_and_sites():
    r = scenarios.cold_emergency(n_people=50, borough="Bronx", seed=1)
    assert len(r["arcs"]) == 50, "every demand point gets an arc"
    assert len(r["sites"]) > 0, "at least one site should receive people"
    for a in r["arcs"]:
        assert len(a["from"]) == 2 and len(a["to"]) == 2
        for v in a["from"] + a["to"]:
            assert _finite(v)
        assert a["color"] and len(a["color"]) == 4


def test_migrant_bus_populates():
    r = scenarios.migrant_bus(n_people=40, seed=2)
    assert len(r["arcs"]) == 40
    assert r["stats"]["served"] >= 0
    assert r["stats"]["unmet"] == 40 - r["stats"]["served"]


def test_reset_is_empty():
    r = scenarios.reset()
    assert r["demand"] == []
    assert r["sites"] == []
    assert r["arcs"] == []


def test_stats_contain_numeric_elapsed():
    r = scenarios.cold_emergency(n_people=20, seed=3)
    assert "elapsed_ms" in r["stats"]
    assert _finite(r["stats"]["avg_km"])
    assert r["stats"]["served"] + r["stats"]["unmet"] == 20


def test_vulnerability_hexes():
    hexes = scenarios._vulnerability_hex_grid()
    assert len(hexes) > 100, "expect a dense hex grid over NYC bbox"
    for h in hexes[:20]:
        assert _finite(h["lat"]) and _finite(h["lon"])
        assert 40.0 < h["lat"] < 41.5
        assert -75.0 < h["lon"] < -73.0
        assert 0 < h["weight"] <= 3.0
