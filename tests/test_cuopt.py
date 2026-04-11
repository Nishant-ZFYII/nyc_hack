"""
Test cuOpt VRP integration on DGX Spark.

Run: python tests/test_cuopt.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time


def test_cuopt_import():
    """Test that cuOpt is importable."""
    try:
        from cuopt import routing
        import cudf
        print("✅ cuOpt imported successfully")
        return True
    except ImportError as e:
        print(f"❌ cuOpt import failed: {e}")
        return False


def test_cuopt_simple_vrp():
    """Test a simple VRP: allocate 200 people across 5 shelter sites."""
    try:
        from cuopt import routing
        import cudf
        import numpy as np

        # 5 shelter locations + 1 depot (crisis epicenter)
        # Depot at Flatbush, shelters spread across Brooklyn
        locations = [
            (40.6501, -73.9496),  # depot: Flatbush centroid
            (40.6682, -73.9796),  # shelter 1
            (40.6782, -73.9496),  # shelter 2
            (40.6402, -73.9196),  # shelter 3
            (40.6582, -73.9696),  # shelter 4
            (40.6301, -73.9896),  # shelter 5
        ]

        n = len(locations)
        LAT_M, LON_M = 111_320, 85_390

        # Build cost matrix (travel time in minutes, walking 80m/min)
        cost = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                d = np.sqrt(
                    ((locations[i][0] - locations[j][0]) * LAT_M) ** 2 +
                    ((locations[i][1] - locations[j][1]) * LON_M) ** 2
                )
                cost[i][j] = d / 80.0  # minutes

        cost_df = cudf.DataFrame(cost)

        # 5 vehicles (one per shelter), capacity 50 each
        n_vehicles = 5
        data_model = routing.DataModel(n, n_vehicles)
        data_model.add_cost_matrix(cost_df)

        # Demands: depot=0, shelters get people
        demands = [0, 40, 40, 40, 40, 40]  # 200 total
        data_model.set_order_demands(demands)

        # Vehicle capacities
        capacities = [50, 50, 50, 50, 50]
        data_model.set_vehicle_capacities(capacities)

        # Solve
        solver = routing.SolverSettings()
        solver.set_time_limit(2.0)

        t0 = time.time()
        result = routing.Solve(data_model, solver)
        solve_time = time.time() - t0

        status = result.get_status()
        if status == 0:
            print(f"✅ cuOpt VRP solved in {solve_time:.3f}s")
            routes = result.get_routes()
            print(f"   Routes: {routes}")
            return True
        else:
            print(f"⚠️ cuOpt returned status {status} (non-zero = suboptimal/infeasible)")
            return False

    except Exception as e:
        print(f"❌ cuOpt VRP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cold_emergency_cuopt():
    """Test the full cold_emergency simulation with cuOpt."""
    try:
        from pipeline.executor import execute, load_state

        plan = {
            "intent": "simulate",
            "scenario": "cold_emergency",
            "params": {
                "borough": "BK",
                "people_displaced": 200,
                "temperature_f": 15,
            }
        }

        t0 = time.time()
        result = execute(plan)
        exec_time = time.time() - t0

        optimizer = result.get("optimizer", "unknown")
        allocation = result.get("allocation", [])
        shelters = result.get("available_shelters", [])
        overflow = result.get("overflow_sites", [])

        print(f"✅ Cold emergency simulation completed in {exec_time:.3f}s")
        print(f"   Optimizer: {optimizer}")
        print(f"   Shelters allocated: {len(allocation)}")
        print(f"   Overflow sites: {len(overflow)}")
        print(f"   Recommendation: {result.get('recommendation', '')[:100]}...")

        if allocation:
            total_assigned = sum(a.get("assigned_people", 0) for a in allocation)
            print(f"   Total people assigned: {total_assigned}")
            for a in allocation[:3]:
                print(f"     - {a.get('name', '?')}: {a.get('assigned_people', 0)} people")

        return True

    except Exception as e:
        print(f"❌ Cold emergency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_migrant_allocation_cuopt():
    """Test migrant allocation simulation with cuOpt."""
    try:
        from pipeline.executor import execute

        plan = {
            "intent": "simulate",
            "scenario": "migrant_allocation",
            "params": {
                "people": 80,
                "languages": ["Spanish", "Mandarin"],
                "needs": ["shelter", "food_bank", "school"],
            }
        }

        t0 = time.time()
        result = execute(plan)
        exec_time = time.time() - t0

        optimizer = result.get("optimizer", "unknown")
        allocation = result.get("allocation", [])

        print(f"✅ Migrant allocation completed in {exec_time:.3f}s")
        print(f"   Optimizer: {optimizer}")
        print(f"   Sites allocated: {len(allocation)}")
        print(f"   Recommendation: {result.get('recommendation', '')[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Migrant allocation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("cuOpt VRP Integration Tests")
    print("=" * 60)

    results = {}
    results["import"] = test_cuopt_import()
    print()
    results["simple_vrp"] = test_cuopt_simple_vrp()
    print()
    results["cold_emergency"] = test_cold_emergency_cuopt()
    print()
    results["migrant_allocation"] = test_migrant_allocation_cuopt()

    print()
    print("=" * 60)
    print("Summary:")
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n{sum(results.values())}/{len(results)} passed")
