import argparse
import copy
import importlib
import inspect
import math
import os
import tempfile
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pypower.api import case300, ppoption, runopf, rundcopf
from pypower.idx_cost import COST, MODEL, NCOST
from pypower.idx_gen import PMAX, PMIN


@dataclass
class BenchRecord:
    framework: str
    task: str
    times: List[float]
    success: Optional[bool]
    objective: Optional[float]
    error: Optional[str] = None

    @property
    def avg(self) -> float:
        if not self.times:
            return math.nan
        return mean(self.times)

    @property
    def std(self) -> float:
        if len(self.times) <= 1:
            return 0.0 if self.times else math.nan
        return stdev(self.times)


def _fmt_bool(x: Optional[bool]) -> str:
    if x is None:
        return "N/A"
    return "True" if x else "False"


def _fmt_float(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float) and math.isnan(x):
        return "N/A"
    return f"{x:.{digits}f}"


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _filter_supported_kwargs(fn: Callable, kwargs: Dict) -> Dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _run_benchmark(
    framework: str,
    task: str,
    runner: Callable[[], Tuple[bool, float]],
    n_runs: int,
    warmup: int,
) -> BenchRecord:
    try:
        for _ in range(warmup):
            runner()
    except Exception as exc:  # pragma: no cover - runtime dependency paths
        return BenchRecord(framework, task, [], False, None, f"{type(exc).__name__}: {exc}")

    times: List[float] = []
    success: Optional[bool] = None
    objective: Optional[float] = None

    for _ in range(n_runs):
        try:
            t0 = time.perf_counter()
            success, objective = runner()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as exc:  # pragma: no cover - runtime dependency paths
            return BenchRecord(framework, task, times, False, objective, f"{type(exc).__name__}: {exc}")

    return BenchRecord(framework, task, times, success, objective, None)


def _pypower_bench(case: Dict, n_runs: int, warmup: int) -> List[BenchRecord]:
    def run_ac() -> Tuple[bool, float]:
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        result = runopf(copy.deepcopy(case), ppopt)
        return bool(result.get("success", False)), _safe_float(result.get("f"))

    def run_dc() -> Tuple[bool, float]:
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        result = rundcopf(copy.deepcopy(case), ppopt)
        return bool(result.get("success", False)), _safe_float(result.get("f"))

    return [
        _run_benchmark("pypower", "ACOPF", run_ac, n_runs, warmup),
        _run_benchmark("pypower", "DCOPF", run_dc, n_runs, warmup),
    ]


def _resolve_pandapower_from_ppc():
    candidates = [
        "pandapower.converter:from_ppc",
        "pandapower.converter.pypower:from_ppc",
        "pandapower.converter.pypower.from_ppc:from_ppc",
        "pandapower.converter.matpower:from_ppc",
        "pandapower.converter.matpower.from_ppc:from_ppc",
    ]
    errors = []
    for candidate in candidates:
        module_name, attr = candidate.split(":")
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, attr, None)
            if callable(fn):
                return fn
        except Exception as exc:
            errors.append(f"{candidate} -> {type(exc).__name__}: {exc}")
    joined = "; ".join(errors) if errors else "no candidate function found"
    raise ImportError(f"Unable to locate pandapower `from_ppc` converter. Tried: {joined}")


def _build_pandapower_net(case: Dict):
    kwargs = {"f_hz": 60.0, "validate_conversion": False}
    try:
        from_ppc = _resolve_pandapower_from_ppc()
        return from_ppc(copy.deepcopy(case), **_filter_supported_kwargs(from_ppc, kwargs))
    except ImportError:
        from pandapower.converter import from_mpc
        from scipy.io import savemat

        # Compatibility fallback for pandapower versions that expose only from_mpc.
        fd, tmp_path = tempfile.mkstemp(suffix=".mat", prefix="ppc_case_")
        os.close(fd)
        try:
            savemat(tmp_path, {"mpc": copy.deepcopy(case)})
            mpc_kwargs = {
                "f_hz": 60.0,
                "casename_mpc_file": "mpc",
                "validate_conversion": False,
            }
            return from_mpc(tmp_path, **_filter_supported_kwargs(from_mpc, mpc_kwargs))
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _pandapower_bench(case: Dict, n_runs: int, warmup: int) -> List[BenchRecord]:
    import pandapower as pp

    base_net = _build_pandapower_net(case)

    def run_ac() -> Tuple[bool, float]:
        attempts = [
            {"init": "flat"},
            {"init": "pf"},
            {"init": "dc"},
        ]
        last_exc: Optional[Exception] = None
        for attempt in attempts:
            net = copy.deepcopy(base_net)
            kwargs = {
                "verbose": False,
                "calculate_voltage_angles": True,
                "suppress_warnings": True,
                "delta": 1e-10,
                "OPF_VIOLATION": 1e-5,
                "PDIPM_MAX_IT": 300,
            }
            kwargs.update(attempt)
            try:
                if attempt["init"] == "pf":
                    pf_kwargs = {
                        "calculate_voltage_angles": True,
                        "init": "flat",
                        "max_iteration": 50,
                    }
                    pp.runpp(net, **_filter_supported_kwargs(pp.runpp, pf_kwargs))
                pp.runopp(net, **_filter_supported_kwargs(pp.runopp, kwargs))
                success = bool(net.get("OPF_converged", False))
                objective = _safe_float(net.get("res_cost", math.nan))
                return success, objective
            except Exception as exc:  # pragma: no cover - runtime dependency paths
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("pandapower ACOPF failed without explicit exception")

    def run_dc() -> Tuple[bool, float]:
        net = copy.deepcopy(base_net)
        kwargs = {"verbose": False, "suppress_warnings": True}
        pp.rundcopp(net, **_filter_supported_kwargs(pp.rundcopp, kwargs))
        success = bool(net.get("OPF_converged", False))
        objective = _safe_float(net.get("res_cost", math.nan))
        return success, objective

    return [
        _run_benchmark("pandapower", "ACOPF", run_ac, n_runs, warmup),
        _run_benchmark("pandapower", "DCOPF", run_dc, n_runs, warmup),
    ]


def _build_pypsa_network(case: Dict):
    import pypsa

    network = pypsa.Network()
    network.import_from_pypower_ppc(copy.deepcopy(case))
    if len(network.snapshots) == 0:
        network.set_snapshots(["now"])
    _prepare_pypsa_generators(network, case)
    _apply_pypsa_generator_costs(network, case)
    _ensure_pypsa_snapshot_weightings(network)
    return network


def _prepare_pypsa_generators(network, case: Dict) -> None:
    if len(network.generators.index) == 0:
        return

    gen = case.get("gen")
    pmax_case = np.array([], dtype=float)
    pmin_case = np.array([], dtype=float)
    if gen is not None and gen.shape[1] > PMAX:
        pmax_case = np.asarray(gen[:, PMAX], dtype=float)
    if gen is not None and gen.shape[1] > PMIN:
        pmin_case = np.asarray(gen[:, PMIN], dtype=float)

    n = len(network.generators.index)
    fallback_nom = np.ones(n, dtype=float)
    m = min(n, pmax_case.size)
    if m > 0:
        fallback_nom[:m] = np.maximum(pmax_case[:m], 1e-3)

    if "p_nom" not in network.generators.columns:
        network.generators.loc[:, "p_nom"] = fallback_nom
    else:
        p_nom = np.asarray(network.generators["p_nom"], dtype=float)
        bad_nom = ~np.isfinite(p_nom) | (p_nom <= 0)
        p_nom[bad_nom] = fallback_nom[bad_nom]
        network.generators.loc[:, "p_nom"] = p_nom

    p_nom = np.asarray(network.generators["p_nom"], dtype=float)
    p_min_pu = np.zeros(n, dtype=float)
    p_max_pu = np.ones(n, dtype=float)
    k = min(n, pmax_case.size, pmin_case.size)
    if k > 0:
        denom = np.maximum(p_nom[:k], 1e-6)
        p_min_pu[:k] = np.clip(pmin_case[:k] / denom, 0.0, 1.0)
        p_max_pu[:k] = np.maximum(p_min_pu[:k], np.clip(pmax_case[:k] / denom, 0.0, 2.0))

    network.generators.loc[:, "p_min_pu"] = p_min_pu
    network.generators.loc[:, "p_max_pu"] = p_max_pu


def _poly_marginal_cost(coeffs: np.ndarray, p_ref: float) -> float:
    degree = len(coeffs) - 1
    if degree <= 0:
        return 0.0
    marginal = 0.0
    for i, coeff in enumerate(coeffs[:-1]):
        power = degree - i
        marginal += power * float(coeff) * (p_ref ** (power - 1))
    return marginal


def _estimate_marginal_costs_from_ppc(case: Dict) -> np.ndarray:
    gen = case.get("gen")
    gencost = case.get("gencost")
    if gen is None:
        return np.array([], dtype=float)

    n_gen = int(gen.shape[0])
    costs = np.ones(n_gen, dtype=float)
    if gencost is None:
        return costs

    n_rows = min(n_gen, int(gencost.shape[0]))
    for i in range(n_rows):
        row = np.asarray(gencost[i], dtype=float)
        model = int(round(row[MODEL])) if row.size > MODEL else 2
        ncost = int(round(row[NCOST])) if row.size > NCOST else 0
        pmin = _safe_float(gen[i, PMIN], 0.0)
        pmax = _safe_float(gen[i, PMAX], pmin)
        p_ref = 0.5 * (pmin + pmax)
        marginal = 1.0

        if model == 2 and ncost > 0:
            coeffs = row[COST: COST + ncost]
            if coeffs.size >= 2:
                marginal = _poly_marginal_cost(coeffs, p_ref)
            elif coeffs.size == 1:
                marginal = 0.0
        elif model == 1 and ncost >= 2:
            xy = row[COST: COST + 2 * ncost]
            if xy.size >= 4 and abs(xy[2] - xy[0]) > 1e-9:
                marginal = (xy[3] - xy[1]) / (xy[2] - xy[0])

        if not np.isfinite(marginal):
            marginal = 1.0
        costs[i] = float(marginal)

    return costs


def _apply_pypsa_generator_costs(network, case: Dict) -> None:
    n_pypsa_gen = len(network.generators.index)
    if n_pypsa_gen == 0:
        return

    estimated = _estimate_marginal_costs_from_ppc(case)
    marginal_costs = np.ones(n_pypsa_gen, dtype=float)
    n = min(n_pypsa_gen, estimated.size)
    if n > 0:
        marginal_costs[:n] = estimated[:n]

    network.generators.loc[:, "marginal_cost"] = marginal_costs
    if len(network.snapshots) > 0:
        mc_df = pd.DataFrame(
            np.repeat(marginal_costs.reshape(1, -1), len(network.snapshots), axis=0),
            index=network.snapshots,
            columns=network.generators.index,
        )
        network.generators_t["marginal_cost"] = mc_df


def _ensure_pypsa_snapshot_weightings(network) -> None:
    if len(network.snapshots) == 0:
        return
    if not hasattr(network, "snapshot_weightings"):
        return
    cols = list(network.snapshot_weightings.columns)
    if "objective" in cols:
        network.snapshot_weightings.loc[network.snapshots, "objective"] = 1.0


def _solve_pypsa_linear(network, solver_name: Optional[str]) -> Tuple[bool, float]:
    solver_kwargs = {"snapshots": network.snapshots}
    if solver_name:
        solver_kwargs["solver_name"] = solver_name

    if hasattr(network, "optimize") and callable(network.optimize):
        result = network.optimize(**_filter_supported_kwargs(network.optimize, solver_kwargs))
        if isinstance(result, tuple) and len(result) >= 2:
            status = str(result[0]).lower()
            condition = str(result[1]).lower()
            success = ("ok" in status or "optimal" in status) and "optimal" in condition
        elif isinstance(result, bool):
            success = result
        else:
            success = True
        return success, _safe_float(getattr(network, "objective", math.nan))

    if hasattr(network, "lopf") and callable(network.lopf):
        kwargs = _filter_supported_kwargs(network.lopf, solver_kwargs)
        status = network.lopf(network.snapshots, **kwargs)
        success = bool(status)
        return success, _safe_float(getattr(network, "objective", math.nan))

    raise RuntimeError("Neither `network.optimize` nor `network.lopf` is available in this PyPSA version")


def _solve_pypsa_ac(network, solver_name: Optional[str]) -> Tuple[bool, float]:
    optimize_accessor = getattr(network, "optimize", None)
    if optimize_accessor is None:
        raise RuntimeError("PyPSA ACOPF path unavailable: missing `network.optimize` accessor")

    ac_runner = getattr(optimize_accessor, "optimize_and_run_non_linear_powerflow", None)
    if ac_runner is None or not callable(ac_runner):
        raise RuntimeError(
            "PyPSA ACOPF is not directly available in this version "
            "(missing `optimize_and_run_non_linear_powerflow`)"
        )

    solver_kwargs = {"snapshots": network.snapshots}
    if solver_name:
        solver_kwargs["solver_name"] = solver_name

    result = ac_runner(**_filter_supported_kwargs(ac_runner, solver_kwargs))
    if isinstance(result, tuple) and len(result) > 0:
        success = str(result[0]).lower() in {"ok", "optimal", "true"}
    else:
        success = True
    return success, _safe_float(getattr(network, "objective", math.nan))


def _pypsa_bench(case: Dict, n_runs: int, warmup: int, solver_name: Optional[str]) -> List[BenchRecord]:
    base_network = _build_pypsa_network(case)

    def run_dc() -> Tuple[bool, float]:
        network = copy.deepcopy(base_network)
        return _solve_pypsa_linear(network, solver_name=solver_name)

    def run_ac() -> Tuple[bool, float]:
        network = copy.deepcopy(base_network)
        return _solve_pypsa_ac(network, solver_name=solver_name)

    return [
        _run_benchmark("pypsa", "ACOPF", run_ac, n_runs, warmup),
        _run_benchmark("pypsa", "DCOPF", run_dc, n_runs, warmup),
    ]


def _print_report(records: List[BenchRecord], n_runs: int) -> None:
    print("=== IEEE 300-bus OPF benchmark (follow pypower logic) ===")
    print(f"Runs per task: {n_runs}")
    print()
    print(f"{'Framework':<12} {'Task':<8} {'Success':<7} {'Objective':>14} {'Avg Time(s)':>12} {'Std(s)':>10}")
    print("-" * 72)
    for record in records:
        print(
            f"{record.framework:<12} "
            f"{record.task:<8} "
            f"{_fmt_bool(record.success):<7} "
            f"{_fmt_float(record.objective):>14} "
            f"{_fmt_float(record.avg):>12} "
            f"{_fmt_float(record.std):>10}"
        )
        if record.error:
            print(f"{'':<12} {'':<8} error: {record.error}")
    print()

    by_framework: Dict[str, Dict[str, BenchRecord]] = {}
    for record in records:
        by_framework.setdefault(record.framework, {})[record.task] = record

    for framework, mapping in by_framework.items():
        ac = mapping.get("ACOPF")
        dc = mapping.get("DCOPF")
        if not ac or not dc:
            continue
        if math.isnan(ac.avg) or math.isnan(dc.avg) or dc.avg <= 0:
            ratio = "N/A"
        else:
            ratio = f"{ac.avg / dc.avg:.2f}x"
        print(f"{framework} speedup (AC/DC): {ratio}")


def benchmark(n_runs: int = 5, warmup: int = 1, pypsa_solver: Optional[str] = None) -> Dict[str, List[BenchRecord]]:
    case = case300()

    records: List[BenchRecord] = []
    records.extend(_pypower_bench(case, n_runs=n_runs, warmup=warmup))

    try:
        records.extend(_pandapower_bench(case, n_runs=n_runs, warmup=warmup))
    except Exception as exc:  # pragma: no cover - runtime dependency paths
        err = f"{type(exc).__name__}: {exc}"
        records.extend(
            [
                BenchRecord("pandapower", "ACOPF", [], None, None, err),
                BenchRecord("pandapower", "DCOPF", [], None, None, err),
            ]
        )

    try:
        records.extend(_pypsa_bench(case, n_runs=n_runs, warmup=warmup, solver_name=pypsa_solver))
    except Exception as exc:  # pragma: no cover - runtime dependency paths
        err = f"{type(exc).__name__}: {exc}"
        records.extend(
            [
                BenchRecord("pypsa", "ACOPF", [], None, None, err),
                BenchRecord("pypsa", "DCOPF", [], None, None, err),
            ]
        )

    _print_report(records, n_runs=n_runs)
    return {"records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE300 ACOPF/DCOPF timing benchmark across pypower, pandapower, pypsa")
    parser.add_argument("--n-runs", type=int, default=5, help="Benchmark runs per OPF task")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per OPF task")
    parser.add_argument(
        "--pypsa-solver",
        type=str,
        default=None,
        help="Optional solver name for PyPSA (e.g. highs, glpk, gurobi)",
    )
    args = parser.parse_args()

    benchmark(n_runs=args.n_runs, warmup=args.warmup, pypsa_solver=args.pypsa_solver)
