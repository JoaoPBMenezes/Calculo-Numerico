# -*- coding: utf-8 -*-
"""
placa_termica.py
============================================================
PARTE TÉRMICA DO GÊMEO DIGITAL
============================================================

Este script implementa APENAS a parte térmica, sem integrar com a parte
hidráulica ainda.

Bibliotecas necessárias no Windows:
    pip install numpy matplotlib scipy

Como rodar:
    python placa_termica.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


Number = Union[int, float]
ScalarOrCallable = Union[Number, Callable[[float], float]]
FieldFunction2D = Callable[[float, float], float]


def ij2n(i: int, j: int, Nx: int) -> int:
    """Mapeia índice bidimensional (i,j) para índice global n."""
    return i + j * Nx


def n2ij(n: int, Nx: int) -> Tuple[int, int]:
    """Mapeia índice global n para índice bidimensional (i,j)."""
    j = n // Nx
    i = n % Nx
    return i, j


def to_callable(value: ScalarOrCallable) -> Callable[[float], float]:
    """Converte constante ou função em função callable."""
    if callable(value):
        return value
    return lambda x: float(value)


def constant_source(value: float) -> FieldFunction2D:
    """Fonte constante f(x,y)=value."""
    return lambda x, y: float(value)


def variable_k_function(Lx: float, Ly: float) -> FieldFunction2D:
    """Condutividade variável do exercício 3."""
    def _k(x: float, y: float) -> float:
        return 0.2 + 0.05 * np.sin(3.0 * np.pi * x / Lx) * np.sin(3.0 * np.pi * y / Ly)
    return _k


def top_bottom_temperature_function(Lx: float) -> Callable[[float], float]:
    """T(x)=10+20*x/Lx para bordas superior e inferior."""
    return lambda x: 10.0 + 20.0 * x / Lx


@dataclass
class ThermalPlateConfig:
    """Configuração do problema térmico."""
    Lx: float = 0.02
    Ly: float = 0.01
    Nx: int = 101
    Ny: int = 51

    TL: ScalarOrCallable = 10.0
    TR: ScalarOrCallable = 30.0
    TB: ScalarOrCallable = field(default_factory=lambda: top_bottom_temperature_function(0.02))
    TT: ScalarOrCallable = field(default_factory=lambda: top_bottom_temperature_function(0.02))

    source_function: FieldFunction2D = field(default_factory=lambda: constant_source(5.0e5))

    use_variable_k: bool = False
    k_constant: float = 0.25
    k_function: Optional[FieldFunction2D] = None

    use_circle_constraint: bool = False
    circle_center_x: float = 0.75 * 0.02
    circle_center_y: float = 0.50 * 0.01
    circle_radius: float = 0.002
    TC: float = 30.0

    solver_mode: str = "sparse"
    preserve_symmetry: bool = True

    make_plots: bool = True
    print_matrix_info: bool = True
    contour_levels: int = 20

    def __post_init__(self) -> None:
        if not callable(self.TB):
            self.TB = to_callable(self.TB)
        if not callable(self.TT):
            self.TT = to_callable(self.TT)
        if not callable(self.TL):
            self.TL = to_callable(self.TL)
        if not callable(self.TR):
            self.TR = to_callable(self.TR)
        if self.k_function is None and self.use_variable_k:
            self.k_function = variable_k_function(self.Lx, self.Ly)


class BaseLinearSystemSolver:
    """Classe base para problemas físicos que geram A x = b."""

    def __init__(self) -> None:
        self.last_result: Optional[Dict[str, Any]] = None

    def solve_linear_system_dense(self, A_mod: np.ndarray, b_mod: np.ndarray) -> np.ndarray:
        """Resolve sistema denso usando np.linalg.solve."""
        return np.linalg.solve(A_mod, b_mod)

    def solve_linear_system_sparse(self, A_mod_sparse: sparse.csr_matrix, b_mod: np.ndarray) -> np.ndarray:
        """Resolve sistema esparso usando spsolve."""
        return spsolve(A_mod_sparse, b_mod)


class ThermalPlateSolver(BaseLinearSystemSolver):
    """Solver da placa térmica estacionária."""

    def __init__(self, cfg: ThermalPlateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.x = np.linspace(0.0, self.cfg.Lx, self.cfg.Nx)
        self.y = np.linspace(0.0, self.cfg.Ly, self.cfg.Ny)
        self.hx = self.cfg.Lx / (self.cfg.Nx - 1)
        self.hy = self.cfg.Ly / (self.cfg.Ny - 1)
        self.nunk = self.cfg.Nx * self.cfg.Ny

    def get_k(self, x: float, y: float) -> float:
        """Retorna k(x,y) ou k constante."""
        if self.cfg.use_variable_k:
            if self.cfg.k_function is None:
                raise ValueError("use_variable_k=True, mas k_function não foi definida.")
            return float(self.cfg.k_function(x, y))
        return float(self.cfg.k_constant)

    def get_source(self, x: float, y: float) -> float:
        """Retorna f(x,y)."""
        return float(self.cfg.source_function(x, y))

    def is_inside_circle_constraint(self, x: float, y: float) -> bool:
        """Verifica se o ponto está na região circular de temperatura fixa."""
        if not self.cfg.use_circle_constraint:
            return False
        dx = x - self.cfg.circle_center_x
        dy = y - self.cfg.circle_center_y
        return dx * dx + dy * dy <= self.cfg.circle_radius ** 2

    def get_dirichlet_value(self, i: int, j: int) -> Optional[float]:
        """Retorna temperatura prescrita em bordas e círculo, se houver."""
        x = self.x[i]
        y = self.y[j]
        if i == 0:
            return float(self.cfg.TL(y))
        if i == self.cfg.Nx - 1:
            return float(self.cfg.TR(y))
        if j == 0:
            return float(self.cfg.TB(x))
        if j == self.cfg.Ny - 1:
            return float(self.cfg.TT(x))
        if self.is_inside_circle_constraint(x, y):
            return float(self.cfg.TC)
        return None

    def assembly(self, matrix_mode: str = "dense") -> Union[np.ndarray, sparse.lil_matrix]:
        """Monta a matriz A do problema térmico."""
        if matrix_mode == "dense":
            A = np.zeros((self.nunk, self.nunk), dtype=float)
        elif matrix_mode == "sparse":
            A = sparse.lil_matrix((self.nunk, self.nunk), dtype=float)
        else:
            raise ValueError("matrix_mode deve ser 'dense' ou 'sparse'.")

        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)
                if self.get_dirichlet_value(i, j) is not None:
                    continue

                Ie = ij2n(i + 1, j, self.cfg.Nx)
                Iw = ij2n(i - 1, j, self.cfg.Nx)
                In = ij2n(i, j + 1, self.cfg.Nx)
                Is = ij2n(i, j - 1, self.cfg.Nx)

                xc = self.x[i]
                yc = self.y[j]
                ke = self.get_k(xc + 0.5 * self.hx, yc)
                kw = self.get_k(xc - 0.5 * self.hx, yc)
                kn = self.get_k(xc, yc + 0.5 * self.hy)
                ks = self.get_k(xc, yc - 0.5 * self.hy)

                aE = ke
                aW = kw
                aN = kn
                aS = ks
                aP = aE + aW + aN + aS

                A[Ic, Ic] = aP
                A[Ic, Ie] = -aE
                A[Ic, Iw] = -aW
                A[Ic, In] = -aN
                A[Ic, Is] = -aS
        return A

    def build_rhs(self) -> np.ndarray:
        """Monta o vetor b com a fonte térmica."""
        b = np.zeros(self.nunk, dtype=float)
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)
                if self.get_dirichlet_value(i, j) is not None:
                    b[Ic] = 0.0
                else:
                    b[Ic] = self.hx * self.hy * self.get_source(self.x[i], self.y[j])
        return b

    def apply_dirichlet_bc_dense(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica Dirichlet em matriz densa."""
        A_mod = A.copy()
        b_mod = b.copy()
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Tpresc = self.get_dirichlet_value(i, j)
                if Tpresc is None:
                    continue
                Ic = ij2n(i, j, self.cfg.Nx)
                if self.cfg.preserve_symmetry:
                    b_mod -= A_mod[:, Ic] * Tpresc
                    A_mod[Ic, :] = 0.0
                    A_mod[:, Ic] = 0.0
                    A_mod[Ic, Ic] = 1.0
                    b_mod[Ic] = Tpresc
                else:
                    A_mod[Ic, :] = 0.0
                    A_mod[Ic, Ic] = 1.0
                    b_mod[Ic] = Tpresc
        return A_mod, b_mod

    def apply_dirichlet_bc_sparse(self, A: sparse.lil_matrix, b: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Aplica Dirichlet em matriz esparsa."""
        A_mod = A.copy().tolil()
        b_mod = b.copy()
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Tpresc = self.get_dirichlet_value(i, j)
                if Tpresc is None:
                    continue
                Ic = ij2n(i, j, self.cfg.Nx)
                if self.cfg.preserve_symmetry:
                    col = A_mod[:, Ic].toarray().ravel()
                    b_mod -= col * Tpresc
                    A_mod.rows[Ic] = []
                    A_mod.data[Ic] = []
                    for row in range(A_mod.shape[0]):
                        row_cols = A_mod.rows[row]
                        row_data = A_mod.data[row]
                        for idx in range(len(row_cols) - 1, -1, -1):
                            if row_cols[idx] == Ic:
                                row_cols.pop(idx)
                                row_data.pop(idx)
                    A_mod[Ic, Ic] = 1.0
                    b_mod[Ic] = Tpresc
                else:
                    A_mod.rows[Ic] = []
                    A_mod.data[Ic] = []
                    A_mod[Ic, Ic] = 1.0
                    b_mod[Ic] = Tpresc
        return A_mod.tocsr(), b_mod

    def solve_system(self, matrix_mode: Optional[str] = None) -> Dict[str, Any]:
        """Resolve o problema térmico completo."""
        if matrix_mode is None:
            matrix_mode = self.cfg.solver_mode

        t0 = time.perf_counter()
        A = self.assembly(matrix_mode=matrix_mode)
        b = self.build_rhs()
        t1 = time.perf_counter()

        if matrix_mode == "dense":
            A_mod, b_mod = self.apply_dirichlet_bc_dense(A, b)
        else:
            A_mod, b_mod = self.apply_dirichlet_bc_sparse(A, b)
        t2 = time.perf_counter()

        if matrix_mode == "dense":
            T_vec = self.solve_linear_system_dense(A_mod, b_mod)
        else:
            T_vec = self.solve_linear_system_sparse(A_mod, b_mod)
        t3 = time.perf_counter()

        T_grid = T_vec.reshape((self.cfg.Ny, self.cfg.Nx))
        Tmax = float(np.max(T_grid))
        Tmean = float(np.mean(T_grid))
        center_j = self.cfg.Ny // 2
        T_centerline = T_grid[center_j, :].copy()

        result = {
            "matrix_mode": matrix_mode,
            "A": A,
            "A_mod": A_mod,
            "b": b,
            "b_mod": b_mod,
            "T_vec": T_vec,
            "T_grid": T_grid,
            "x": self.x.copy(),
            "y": self.y.copy(),
            "Tmax": Tmax,
            "Tmean": Tmean,
            "centerline_x": self.x.copy(),
            "centerline_T": T_centerline,
            "assembly_time": t1 - t0,
            "bc_time": t2 - t1,
            "solve_time": t3 - t2,
            "total_time": t3 - t0,
        }
        self.last_result = result
        return result

    def print_inputs_summary(self) -> None:
        """Imprime entradas do problema."""
        print("=" * 100)
        print("ENTRADAS UTILIZADAS NA PLACA TÉRMICA")
        print("=" * 100)
        print(f"Lx [m]                         : {self.cfg.Lx}")
        print(f"Ly [m]                         : {self.cfg.Ly}")
        print(f"Nx                             : {self.cfg.Nx}")
        print(f"Ny                             : {self.cfg.Ny}")
        print(f"nunk = Nx*Ny                   : {self.nunk}")
        print(f"hx [m]                         : {self.hx:.6e}")
        print(f"hy [m]                         : {self.hy:.6e}")
        print(f"solver_mode                    : {self.cfg.solver_mode}")
        print(f"preserve_symmetry              : {self.cfg.preserve_symmetry}")
        print(f"use_variable_k                 : {self.cfg.use_variable_k}")
        print(f"k_constant [W/(m.K)]           : {self.cfg.k_constant}")
        print(f"use_circle_constraint          : {self.cfg.use_circle_constraint}")
        print(f"TC [°C]                        : {self.cfg.TC}")
        print(f"circle_center_x [m]            : {self.cfg.circle_center_x}")
        print(f"circle_center_y [m]            : {self.cfg.circle_center_y}")
        print(f"circle_radius [m]              : {self.cfg.circle_radius}")
        print("=" * 100)

    def print_output_summary(self, result: Dict[str, Any]) -> None:
        """Imprime saídas principais."""
        print("=" * 100)
        print("RESULTADOS DA PLACA TÉRMICA")
        print("=" * 100)
        print(f"Modo da matriz                 : {result['matrix_mode']}")
        print(f"Tempo de montagem [s]          : {result['assembly_time']:.6e}")
        print(f"Tempo de BC [s]                : {result['bc_time']:.6e}")
        print(f"Tempo de solução [s]           : {result['solve_time']:.6e}")
        print(f"Tempo total [s]                : {result['total_time']:.6e}")
        print(f"Temperatura máxima [°C]        : {result['Tmax']:.6f}")
        print(f"Temperatura média [°C]         : {result['Tmean']:.6f}")
        print("=" * 100)

    def print_final_explanation(self, result: Dict[str, Any]) -> None:
        """Explica ao usuário o que aconteceu na execução."""
        print("=" * 100)
        print("COMENTÁRIO FINAL AO USUÁRIO")
        print("=" * 100)
        print("O programa acabou de:")
        print("1. Discretizar a placa térmica em uma malha 2D.")
        print("2. Montar um sistema linear A*T = b para as temperaturas nodais.")
        print("3. Aplicar temperaturas prescritas nas bordas e, se solicitado, no círculo interno.")
        print("4. Resolver o sistema linear com álgebra linear numérica.")
        print("5. Reconstruir o campo de temperaturas e calcular Tmax, Tmédia e eixo central.")
        print()
        print("Fundamento de álgebra linear usado:")
        print("- Cada nó da malha gera uma equação linear.")
        print("- As equações são empilhadas na matriz A.")
        print("- O vetor incógnita contém as temperaturas em todos os pontos.")
        print("- Resolver a placa térmica significa resolver A*T = b.")
        print()
        print(f"Nesta execução, a temperatura máxima foi {result['Tmax']:.6f} °C")
        print(f"e a temperatura média foi {result['Tmean']:.6f} °C.")
        print("=" * 100)

    def plot_temperature_contours(self, result: Dict[str, Any], title: str = "") -> None:
        """Plota curvas de nível da temperatura."""
        X, Y = np.meshgrid(result["x"], result["y"])
        T_grid = result["T_grid"]
        plt.figure(figsize=(10, 4.5))
        contour = plt.contourf(X, Y, T_grid, levels=self.cfg.contour_levels)
        plt.colorbar(contour, label="Temperatura [°C]")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(title if title else "Curvas de nível da temperatura")
        plt.tight_layout()
        plt.show()

    def plot_centerline(self, result: Dict[str, Any], title: str = "") -> None:
        """Plota temperatura ao longo do eixo central."""
        plt.figure(figsize=(9, 4))
        plt.plot(result["centerline_x"], result["centerline_T"], marker="o", markersize=3)
        plt.xlabel("x [m]")
        plt.ylabel("Temperatura [°C]")
        plt.title(title if title else "Temperatura ao longo do eixo central")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def print_matrix_info(self, result: Dict[str, Any]) -> None:
        """Imprime informações estruturais da matriz do sistema."""
        if not self.cfg.print_matrix_info:
            return
        A_mod = result["A_mod"]
        print("=" * 100)
        print("INFORMAÇÕES MATRICIAIS")
        print("=" * 100)
        print(f"Número de incógnitas           : {self.nunk}")
        if sparse.issparse(A_mod):
            nnz = A_mod.nnz
            density = nnz / (self.nunk * self.nunk)
            print("Tipo de matriz                 : esparsa")
            print(f"Número de não nulos            : {nnz}")
            print(f"Densidade da matriz            : {density:.6e}")
        else:
            nnz = np.count_nonzero(A_mod)
            density = nnz / (self.nunk * self.nunk)
            print("Tipo de matriz                 : densa")
            print(f"Número de não nulos            : {nnz}")
            print(f"Densidade da matriz            : {density:.6e}")
        print("=" * 100)


class ThermalExercises:
    """Classe que organiza os exercícios pedidos na aba amarela."""

    def __init__(self, base_cfg: ThermalPlateConfig) -> None:
        self.base_cfg = base_cfg

    def clone_cfg(self, **kwargs: Any) -> ThermalPlateConfig:
        """Cria configuração derivada da configuração base."""
        data = self.base_cfg.__dict__.copy()
        data.update(kwargs)
        return ThermalPlateConfig(**data)

    def exercise_1_nominal_cases(self) -> List[Dict[str, Any]]:
        """Exercício 1: caso nominal, malhas refinadas, denso vs esparso."""
        grid_list = [(21, 11), (41, 21), (81, 41), (161, 81), (321, 161)]
        nominal_cfg = self.clone_cfg(
            use_variable_k=False,
            k_constant=0.25,
            source_function=constant_source(5.0e5),
            TL=10.0,
            TR=30.0,
            TB=top_bottom_temperature_function(self.base_cfg.Lx),
            TT=top_bottom_temperature_function(self.base_cfg.Lx),
            use_circle_constraint=False,
        )
        summary = []
        print("\n" + "#" * 100)
        print("EXERCÍCIO 1 - CASO NOMINAL / DENSO vs ESPARSO")
        print("#" * 100)
        for Nx, Ny in grid_list:
            row = {"Nx": Nx, "Ny": Ny}
            for mode in ["dense", "sparse"]:
                cfg = self.clone_cfg(
                    Nx=Nx, Ny=Ny, solver_mode=mode,
                    use_variable_k=nominal_cfg.use_variable_k,
                    k_constant=nominal_cfg.k_constant,
                    source_function=nominal_cfg.source_function,
                    TL=nominal_cfg.TL, TR=nominal_cfg.TR,
                    TB=nominal_cfg.TB, TT=nominal_cfg.TT,
                    use_circle_constraint=nominal_cfg.use_circle_constraint,
                )
                solver = ThermalPlateSolver(cfg)
                result = solver.solve_system()
                row[f"{mode}_assembly_time"] = result["assembly_time"]
                row[f"{mode}_bc_time"] = result["bc_time"]
                row[f"{mode}_solve_time"] = result["solve_time"]
                row[f"{mode}_total_time"] = result["total_time"]
                row[f"{mode}_Tmax"] = result["Tmax"]
                if cfg.make_plots and (Nx, Ny) in [(21, 11), (81, 41), (321, 161)] and mode == "sparse":
                    solver.plot_temperature_contours(result, title=f"Ex.1 - Contours - {Nx}x{Ny} - {mode}")
                    solver.plot_centerline(result, title=f"Ex.1 - Eixo central - {Nx}x{Ny} - {mode}")
            summary.append(row)
        print("=" * 140)
        print("Tabela resumo - Exercício 1")
        print("=" * 140)
        print(" Nx |  Ny | dense_total[s] | sparse_total[s] | dense_solve[s] | sparse_solve[s] | Tmax_dense[°C] | Tmax_sparse[°C]")
        for row in summary:
            print(f"{row['Nx']:3d} | {row['Ny']:3d} | {row['dense_total_time']:.6e} | {row['sparse_total_time']:.6e} | {row['dense_solve_time']:.6e} | {row['sparse_solve_time']:.6e} | {row['dense_Tmax']:.6f} | {row['sparse_Tmax']:.6f}")
        print("=" * 140)
        return summary

    def exercise_2_circle_region(self) -> List[Dict[str, Any]]:
        """Exercício 2: círculo com temperatura fixa."""
        grid_list = [(21, 11), (41, 21), (81, 41), (161, 81), (321, 161)]
        summary = []
        print("\n" + "#" * 100)
        print("EXERCÍCIO 2 - REGIÃO CIRCULAR COM TEMPERATURA FIXA")
        print("#" * 100)
        for Nx, Ny in grid_list:
            cfg = self.clone_cfg(
                Nx=Nx, Ny=Ny, solver_mode="sparse", use_circle_constraint=True, TC=30.0,
                use_variable_k=False, k_constant=0.25,
                source_function=constant_source(5.0e5),
                TL=10.0, TR=30.0,
                TB=top_bottom_temperature_function(self.base_cfg.Lx),
                TT=top_bottom_temperature_function(self.base_cfg.Lx),
            )
            solver = ThermalPlateSolver(cfg)
            result = solver.solve_system()
            summary.append({"Nx": Nx, "Ny": Ny, "Tmax": result["Tmax"], "Tmean": result["Tmean"], "total_time": result["total_time"]})
            if cfg.make_plots and (Nx, Ny) in [(21, 11), (81, 41), (321, 161)]:
                solver.plot_temperature_contours(result, title=f"Ex.2 - Região circular - {Nx}x{Ny}")
                solver.plot_centerline(result, title=f"Ex.2 - Eixo central - {Nx}x{Ny}")
        print("=" * 100)
        print("Tabela resumo - Exercício 2")
        print("=" * 100)
        print(" Nx |  Ny | Tmax [°C] | Tmean [°C] | total_time [s]")
        for row in summary:
            print(f"{row['Nx']:3d} | {row['Ny']:3d} | {row['Tmax']:.6f} | {row['Tmean']:.6f} | {row['total_time']:.6e}")
        print("=" * 100)
        return summary

    def exercise_3_variable_conductivity(self) -> Dict[str, Any]:
        """Exercício 3: condutividade variável."""
        print("\n" + "#" * 100)
        print("EXERCÍCIO 3 - CONDUTIVIDADE VARIÁVEL")
        print("#" * 100)
        cfg = self.clone_cfg(
            solver_mode="sparse", use_variable_k=True,
            k_function=variable_k_function(self.base_cfg.Lx, self.base_cfg.Ly),
            use_circle_constraint=False,
            source_function=constant_source(5.0e5),
            TL=10.0, TR=30.0,
            TB=top_bottom_temperature_function(self.base_cfg.Lx),
            TT=top_bottom_temperature_function(self.base_cfg.Lx),
        )
        solver = ThermalPlateSolver(cfg)
        result = solver.solve_system()
        solver.print_output_summary(result)
        if cfg.make_plots:
            solver.plot_temperature_contours(result, title="Ex.3 - Condutividade variável")
            solver.plot_centerline(result, title="Ex.3 - Eixo central")
            X, Y = np.meshgrid(solver.x, solver.y)
            Kmap = cfg.k_function(X, Y)
            plt.figure(figsize=(10, 4.5))
            contour = plt.contourf(X, Y, Kmap, levels=20)
            plt.colorbar(contour, label="k [W/(m.K)]")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("Ex.3 - Mapa da condutividade variável")
            plt.tight_layout()
            plt.show()
        return result

    def exercise_4_Tmax_Tmean_vs_TC(self) -> Dict[str, np.ndarray]:
        """Exercício 4: Tmax e Tmédia em função de Tc."""
        print("\n" + "#" * 100)
        print("EXERCÍCIO 4 - TMAX E TMÉDIA EM FUNÇÃO DE TC")
        print("#" * 100)
        TC_values = np.linspace(10.0, 60.0, 11)
        Tmax_values = []
        Tmean_values = []
        for TC in TC_values:
            cfg = self.clone_cfg(
                Nx=101, Ny=51, solver_mode="sparse",
                use_variable_k=False, k_constant=0.25,
                use_circle_constraint=True, TC=float(TC),
                source_function=constant_source(5.0e5),
                TL=10.0, TR=30.0,
                TB=top_bottom_temperature_function(self.base_cfg.Lx),
                TT=top_bottom_temperature_function(self.base_cfg.Lx),
            )
            solver = ThermalPlateSolver(cfg)
            result = solver.solve_system()
            Tmax_values.append(result["Tmax"])
            Tmean_values.append(result["Tmean"])
        TC_values = np.array(TC_values)
        Tmax_values = np.array(Tmax_values)
        Tmean_values = np.array(Tmean_values)
        plt.figure(figsize=(9, 4.5))
        plt.plot(TC_values, Tmax_values, marker="o", label="Tmax")
        plt.plot(TC_values, Tmean_values, marker="s", label="Tmédia")
        plt.xlabel("TC [°C]")
        plt.ylabel("Temperatura [°C]")
        plt.title("Ex.4 - Tmax e Tmédia em função de TC")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("=" * 100)
        print("Tabela resumo - Exercício 4")
        print("=" * 100)
        print("TC [°C] | Tmax [°C] | Tmean [°C]")
        for tc, tmax, tmean in zip(TC_values, Tmax_values, Tmean_values):
            print(f"{tc:7.3f} | {tmax:10.6f} | {tmean:11.6f}")
        print("=" * 100)
        return {"TC_values": TC_values, "Tmax_values": Tmax_values, "Tmean_values": Tmean_values}

    def exercise_5_linear_dependence(self, k_index: int = 233) -> Dict[str, Any]:
        """Exercício 5: determina a, b, c em Tk = a*TR + b*TC + c."""
        print("\n" + "#" * 100)
        print("EXERCÍCIO 5 - DETERMINAÇÃO DE a, b, c")
        print("#" * 100)
        cfg_base = self.clone_cfg(
            Nx=101, Ny=51, solver_mode="sparse", use_variable_k=False,
            k_constant=0.25, use_circle_constraint=True,
            source_function=constant_source(5.0e5),
            TL=10.0, TB=top_bottom_temperature_function(self.base_cfg.Lx), TT=top_bottom_temperature_function(self.base_cfg.Lx),
        )
        nunk = cfg_base.Nx * cfg_base.Ny
        if not (0 <= k_index < nunk):
            raise ValueError(f"O índice global k={k_index} não existe para a malha escolhida.")
        cases = [{"TR": 20.0, "TC": 30.0}, {"TR": 35.0, "TC": 25.0}, {"TR": 45.0, "TC": 50.0}]
        y = []
        M = []
        for case in cases:
            cfg = self.clone_cfg(
                Nx=cfg_base.Nx, Ny=cfg_base.Ny, solver_mode=cfg_base.solver_mode,
                use_variable_k=cfg_base.use_variable_k, k_constant=cfg_base.k_constant,
                use_circle_constraint=cfg_base.use_circle_constraint, source_function=cfg_base.source_function,
                TL=cfg_base.TL, TR=case["TR"], TB=cfg_base.TB, TT=cfg_base.TT, TC=case["TC"],
            )
            solver = ThermalPlateSolver(cfg)
            result = solver.solve_system()
            Tk = float(result["T_vec"][k_index])
            y.append(Tk)
            M.append([case["TR"], case["TC"], 1.0])
        M = np.array(M, dtype=float)
        y = np.array(y, dtype=float)
        coeff = np.linalg.solve(M, y)
        a, b, c = coeff
        print("=" * 100)
        print("Sistema usado para determinar os coeficientes")
        print("=" * 100)
        print("M =")
        print(M)
        print("y =")
        print(y)
        print()
        print(f"a = {a:.12e}")
        print(f"b = {b:.12e}")
        print(f"c = {c:.12e}")
        print("=" * 100)
        return {"k_index": k_index, "M": M, "y": y, "a": a, "b": b, "c": c}


def print_user_guide() -> None:
    """Explica como o usuário pode interagir com o programa."""
    print("=" * 100)
    print("GUIA RÁPIDO DE USO")
    print("=" * 100)
    print("1) Para mudar a malha, altere Nx e Ny em USER_CONFIG.")
    print("2) Para mudar o tamanho da placa, altere Lx e Ly.")
    print("3) Para mudar temperaturas das bordas, altere TL, TR, TB, TT.")
    print("4) TL e TR podem ser constantes; TB e TT podem ser funções de x.")
    print("5) Para ligar a região circular, coloque use_circle_constraint=True.")
    print("6) Para mudar Tc, altere TC.")
    print("7) Para usar condutividade variável, coloque use_variable_k=True.")
    print("8) Para resolver como matriz densa, use solver_mode='dense'.")
    print("9) Para resolver como matriz esparsa, use solver_mode='sparse'.")
    print("10) Para rodar os exercícios, altere as flags em USER_CONFIG.")
    print("=" * 100)


USER_CONFIG = {
    "single_run": True,
    "solver_mode": "sparse",
    "Lx": 0.02,
    "Ly": 0.01,
    "Nx": 101,
    "Ny": 51,
    "TL": 10.0,
    "TR": 30.0,
    "use_nominal_top_bottom_function": True,
    "source_value": 5.0e5,
    "use_variable_k": False,
    "k_constant": 0.25,
    "use_circle_constraint": False,
    "TC": 30.0,
    "circle_center_x": 0.75 * 0.02,
    "circle_center_y": 0.50 * 0.01,
    "circle_radius": 0.002,
    "make_plots": True,
    "print_matrix_info": True,
    "contour_levels": 20,
    "preserve_symmetry": True,
    "run_exercise_1": True,
    "run_exercise_2": True,
    "run_exercise_3": True,
    "run_exercise_4": True,
    "run_exercise_5": True,
    "exercise_5_k_index": 233,
}


def build_base_config_from_user() -> ThermalPlateConfig:
    """Monta a configuração principal a partir de USER_CONFIG."""
    Lx = float(USER_CONFIG["Lx"])
    if USER_CONFIG["use_nominal_top_bottom_function"]:
        TB = top_bottom_temperature_function(Lx)
        TT = top_bottom_temperature_function(Lx)
    else:
        TB = 10.0
        TT = 10.0
    cfg = ThermalPlateConfig(
        Lx=Lx,
        Ly=float(USER_CONFIG["Ly"]),
        Nx=int(USER_CONFIG["Nx"]),
        Ny=int(USER_CONFIG["Ny"]),
        TL=float(USER_CONFIG["TL"]),
        TR=float(USER_CONFIG["TR"]),
        TB=TB,
        TT=TT,
        source_function=constant_source(float(USER_CONFIG["source_value"])),
        use_variable_k=bool(USER_CONFIG["use_variable_k"]),
        k_constant=float(USER_CONFIG["k_constant"]),
        k_function=variable_k_function(Lx, float(USER_CONFIG["Ly"])) if USER_CONFIG["use_variable_k"] else None,
        use_circle_constraint=bool(USER_CONFIG["use_circle_constraint"]),
        circle_center_x=float(USER_CONFIG["circle_center_x"]),
        circle_center_y=float(USER_CONFIG["circle_center_y"]),
        circle_radius=float(USER_CONFIG["circle_radius"]),
        TC=float(USER_CONFIG["TC"]),
        solver_mode=str(USER_CONFIG["solver_mode"]),
        preserve_symmetry=bool(USER_CONFIG["preserve_symmetry"]),
        make_plots=bool(USER_CONFIG["make_plots"]),
        print_matrix_info=bool(USER_CONFIG["print_matrix_info"]),
        contour_levels=int(USER_CONFIG["contour_levels"]),
    )
    return cfg


def main() -> None:
    """Função principal."""
    print_user_guide()
    base_cfg = build_base_config_from_user()

    if USER_CONFIG["single_run"]:
        print("\n" + "#" * 100)
        print("CASO ÚNICO - RESOLUÇÃO DA PLACA TÉRMICA")
        print("#" * 100)
        solver = ThermalPlateSolver(base_cfg)
        solver.print_inputs_summary()
        result = solver.solve_system()
        solver.print_output_summary(result)
        solver.print_matrix_info(result)
        solver.print_final_explanation(result)
        if base_cfg.make_plots:
            solver.plot_temperature_contours(result, title="Caso único - Contours")
            solver.plot_centerline(result, title="Caso único - Eixo central")

    exercises = ThermalExercises(base_cfg)
    if USER_CONFIG["run_exercise_1"]:
        exercises.exercise_1_nominal_cases()
    if USER_CONFIG["run_exercise_2"]:
        exercises.exercise_2_circle_region()
    if USER_CONFIG["run_exercise_3"]:
        exercises.exercise_3_variable_conductivity()
    if USER_CONFIG["run_exercise_4"]:
        exercises.exercise_4_Tmax_Tmean_vs_TC()
    if USER_CONFIG["run_exercise_5"]:
        exercises.exercise_5_linear_dependence(k_index=int(USER_CONFIG["exercise_5_k_index"]))


if __name__ == "__main__":
    main()
