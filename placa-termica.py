# -*- coding: utf-8 -*-
"""
placa_termica_otimizada.py
============================================================
PARTE TÉRMICA DO GÊMEO DIGITAL - VERSÃO OTIMIZADA
============================================================

Objetivos desta versão:
- manter a parte térmica separada da hidráulica
- preservar nomes semânticos importantes como assembly, solve_system,
  print_inputs_summary, print_output_summary e print_final_explanation
- reforçar a impressão organizada dos exercícios pedidos na apostila
- ampliar a base teórica de álgebra linear dentro do próprio código,
  com comentários explicando o porquê das escolhas numéricas
- implementar também os exercícios iterativos (Jacobi, Gauss-Seidel,
  animação e ajuste iterativo de Tc para uma Tmax desejada)

Dependências no Windows:
    pip install numpy matplotlib scipy pillow

Como rodar:
    python placa_termica_otimizada.py

Observação:
Esta versão privilegia clareza, rastreabilidade e visualização dos resultados,
sem integrar ainda com a parte hidráulica.
"""

from __future__ import annotations

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------
# time: medir custo computacional de montagem, solução e iterações.
import time

# dataclass: simplifica a criação de classes de configuração.
from dataclasses import dataclass, field

# Tipagem: deixa o código mais legível e facilita manutenção futura.
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

# numpy: base numérica do código (vetores, matrizes, álgebra linear básica).
import numpy as np

# matplotlib: geração de gráficos, contours, perfis e animações.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# scipy.sparse: matrizes esparsas, fundamentais para discretizações 2D.
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ============================================================
# TIPOS AUXILIARES
# ============================================================

# Tipo numérico simples.
Number = Union[int, float]

# Pode ser constante ou função escalar de uma variável.
ScalarOrCallable = Union[Number, Callable[[float], float]]

# Campo escalar 2D: recebe (x,y) e retorna valor.
FieldFunction2D = Callable[[float, float], float]


# ============================================================
# FUNÇÕES DE MAPEAMENTO DE ÍNDICES
# ============================================================


def ij2n(i: int, j: int, Nx: int) -> int:
    """
    Mapeia o par de índices (i, j) para um índice global n.

    Ideia de álgebra linear:
    -----------------------
    Para transformar a malha 2D em um sistema linear A*T=b, precisamos
    reordenar todas as temperaturas T[i,j] em um único vetor T[n].
    O mapeamento adotado é:
        n = i + j*Nx
    """
    return i + j * Nx



def n2ij(n: int, Nx: int) -> Tuple[int, int]:
    """
    Faz a operação inversa de ij2n.
    Dado um índice global n, devolve os índices de malha (i, j).
    """
    j = n // Nx
    i = n % Nx
    return i, j


# ============================================================
# FUNÇÕES DE CAMPO / CONTORNO
# ============================================================


def to_callable(value: ScalarOrCallable) -> Callable[[float], float]:
    """
    Converte valor constante ou função em função callable.

    Isso permite que TL, TR, TB, TT aceitem:
    - um número fixo
    - uma função dependente da posição
    """
    if callable(value):
        return value
    return lambda x: float(value)



def constant_source(value: float) -> FieldFunction2D:
    """Retorna f(x,y)=constante."""
    return lambda x, y: float(value)



def variable_k_function(Lx: float, Ly: float) -> FieldFunction2D:
    """
    Condutividade variável do exercício 3:
        k(x, y) = 0.2 + 0.05 sin(3πx/Lx) sin(3πy/Ly)
    """
    def _k(x: float, y: float) -> float:
        return 0.2 + 0.05 * np.sin(3.0 * np.pi * x / Lx) * np.sin(3.0 * np.pi * y / Ly)
    return _k



def top_bottom_temperature_function(Lx: float) -> Callable[[float], float]:
    """
    Temperatura nas bordas superior e inferior do caso nominal:
        T(x) = 10 + 20*x/Lx
    """
    return lambda x: 10.0 + 20.0 * x / Lx


# ============================================================
# CONFIGURAÇÃO DO PROBLEMA
# ============================================================

@dataclass
class ThermalPlateConfig:
    """
    Classe de configuração da placa térmica.

    O objetivo é concentrar todas as entradas físicas, geométricas e
    numéricas em um único lugar, facilitando o uso e a integração futura.
    """

    # Dimensões da placa.
    Lx: float = 0.02
    Ly: float = 0.01

    # Número de pontos na malha.
    Nx: int = 101
    Ny: int = 51

    # Temperaturas prescritas nas bordas.
    TL: ScalarOrCallable = 10.0
    TR: ScalarOrCallable = 30.0
    TB: ScalarOrCallable = field(default_factory=lambda: top_bottom_temperature_function(0.02))
    TT: ScalarOrCallable = field(default_factory=lambda: top_bottom_temperature_function(0.02))

    # Fonte térmica interna f(x,y).
    source_function: FieldFunction2D = field(default_factory=lambda: constant_source(5.0e5))

    # Condutividade térmica.
    use_variable_k: bool = False
    k_constant: float = 0.25
    k_function: Optional[FieldFunction2D] = None

    # Região circular de temperatura prescrita.
    use_circle_constraint: bool = False
    circle_center_x: float = 0.75 * 0.02
    circle_center_y: float = 0.50 * 0.01
    circle_radius: float = 0.002
    TC: float = 30.0

    # Modo do solver direto.
    solver_mode: str = "sparse"  # "dense" ou "sparse"

    # Se True, impõe Dirichlet preservando a simetria do sistema.
    preserve_symmetry: bool = True

    # Opções de visualização.
    make_plots: bool = True
    save_figures: bool = False
    output_dir: str = "."
    contour_levels: int = 20

    # Opções de impressão.
    print_matrix_info: bool = True
    print_full_tables: bool = True

    def __post_init__(self) -> None:
        """
        Ajusta dados após a inicialização.

        Esta etapa garante que:
        - TL, TR, TB, TT sejam callables
        - k_function exista caso use_variable_k=True
        """
        if not callable(self.TL):
            self.TL = to_callable(self.TL)
        if not callable(self.TR):
            self.TR = to_callable(self.TR)
        if not callable(self.TB):
            self.TB = to_callable(self.TB)
        if not callable(self.TT):
            self.TT = to_callable(self.TT)
        if self.k_function is None and self.use_variable_k:
            self.k_function = variable_k_function(self.Lx, self.Ly)


# ============================================================
# CLASSE BASE DE SISTEMA LINEAR
# ============================================================

class BaseLinearSystemSolver:
    """
    Classe base para problemas físicos discretizados em forma linear.

    Fundamento matemático:
    ----------------------
    A discretização de muitos problemas físicos leva a um sistema:
        A x = b
    Nesta classe base concentramos os métodos de solução direta.
    """

    def __init__(self) -> None:
        self.last_result: Optional[Dict[str, Any]] = None

    def solve_linear_system_dense(self, A_mod: np.ndarray, b_mod: np.ndarray) -> np.ndarray:
        """
        Resolve sistema denso usando fatoração interna do NumPy.

        Importância em álgebra linear:
        -----------------------------
        Resolver A x = b por fatoração é melhor do que calcular A^{-1} b.
        Evita custo desnecessário e melhora estabilidade numérica.
        """
        return np.linalg.solve(A_mod, b_mod)

    def solve_linear_system_sparse(self, A_mod_sparse: sparse.csr_matrix, b_mod: np.ndarray) -> np.ndarray:
        """
        Resolve sistema esparso com spsolve.

        Relevância:
        ----------
        Em placas 2D, A é tipicamente muito esparsa. Usar CSR reduz memória
        e acelera a solução em comparação com a forma densa, sobretudo em malhas finas.
        """
        return spsolve(A_mod_sparse, b_mod)


# ============================================================
# SOLVER DA PLACA TÉRMICA
# ============================================================

class ThermalPlateSolver(BaseLinearSystemSolver):
    """
    Solver da placa térmica estacionária.

    Este solver implementa:
    - montagem da matriz do problema
    - aplicação de condições de contorno de Dirichlet
    - solução direta (densa ou esparsa)
    - solução iterativa (Jacobi e Gauss-Seidel)
    - gráficos e relatórios detalhados
    """

    def __init__(self, cfg: ThermalPlateConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Vetores de coordenadas da malha.
        self.x = np.linspace(0.0, self.cfg.Lx, self.cfg.Nx)
        self.y = np.linspace(0.0, self.cfg.Ly, self.cfg.Ny)

        # Passos espaciais.
        self.hx = self.cfg.Lx / (self.cfg.Nx - 1)
        self.hy = self.cfg.Ly / (self.cfg.Ny - 1)

        # Número total de incógnitas.
        self.nunk = self.cfg.Nx * self.cfg.Ny

        # Máscaras e vetores de Dirichlet pré-computados.
        self.dirichlet_ids, self.dirichlet_values = self._build_dirichlet_data()
        self.dirichlet_set = set(self.dirichlet_ids.tolist())

    # --------------------------------------------------------
    # Pré-processamento de Dirichlet
    # --------------------------------------------------------

    def _build_dirichlet_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrói os índices e valores de todos os pontos com temperatura prescrita.

        Otimização:
        ----------
        Em vez de recalcular a informação de contorno repetidamente durante
        assembly e build_rhs, armazenamos isso uma única vez.
        """
        ids: List[int] = []
        vals: List[float] = []

        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                value = self.get_dirichlet_value_raw(i, j)
                if value is not None:
                    ids.append(ij2n(i, j, self.cfg.Nx))
                    vals.append(float(value))

        return np.array(ids, dtype=int), np.array(vals, dtype=float)

    def get_k(self, x: float, y: float) -> float:
        """Retorna k(x,y) constante ou variável."""
        if self.cfg.use_variable_k:
            if self.cfg.k_function is None:
                raise ValueError("use_variable_k=True, mas k_function não foi definida.")
            return float(self.cfg.k_function(x, y))
        return float(self.cfg.k_constant)

    def get_source(self, x: float, y: float) -> float:
        """Retorna f(x,y)."""
        return float(self.cfg.source_function(x, y))

    def is_inside_circle_constraint(self, x: float, y: float) -> bool:
        """Retorna True se o ponto está no círculo com temperatura fixa."""
        if not self.cfg.use_circle_constraint:
            return False
        dx = x - self.cfg.circle_center_x
        dy = y - self.cfg.circle_center_y
        return dx * dx + dy * dy <= self.cfg.circle_radius ** 2

    def get_dirichlet_value_raw(self, i: int, j: int) -> Optional[float]:
        """
        Retorna valor de temperatura prescrita no nó (i,j), se houver.

        Ordem de prioridade:
        - bordas
        - círculo interno
        """
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

    def get_dirichlet_value(self, i: int, j: int) -> Optional[float]:
        """Interface pública para valor de Dirichlet."""
        return self.get_dirichlet_value_raw(i, j)

    # --------------------------------------------------------
    # Montagem do sistema
    # --------------------------------------------------------

    def assembly(self, matrix_mode: str = "dense") -> Union[np.ndarray, sparse.csr_matrix]:
        """
        Monta a matriz A do problema térmico.

        Conceito de álgebra linear:
        ---------------------------
        Cada linha de A representa uma equação associada a um nó.
        Para pontos internos, a equação liga o nó aos 4 vizinhos.
        Isso gera uma estrutura quase pentadiagonal em ordenação natural.

        Otimização:
        ----------
        Para a forma esparsa, montamos via listas COO (rows, cols, data),
        que é mais eficiente do que preencher elemento a elemento em matrizes grandes.
        """
        if matrix_mode not in ("dense", "sparse"):
            raise ValueError("matrix_mode deve ser 'dense' ou 'sparse'.")

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)

                # Pontos de Dirichlet não recebem a equação interna aqui;
                # serão tratados depois na aplicação das condições de contorno.
                if Ic in self.dirichlet_set:
                    continue

                Ie = ij2n(i + 1, j, self.cfg.Nx)
                Iw = ij2n(i - 1, j, self.cfg.Nx)
                In = ij2n(i, j + 1, self.cfg.Nx)
                Is = ij2n(i, j - 1, self.cfg.Nx)

                xc = self.x[i]
                yc = self.y[j]

                # Avaliação de k nas faces, como pedido no exercício 3.
                ke = self.get_k(xc + 0.5 * self.hx, yc)
                kw = self.get_k(xc - 0.5 * self.hx, yc)
                kn = self.get_k(xc, yc + 0.5 * self.hy)
                ks = self.get_k(xc, yc - 0.5 * self.hy)

                # Coeficientes locais.
                # Em formulação conservativa simples, com hx≈hy,
                # aP = aE+aW+aN+aS e as vizinhanças entram com sinal negativo.
                aE = ke
                aW = kw
                aN = kn
                aS = ks
                aP = aE + aW + aN + aS

                rows.extend([Ic, Ic, Ic, Ic, Ic])
                cols.extend([Ic, Ie, Iw, In, Is])
                data.extend([aP, -aE, -aW, -aN, -aS])

        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(self.nunk, self.nunk)).tocsr()
        if matrix_mode == "sparse":
            return A_sparse
        return A_sparse.toarray()

    def build_rhs(self) -> np.ndarray:
        """
        Monta o vetor b com o termo fonte.

        Conceito:
        ---------
        O lado direito b representa, neste caso, o termo gerador de calor.
        Cada entrada corresponde ao aporte térmico no nó associado.
        """
        b = np.zeros(self.nunk, dtype=float)
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)
                if Ic in self.dirichlet_set:
                    b[Ic] = 0.0
                else:
                    b[Ic] = self.hx * self.hy * self.get_source(self.x[i], self.y[j])
        return b

    # --------------------------------------------------------
    # Aplicação de Dirichlet
    # --------------------------------------------------------

    def apply_dirichlet_bc_dense(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica Dirichlet em matriz densa.

        preserve_symmetry=True:
            move a contribuição da coluna para o lado direito,
            zera linha e coluna e coloca 1 na diagonal.

        preserve_symmetry=False:
            troca apenas a linha. É simples, mas quebra a simetria.
        """
        A_mod = A.copy()
        b_mod = b.copy()

        for Ic, Tpresc in zip(self.dirichlet_ids, self.dirichlet_values):
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

    def apply_dirichlet_bc_sparse(self, A: sparse.csr_matrix, b: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Aplica Dirichlet em matriz esparsa.

        Estratégia:
        -----------
        Convertimos para LIL para facilitar alterações estruturais e, ao final,
        retornamos em CSR, que é melhor para operações algébricas e solução.
        """
        A_mod = A.copy().tolil()
        b_mod = b.copy()

        for Ic, Tpresc in zip(self.dirichlet_ids, self.dirichlet_values):
            if self.cfg.preserve_symmetry:
                col = A_mod[:, Ic].toarray().ravel()
                b_mod -= col * Tpresc
                A_mod[:, Ic] = 0.0
                A_mod[Ic, :] = 0.0
                A_mod[Ic, Ic] = 1.0
                b_mod[Ic] = Tpresc
            else:
                A_mod[Ic, :] = 0.0
                A_mod[Ic, Ic] = 1.0
                b_mod[Ic] = Tpresc

        return A_mod.tocsr(), b_mod

    # --------------------------------------------------------
    # Solução direta
    # --------------------------------------------------------

    def solve_system(self, matrix_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve o sistema térmico por método direto.

        Mede tempos separadamente:
        - assembly_time: montagem da matriz e rhs
        - bc_time: aplicação das condições de contorno
        - solve_time: solução do sistema linear
        - total_time: tudo acima junto
        """
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

        return self._build_result_dict(A, A_mod, b, b_mod, T_vec, t1 - t0, t2 - t1, t3 - t2)

    def _build_result_dict(
        self,
        A: Union[np.ndarray, sparse.spmatrix],
        A_mod: Union[np.ndarray, sparse.spmatrix],
        b: np.ndarray,
        b_mod: np.ndarray,
        T_vec: np.ndarray,
        assembly_time: float,
        bc_time: float,
        solve_time: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Organiza a solução em um dicionário padronizado.

        Isso facilita impressões, gráficos e integração futura.
        """
        T_grid = T_vec.reshape((self.cfg.Ny, self.cfg.Nx))
        Tmax = float(np.max(T_grid))
        Tmean = float(np.mean(T_grid))
        center_j = self.cfg.Ny // 2
        centerline_T = T_grid[center_j, :].copy()

        result = {
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
            "centerline_T": centerline_T,
            "assembly_time": assembly_time,
            "bc_time": bc_time,
            "solve_time": solve_time,
            "total_time": assembly_time + bc_time + solve_time,
            "matrix_mode": self.cfg.solver_mode,
        }
        if extra:
            result.update(extra)
        self.last_result = result
        return result

    # --------------------------------------------------------
    # Solução iterativa
    # --------------------------------------------------------

    def solve_iterative(
        self,
        method: str = "jacobi",
        tol: float = 1e-6,
        maxit: int = 5000,
        x0: Optional[np.ndarray] = None,
        store_history: bool = False,
        frame_stride: int = 20,
    ) -> Dict[str, Any]:
        """
        Resolve o sistema térmico por método iterativo.

        Métodos disponíveis:
        - jacobi
        - gauss-seidel

        Fundamento de álgebra linear:
        -----------------------------
        Para A x = b, métodos iterativos constroem aproximações sucessivas x^(k)
        sem fatorar completamente A. Isso pode ser vantajoso em sistemas muito grandes,
        embora a velocidade de convergência dependa da estrutura da matriz.
        """
        method = method.lower()
        if method not in ("jacobi", "gauss-seidel", "gauss_seidel"):
            raise ValueError("method deve ser 'jacobi' ou 'gauss-seidel'.")

        # Para os métodos iterativos, usamos a forma densa para simplificar a escrita pedagógica.
        # Em malhas muito grandes isso não é ideal, por isso os exercícios iterativos foram pensados
        # mais como estudo de método do que como produção máxima de desempenho.
        t0 = time.perf_counter()
        A = self.assembly(matrix_mode="dense")
        b = self.build_rhs()
        A_mod, b_mod = self.apply_dirichlet_bc_dense(A, b)
        t1 = time.perf_counter()

        n = A_mod.shape[0]

        if x0 is None:
            x = np.zeros(n, dtype=float)
        else:
            x = np.array(x0, dtype=float).copy()

        D = np.diag(np.diag(A_mod))
        R = A_mod - D

        # Para Gauss-Seidel, separamos parte inferior+diagonal e parte superior.
        LplusD = np.tril(A_mod)
        U = A_mod - LplusD

        residual_history: List[float] = []
        frame_history: List[np.ndarray] = []

        t2 = time.perf_counter()
        for k in range(maxit):
            if method == "jacobi":
                # Fórmula do Jacobi:
                #   x^(k+1) = D^{-1}(b - R x^(k))
                x_new = (b_mod - R @ x) / np.diag(A_mod)
            else:
                # Fórmula do Gauss-Seidel:
                #   (L + D)x^(k+1) = b - U x^(k)
                rhs = b_mod - U @ x
                x_new = np.linalg.solve(LplusD, rhs)

            residual = np.linalg.norm(A_mod @ x_new - b_mod)
            residual_history.append(float(residual))

            if store_history and (k % frame_stride == 0):
                frame_history.append(x_new.reshape((self.cfg.Ny, self.cfg.Nx)).copy())

            if residual < tol:
                x = x_new
                break

            x = x_new
        t3 = time.perf_counter()

        extra = {
            "iterative_method": method,
            "iterations": k + 1,
            "residual_history": np.array(residual_history, dtype=float),
            "frame_history": frame_history,
            "tol": tol,
            "maxit": maxit,
        }

        return self._build_result_dict(A, A_mod, b, b_mod, x, t1 - t0, 0.0, t3 - t2, extra=extra)

    # --------------------------------------------------------
    # Impressões
    # --------------------------------------------------------

    def print_inputs_summary(self) -> None:
        """Imprime entradas do problema de forma detalhada."""
        print("=" * 110)
        print("ENTRADAS UTILIZADAS NA PLACA TÉRMICA")
        print("=" * 110)
        print(f"Lx [m]                              : {self.cfg.Lx}")
        print(f"Ly [m]                              : {self.cfg.Ly}")
        print(f"Nx                                  : {self.cfg.Nx}")
        print(f"Ny                                  : {self.cfg.Ny}")
        print(f"Número total de incógnitas          : {self.nunk}")
        print(f"hx [m]                              : {self.hx:.6e}")
        print(f"hy [m]                              : {self.hy:.6e}")
        print(f"solver_mode                         : {self.cfg.solver_mode}")
        print(f"preserve_symmetry                   : {self.cfg.preserve_symmetry}")
        print(f"use_variable_k                      : {self.cfg.use_variable_k}")
        print(f"k_constant [W/(m.K)]                : {self.cfg.k_constant}")
        print(f"use_circle_constraint               : {self.cfg.use_circle_constraint}")
        print(f"TC [°C]                             : {self.cfg.TC}")
        print(f"circle_center_x [m]                 : {self.cfg.circle_center_x}")
        print(f"circle_center_y [m]                 : {self.cfg.circle_center_y}")
        print(f"circle_radius [m]                   : {self.cfg.circle_radius}")
        print(f"nós de Dirichlet totais             : {len(self.dirichlet_ids)}")
        print("=" * 110)

    def print_output_summary(self, result: Dict[str, Any]) -> None:
        """Imprime as saídas principais do solver."""
        print("=" * 110)
        print("RESULTADOS DA PLACA TÉRMICA")
        print("=" * 110)
        if "iterative_method" in result:
            print(f"Método                             : iterativo ({result['iterative_method']})")
            print(f"Iterações                          : {result['iterations']}")
            print(f"Tolerância                         : {result['tol']:.3e}")
        else:
            print(f"Método                             : direto ({result['matrix_mode']})")
        print(f"Tempo de montagem [s]              : {result['assembly_time']:.6e}")
        print(f"Tempo de BC [s]                    : {result['bc_time']:.6e}")
        print(f"Tempo de solução [s]               : {result['solve_time']:.6e}")
        print(f"Tempo total [s]                    : {result['total_time']:.6e}")
        print(f"Temperatura máxima [°C]            : {result['Tmax']:.6f}")
        print(f"Temperatura média [°C]             : {result['Tmean']:.6f}")
        print("=" * 110)

    def print_matrix_info(self, result: Dict[str, Any]) -> None:
        """Imprime informações estruturais da matriz do sistema."""
        if not self.cfg.print_matrix_info:
            return
        A_mod = result["A_mod"]
        print("=" * 110)
        print("INFORMAÇÕES MATRICIAIS")
        print("=" * 110)
        print(f"Número de incógnitas               : {self.nunk}")
        if sparse.issparse(A_mod):
            nnz = A_mod.nnz
            density = nnz / (self.nunk * self.nunk)
            print("Tipo de matriz                     : esparsa")
            print(f"Número de não nulos                : {nnz}")
            print(f"Densidade da matriz                : {density:.6e}")
        else:
            nnz = np.count_nonzero(A_mod)
            density = nnz / (self.nunk * self.nunk)
            print("Tipo de matriz                     : densa")
            print(f"Número de não nulos                : {nnz}")
            print(f"Densidade da matriz                : {density:.6e}")
        print("=" * 110)

    def print_final_explanation(self, result: Dict[str, Any]) -> None:
        """Explica ao usuário o que o programa fez e por que isso importa."""
        print("=" * 110)
        print("COMENTÁRIO FINAL AO USUÁRIO")
        print("=" * 110)
        print("O código acabou de:")
        print("1. Discretizar a placa térmica em uma malha 2D.")
        print("2. Traduzir o problema contínuo em um sistema linear A*T = b.")
        print("3. Incorporar as temperaturas prescritas como condições de Dirichlet.")
        print("4. Resolver o sistema com álgebra linear direta ou iterativa.")
        print("5. Reconstruir o campo T(x,y) e extrair Tmax, Tmédia e o perfil central.")
        print()
        print("Conceitos de álgebra linear usados na otimização:")
        print("- Estrutura esparsa: economiza memória e acelera a solução de malhas grandes.")
        print("- Preservação da simetria: importante para estabilidade e para métodos apropriados.")
        print("- Separação de montagem, BC e solução: ajuda a medir custo computacional com precisão.")
        print("- Métodos iterativos: evitam fatoração completa e são úteis para sistemas grandes.")
        print()
        print(f"Tmax obtida = {result['Tmax']:.6f} °C")
        print(f"Tmédia obtida = {result['Tmean']:.6f} °C")
        print("=" * 110)

    # --------------------------------------------------------
    # Gráficos
    # --------------------------------------------------------

    def _maybe_save(self, fig: plt.Figure, name: str) -> None:
        """Salva figura se save_figures=True."""
        if self.cfg.save_figures:
            fig.savefig(f"{self.cfg.output_dir}/{name}", dpi=200, bbox_inches="tight")

    def plot_temperature_contours(self, result: Dict[str, Any], title: str = "", file_name: str = "contours.png") -> None:
        """Plota curvas de nível e mapa preenchido de temperatura."""
        X, Y = np.meshgrid(result["x"], result["y"])
        T_grid = result["T_grid"]
        fig = plt.figure(figsize=(10, 4.8))
        contour = plt.contourf(X, Y, T_grid, levels=self.cfg.contour_levels)
        plt.colorbar(contour, label="Temperatura [°C]")
        plt.contour(X, Y, T_grid, levels=self.cfg.contour_levels, colors="k", linewidths=0.3)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(title if title else "Curvas de nível da temperatura")
        plt.tight_layout()
        self._maybe_save(fig, file_name)
        plt.show()

    def plot_centerline(self, result: Dict[str, Any], title: str = "", file_name: str = "centerline.png") -> None:
        """Plota temperatura ao longo do eixo central da placa."""
        fig = plt.figure(figsize=(9, 4))
        plt.plot(result["centerline_x"], result["centerline_T"], marker="o", markersize=3, linewidth=1.2)
        plt.xlabel("x [m]")
        plt.ylabel("Temperatura [°C]")
        plt.title(title if title else "Temperatura ao longo do eixo central")
        plt.grid(True)
        plt.tight_layout()
        self._maybe_save(fig, file_name)
        plt.show()

    def plot_residual_history(self, result: Dict[str, Any], title: str = "", file_name: str = "residual.png") -> None:
        """Plota histórico do residual para métodos iterativos."""
        if "residual_history" not in result:
            return
        fig = plt.figure(figsize=(8, 4))
        plt.semilogy(result["residual_history"], linewidth=1.5)
        plt.xlabel("Iteração")
        plt.ylabel("||A x - b||")
        plt.title(title if title else "Convergência iterativa")
        plt.grid(True)
        plt.tight_layout()
        self._maybe_save(fig, file_name)
        plt.show()

    def save_iteration_animation(self, result: Dict[str, Any], file_name: str = "animacao_iteracoes.gif", interval_ms: int = 250) -> None:
        """
        Salva uma animação da evolução do campo de temperatura.

        Requer que solve_iterative tenha sido chamado com store_history=True.
        """
        frames = result.get("frame_history", [])
        if not frames:
            print("Nenhum frame disponível para animação. Use store_history=True.")
            return

        X, Y = np.meshgrid(self.x, self.y)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        vmin = min(float(np.min(f)) for f in frames)
        vmax = max(float(np.max(f)) for f in frames)
        contour = ax.contourf(X, Y, frames[0], levels=self.cfg.contour_levels, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contour, ax=ax, label="Temperatura [°C]")

        def update(frame_idx: int):
            ax.clear()
            cf = ax.contourf(X, Y, frames[frame_idx], levels=self.cfg.contour_levels, vmin=vmin, vmax=vmax)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(f"Evolução iterativa - frame {frame_idx}")
            return cf.collections

        anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)
        try:
            anim.save(f"{self.cfg.output_dir}/{file_name}", writer="pillow")
            print(f"Animação salva em {self.cfg.output_dir}/{file_name}")
        except Exception as exc:
            print("Não foi possível salvar a animação. Instale pillow se necessário.")
            print(f"Erro: {exc}")
        plt.close(fig)


# ============================================================
# EXERCÍCIOS
# ============================================================

class ThermalExercises:
    """Classe que organiza os exercícios pedidos nas imagens e na apostila."""

    def __init__(self, base_cfg: ThermalPlateConfig) -> None:
        self.base_cfg = base_cfg

    def clone_cfg(self, **kwargs: Any) -> ThermalPlateConfig:
        """Cria uma cópia configurável da configuração base."""
        data = self.base_cfg.__dict__.copy()
        data.update(kwargs)
        return ThermalPlateConfig(**data)

    def exercise_1_nominal_cases(self) -> List[Dict[str, Any]]:
        """
        Exercício 1:
        - malhas (21,11), (41,21), (81,41), (161,81), (321,161)
        - comparar tempos densos e esparsos
        - plotar contours
        - reportar Tmax
        - plotar eixo central
        """
        grid_list = [(21, 11), (41, 21), (81, 41), (161, 81), (321, 161)]
        summary: List[Dict[str, Any]] = []

        print("\n" + "#" * 120)
        print("EXERCÍCIO 1 - CASO NOMINAL, REFINAMENTO DE MALHA, MATRIZ DENSA vs ESPARSA")
        print("#" * 120)

        for Nx, Ny in grid_list:
            row: Dict[str, Any] = {"Nx": Nx, "Ny": Ny}
            for mode in ["dense", "sparse"]:
                cfg = self.clone_cfg(
                    Nx=Nx,
                    Ny=Ny,
                    solver_mode=mode,
                    use_variable_k=False,
                    k_constant=0.25,
                    source_function=constant_source(5.0e5),
                    TL=10.0,
                    TR=30.0,
                    TB=top_bottom_temperature_function(self.base_cfg.Lx),
                    TT=top_bottom_temperature_function(self.base_cfg.Lx),
                    use_circle_constraint=False,
                )
                solver = ThermalPlateSolver(cfg)
                result = solver.solve_system()
                row[f"{mode}_assembly_time"] = result["assembly_time"]
                row[f"{mode}_bc_time"] = result["bc_time"]
                row[f"{mode}_solve_time"] = result["solve_time"]
                row[f"{mode}_total_time"] = result["total_time"]
                row[f"{mode}_Tmax"] = result["Tmax"]
                row[f"{mode}_Tmean"] = result["Tmean"]

                if cfg.make_plots and mode == "sparse":
                    solver.plot_temperature_contours(
                        result,
                        title=f"Ex.1 - Contours - malha {Nx}x{Ny} - {mode}",
                        file_name=f"ex1_contours_{Nx}x{Ny}_{mode}.png",
                    )
                    solver.plot_centerline(
                        result,
                        title=f"Ex.1 - Eixo central - malha {Nx}x{Ny} - {mode}",
                        file_name=f"ex1_centerline_{Nx}x{Ny}_{mode}.png",
                    )

            summary.append(row)

        print("=" * 160)
        print("TABELA RESUMO - EXERCÍCIO 1")
        print("=" * 160)
        print(" Nx |  Ny | dense_total[s] | sparse_total[s] | dense_solve[s] | sparse_solve[s] | Tmax_dense | Tmax_sparse | Tmean_dense | Tmean_sparse")
        for row in summary:
            print(
                f"{row['Nx']:3d} | {row['Ny']:3d} | {row['dense_total_time']:.6e} | {row['sparse_total_time']:.6e} | "
                f"{row['dense_solve_time']:.6e} | {row['sparse_solve_time']:.6e} | {row['dense_Tmax']:.6f} | {row['sparse_Tmax']:.6f} | "
                f"{row['dense_Tmean']:.6f} | {row['sparse_Tmean']:.6f}"
            )
        print("=" * 160)
        return summary

    def exercise_2_circle_region(self) -> List[Dict[str, Any]]:
        """
        Exercício 2:
        - introduz região circular com temperatura fixa Tc=30°C
        - usa vários refinamentos de malha
        - reporta Tmax, Tmean e perfil no eixo central
        """
        grid_list = [(21, 11), (41, 21), (81, 41), (161, 81), (321, 161)]
        summary: List[Dict[str, Any]] = []

        print("\n" + "#" * 120)
        print("EXERCÍCIO 2 - REGIÃO CIRCULAR COM TEMPERATURA FIXA")
        print("#" * 120)

        for Nx, Ny in grid_list:
            cfg = self.clone_cfg(
                Nx=Nx,
                Ny=Ny,
                solver_mode="sparse",
                use_circle_constraint=True,
                TC=30.0,
                use_variable_k=False,
                k_constant=0.25,
                source_function=constant_source(5.0e5),
                TL=10.0,
                TR=30.0,
                TB=top_bottom_temperature_function(self.base_cfg.Lx),
                TT=top_bottom_temperature_function(self.base_cfg.Lx),
            )
            solver = ThermalPlateSolver(cfg)
            result = solver.solve_system()
            summary.append({
                "Nx": Nx,
                "Ny": Ny,
                "Tmax": result["Tmax"],
                "Tmean": result["Tmean"],
                "total_time": result["total_time"],
            })

            if cfg.make_plots:
                solver.plot_temperature_contours(
                    result,
                    title=f"Ex.2 - Círculo interno - malha {Nx}x{Ny}",
                    file_name=f"ex2_contours_{Nx}x{Ny}.png",
                )
                solver.plot_centerline(
                    result,
                    title=f"Ex.2 - Eixo central - malha {Nx}x{Ny}",
                    file_name=f"ex2_centerline_{Nx}x{Ny}.png",
                )

        print("=" * 120)
        print("TABELA RESUMO - EXERCÍCIO 2")
        print("=" * 120)
        print(" Nx |  Ny | Tmax [°C] | Tmean [°C] | total_time [s]")
        for row in summary:
            print(f"{row['Nx']:3d} | {row['Ny']:3d} | {row['Tmax']:.6f} | {row['Tmean']:.6f} | {row['total_time']:.6e}")
        print("=" * 120)
        return summary

    def exercise_3_variable_conductivity(self) -> Dict[str, Any]:
        """Exercício 3: condutividade variável k(x,y)."""
        print("\n" + "#" * 120)
        print("EXERCÍCIO 3 - CONDUTIVIDADE VARIÁVEL")
        print("#" * 120)

        cfg = self.clone_cfg(
            solver_mode="sparse",
            use_variable_k=True,
            k_function=variable_k_function(self.base_cfg.Lx, self.base_cfg.Ly),
            use_circle_constraint=False,
            source_function=constant_source(5.0e5),
            TL=10.0,
            TR=30.0,
            TB=top_bottom_temperature_function(self.base_cfg.Lx),
            TT=top_bottom_temperature_function(self.base_cfg.Lx),
        )
        solver = ThermalPlateSolver(cfg)
        result = solver.solve_system()
        solver.print_output_summary(result)

        if cfg.make_plots:
            solver.plot_temperature_contours(result, title="Ex.3 - Contours com k variável", file_name="ex3_contours.png")
            solver.plot_centerline(result, title="Ex.3 - Eixo central com k variável", file_name="ex3_centerline.png")

            X, Y = np.meshgrid(solver.x, solver.y)
            Kmap = cfg.k_function(X, Y)
            fig = plt.figure(figsize=(10, 4.5))
            contour = plt.contourf(X, Y, Kmap, levels=20)
            plt.colorbar(contour, label="k [W/(m.K)]")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("Ex.3 - Mapa da condutividade variável")
            plt.tight_layout()
            if cfg.save_figures:
                fig.savefig(f"{cfg.output_dir}/ex3_kmap.png", dpi=200, bbox_inches="tight")
            plt.show()

        return result

    def exercise_4_Tmax_Tmean_vs_TC(self) -> Dict[str, np.ndarray]:
        """Exercício 4: Tmax e Tmédia como funções de Tc."""
        print("\n" + "#" * 120)
        print("EXERCÍCIO 4 - TMAX E TMÉDIA EM FUNÇÃO DE TC")
        print("#" * 120)

        TC_values = np.linspace(10.0, 60.0, 11)
        Tmax_values = []
        Tmean_values = []

        for TC in TC_values:
            cfg = self.clone_cfg(
                Nx=101,
                Ny=51,
                solver_mode="sparse",
                use_variable_k=False,
                k_constant=0.25,
                use_circle_constraint=True,
                TC=float(TC),
                source_function=constant_source(5.0e5),
                TL=10.0,
                TR=30.0,
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

        fig = plt.figure(figsize=(9, 4.5))
        plt.plot(TC_values, Tmax_values, marker="o", label="Tmax")
        plt.plot(TC_values, Tmean_values, marker="s", label="Tmédia")
        plt.xlabel("TC [°C]")
        plt.ylabel("Temperatura [°C]")
        plt.title("Ex.4 - Tmax e Tmédia em função de TC")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if self.base_cfg.save_figures:
            fig.savefig(f"{self.base_cfg.output_dir}/ex4_tc_scan.png", dpi=200, bbox_inches="tight")
        plt.show()

        print("=" * 120)
        print("TABELA RESUMO - EXERCÍCIO 4")
        print("=" * 120)
        print("TC [°C] | Tmax [°C] | Tmean [°C]")
        for tc, tmax, tmean in zip(TC_values, Tmax_values, Tmean_values):
            print(f"{tc:7.3f} | {tmax:10.6f} | {tmean:11.6f}")
        print("=" * 120)

        return {"TC_values": TC_values, "Tmax_values": Tmax_values, "Tmean_values": Tmean_values}

    def exercise_5_linear_dependence(self, k_index: int = 233) -> Dict[str, Any]:
        """Exercício 5: determina a, b, c em Tk = a TR + b TC + c."""
        print("\n" + "#" * 120)
        print("EXERCÍCIO 5 - DETERMINAÇÃO DE a, b, c")
        print("#" * 120)

        cfg_base = self.clone_cfg(
            Nx=101,
            Ny=51,
            solver_mode="sparse",
            use_variable_k=False,
            k_constant=0.25,
            use_circle_constraint=True,
            source_function=constant_source(5.0e5),
            TL=10.0,
            TB=top_bottom_temperature_function(self.base_cfg.Lx),
            TT=top_bottom_temperature_function(self.base_cfg.Lx),
        )

        nunk = cfg_base.Nx * cfg_base.Ny
        if not (0 <= k_index < nunk):
            raise ValueError(f"O índice global k={k_index} não existe para a malha escolhida.")

        cases = [
            {"TR": 20.0, "TC": 30.0},
            {"TR": 35.0, "TC": 25.0},
            {"TR": 45.0, "TC": 50.0},
        ]

        y = []
        M = []

        for case in cases:
            cfg = self.clone_cfg(
                Nx=cfg_base.Nx,
                Ny=cfg_base.Ny,
                solver_mode=cfg_base.solver_mode,
                use_variable_k=cfg_base.use_variable_k,
                k_constant=cfg_base.k_constant,
                use_circle_constraint=cfg_base.use_circle_constraint,
                source_function=cfg_base.source_function,
                TL=cfg_base.TL,
                TR=case["TR"],
                TB=cfg_base.TB,
                TT=cfg_base.TT,
                TC=case["TC"],
            )
            solver = ThermalPlateSolver(cfg)
            result = solver.solve_system()
            y.append(float(result["T_vec"][k_index]))
            M.append([case["TR"], case["TC"], 1.0])

        M = np.array(M, dtype=float)
        y = np.array(y, dtype=float)
        coeff = np.linalg.solve(M, y)
        a, b, c = coeff

        print("=" * 120)
        print("SISTEMA USADO NO EXERCÍCIO 5")
        print("=" * 120)
        print("M =")
        print(M)
        print("y =")
        print(y)
        print(f"a = {a:.12e}")
        print(f"b = {b:.12e}")
        print(f"c = {c:.12e}")
        print("=" * 120)

        return {"k_index": k_index, "M": M, "y": y, "a": a, "b": b, "c": c}

    def exercise_6_iterative_methods(self) -> List[Dict[str, Any]]:
        """
        Exercício iterativo 1:
        - implementar Jacobi e Gauss-Seidel
        - comparar tempo em função do tamanho da malha e da tolerância
        """
        print("\n" + "#" * 120)
        print("EXERCÍCIO ITERATIVO 1 - JACOBI E GAUSS-SEIDEL")
        print("#" * 120)

        grids = [(21, 11), (41, 21), (81, 41)]
        tols = [1e-4, 1e-6]
        summary: List[Dict[str, Any]] = []

        for Nx, Ny in grids:
            for tol in tols:
                for method in ["jacobi", "gauss-seidel"]:
                    cfg = self.clone_cfg(
                        Nx=Nx,
                        Ny=Ny,
                        solver_mode="dense",
                        use_variable_k=False,
                        k_constant=0.25,
                        source_function=constant_source(5.0e5),
                        TL=10.0,
                        TR=30.0,
                        TB=top_bottom_temperature_function(self.base_cfg.Lx),
                        TT=top_bottom_temperature_function(self.base_cfg.Lx),
                        use_circle_constraint=False,
                    )
                    solver = ThermalPlateSolver(cfg)
                    result = solver.solve_iterative(method=method, tol=tol, maxit=3000, store_history=False)
                    summary.append({
                        "Nx": Nx,
                        "Ny": Ny,
                        "tol": tol,
                        "method": method,
                        "iterations": result["iterations"],
                        "solve_time": result["solve_time"],
                        "Tmax": result["Tmax"],
                    })

        print("=" * 140)
        print("TABELA RESUMO - EXERCÍCIO ITERATIVO 1")
        print("=" * 140)
        print(" Nx |  Ny |   tol   | método        | iterações | tempo_solve [s] | Tmax [°C]")
        for row in summary:
            print(f"{row['Nx']:3d} | {row['Ny']:3d} | {row['tol']:.1e} | {row['method']:<13s} | {row['iterations']:9d} | {row['solve_time']:.6e} | {row['Tmax']:.6f}")
        print("=" * 140)
        return summary

    def exercise_7_animation(self) -> Dict[str, Any]:
        """
        Exercício iterativo 2:
        - elaborar animação da evolução do campo de temperaturas.
        """
        print("\n" + "#" * 120)
        print("EXERCÍCIO ITERATIVO 2 - ANIMAÇÃO DA EVOLUÇÃO DO CAMPO TÉRMICO")
        print("#" * 120)

        cfg = self.clone_cfg(
            Nx=41,
            Ny=21,
            solver_mode="dense",
            use_variable_k=False,
            k_constant=0.25,
            source_function=constant_source(5.0e5),
            TL=10.0,
            TR=30.0,
            TB=top_bottom_temperature_function(self.base_cfg.Lx),
            TT=top_bottom_temperature_function(self.base_cfg.Lx),
            use_circle_constraint=False,
        )
        solver = ThermalPlateSolver(cfg)
        result = solver.solve_iterative(method="gauss-seidel", tol=1e-6, maxit=1000, store_history=True, frame_stride=10)
        solver.plot_residual_history(result, title="Ex. Iterativo 2 - Histórico do residual", file_name="ex_iter2_residual.png")
        solver.save_iteration_animation(result, file_name="ex_iter2_animacao.gif")
        return result

    def exercise_8_iterative_tc_target(self, T_target: float = 39.5, beta: float = 1.0, tol: float = 1e-3, maxit: int = 20) -> Dict[str, Any]:
        """
        Exercício iterativo 3:
        - encontrar Tc tal que Tmax = T* usando iteração escalar

        Estratégia usada:
            Tc_{k+1} = Tc_k - beta * (Tmax(Tc_k) - T_target)

        É uma forma simples de iteração de ponto fixo / correção residual.
        """
        print("\n" + "#" * 120)
        print("EXERCÍCIO ITERATIVO 3 - AJUSTE ITERATIVO DE TC PARA ATINGIR TMAX DESEJADA")
        print("#" * 120)

        cfg = self.clone_cfg(
            Nx=101,
            Ny=51,
            solver_mode="sparse",
            use_variable_k=False,
            k_constant=0.25,
            use_circle_constraint=True,
            source_function=constant_source(5.0e5),
            TL=10.0,
            TR=30.0,
            TB=top_bottom_temperature_function(self.base_cfg.Lx),
            TT=top_bottom_temperature_function(self.base_cfg.Lx),
        )

        Tc = 30.0
        hist_tc = []
        hist_tmax = []
        hist_res = []

        for k in range(maxit):
            cfg_k = self.clone_cfg(**cfg.__dict__, TC=Tc)
            solver = ThermalPlateSolver(cfg_k)
            result = solver.solve_system()
            Tmax = result["Tmax"]
            F = Tmax - T_target

            hist_tc.append(Tc)
            hist_tmax.append(Tmax)
            hist_res.append(F)

            if abs(F) < tol:
                break

            Tc = Tc - beta * F

        hist_tc = np.array(hist_tc, dtype=float)
        hist_tmax = np.array(hist_tmax, dtype=float)
        hist_res = np.array(hist_res, dtype=float)

        print("=" * 120)
        print("TABELA RESUMO - EXERCÍCIO ITERATIVO 3")
        print("=" * 120)
        print("iter | Tc [°C] | Tmax [°C] | F(Tc)=Tmax-T* [°C]")
        for i, (tc, tmax, res) in enumerate(zip(hist_tc, hist_tmax, hist_res)):
            print(f"{i:4d} | {tc:8.4f} | {tmax:9.5f} | {res:16.8e}")
        print("=" * 120)

        fig = plt.figure(figsize=(9, 4.5))
        plt.plot(hist_tc, hist_tmax, marker="o")
        plt.axhline(T_target, linestyle="--")
        plt.xlabel("Tc [°C]")
        plt.ylabel("Tmax [°C]")
        plt.title("Ex. Iterativo 3 - Busca de Tc para Tmax desejada")
        plt.grid(True)
        plt.tight_layout()
        if self.base_cfg.save_figures:
            fig.savefig(f"{self.base_cfg.output_dir}/ex_iter3_tc_target.png", dpi=200, bbox_inches="tight")
        plt.show()

        return {
            "T_target": T_target,
            "beta": beta,
            "tol": tol,
            "iterations": len(hist_tc),
            "Tc_history": hist_tc,
            "Tmax_history": hist_tmax,
            "residual_history": hist_res,
            "Tc_final": hist_tc[-1],
            "Tmax_final": hist_tmax[-1],
        }


# ============================================================
# GUIA DE USO E CONFIGURAÇÃO
# ============================================================


def print_user_guide() -> None:
    """Explica como o usuário pode interagir com o programa."""
    print("=" * 120)
    print("GUIA RÁPIDO DE USO")
    print("=" * 120)
    print("1) Para mudar a malha, altere Nx e Ny em USER_CONFIG.")
    print("2) Para mudar o tamanho da placa, altere Lx e Ly.")
    print("3) Para mudar temperaturas de contorno, altere TL, TR, TB, TT.")
    print("4) Para ligar o círculo interno, use use_circle_constraint=True e ajuste TC.")
    print("5) Para usar k variável, use use_variable_k=True.")
    print("6) Para comparar direto denso/esparso, ligue run_exercise_1.")
    print("7) Para estudar Jacobi e Gauss-Seidel, ligue run_exercise_6.")
    print("8) Para gerar animação das iterações, ligue run_exercise_7.")
    print("9) Para buscar Tc que leve a uma Tmax desejada, ligue run_exercise_8.")
    print("10) Se a execução estiver pesada, desligue exercícios grandes ou reduza as malhas.")
    print("=" * 120)


USER_CONFIG = {
    # Caso único
    "single_run": True,

    # Geometria
    "Lx": 0.02,
    "Ly": 0.01,
    "Nx": 101,
    "Ny": 51,

    # Solver direto
    "solver_mode": "sparse",  # "dense" ou "sparse"
    "preserve_symmetry": True,

    # Contornos
    "TL": 10.0,
    "TR": 30.0,
    "use_nominal_top_bottom_function": True,

    # Fonte
    "source_value": 5.0e5,

    # Condutividade
    "use_variable_k": False,
    "k_constant": 0.25,

    # Região circular
    "use_circle_constraint": False,
    "TC": 30.0,
    "circle_center_x": 0.75 * 0.02,
    "circle_center_y": 0.50 * 0.01,
    "circle_radius": 0.002,

    # Visualização
    "make_plots": True,
    "save_figures": False,
    "output_dir": ".",
    "contour_levels": 20,
    "print_matrix_info": True,
    "print_full_tables": True,

    # Exercícios principais
    "run_exercise_1": True,
    "run_exercise_2": True,
    "run_exercise_3": True,
    "run_exercise_4": True,
    "run_exercise_5": True,

    # Exercícios iterativos
    "run_exercise_6": True,
    "run_exercise_7": False,
    "run_exercise_8": True,

    # Índice k do exercício 5
    "exercise_5_k_index": 233,

    # Parâmetros do exercício iterativo 3
    "exercise_8_T_target": 39.5,
    "exercise_8_beta": 1.0,
    "exercise_8_tol": 1e-3,
    "exercise_8_maxit": 20,
}


def build_base_config_from_user() -> ThermalPlateConfig:
    """Constrói a configuração base a partir do dicionário USER_CONFIG."""
    Lx = float(USER_CONFIG["Lx"])
    Ly = float(USER_CONFIG["Ly"])

    if USER_CONFIG["use_nominal_top_bottom_function"]:
        TB = top_bottom_temperature_function(Lx)
        TT = top_bottom_temperature_function(Lx)
    else:
        TB = 10.0
        TT = 10.0

    cfg = ThermalPlateConfig(
        Lx=Lx,
        Ly=Ly,
        Nx=int(USER_CONFIG["Nx"]),
        Ny=int(USER_CONFIG["Ny"]),
        TL=float(USER_CONFIG["TL"]),
        TR=float(USER_CONFIG["TR"]),
        TB=TB,
        TT=TT,
        source_function=constant_source(float(USER_CONFIG["source_value"])),
        use_variable_k=bool(USER_CONFIG["use_variable_k"]),
        k_constant=float(USER_CONFIG["k_constant"]),
        k_function=variable_k_function(Lx, Ly) if USER_CONFIG["use_variable_k"] else None,
        use_circle_constraint=bool(USER_CONFIG["use_circle_constraint"]),
        circle_center_x=float(USER_CONFIG["circle_center_x"]),
        circle_center_y=float(USER_CONFIG["circle_center_y"]),
        circle_radius=float(USER_CONFIG["circle_radius"]),
        TC=float(USER_CONFIG["TC"]),
        solver_mode=str(USER_CONFIG["solver_mode"]),
        preserve_symmetry=bool(USER_CONFIG["preserve_symmetry"]),
        make_plots=bool(USER_CONFIG["make_plots"]),
        save_figures=bool(USER_CONFIG["save_figures"]),
        output_dir=str(USER_CONFIG["output_dir"]),
        contour_levels=int(USER_CONFIG["contour_levels"]),
        print_matrix_info=bool(USER_CONFIG["print_matrix_info"]),
        print_full_tables=bool(USER_CONFIG["print_full_tables"]),
    )
    return cfg


def main() -> None:
    """Função principal do programa."""
    print_user_guide()
    base_cfg = build_base_config_from_user()

    if USER_CONFIG["single_run"]:
        print("\n" + "#" * 120)
        print("CASO ÚNICO - RESOLUÇÃO DA PLACA TÉRMICA")
        print("#" * 120)
        solver = ThermalPlateSolver(base_cfg)
        solver.print_inputs_summary()
        result = solver.solve_system()
        solver.print_output_summary(result)
        solver.print_matrix_info(result)
        solver.print_final_explanation(result)
        if base_cfg.make_plots:
            solver.plot_temperature_contours(result, title="Caso único - contours", file_name="single_contours.png")
            solver.plot_centerline(result, title="Caso único - eixo central", file_name="single_centerline.png")

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
    if USER_CONFIG["run_exercise_6"]:
        exercises.exercise_6_iterative_methods()
    if USER_CONFIG["run_exercise_7"]:
        exercises.exercise_7_animation()
    if USER_CONFIG["run_exercise_8"]:
        exercises.exercise_8_iterative_tc_target(
            T_target=float(USER_CONFIG["exercise_8_T_target"]),
            beta=float(USER_CONFIG["exercise_8_beta"]),
            tol=float(USER_CONFIG["exercise_8_tol"]),
            maxit=int(USER_CONFIG["exercise_8_maxit"]),
        )


if __name__ == "__main__":
    main()
