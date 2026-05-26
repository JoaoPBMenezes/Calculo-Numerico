# -*- coding: utf-8 -*-
"""
acoplamento_hidraulico_termico_integrado.py
============================================================
INTEGRAÇÃO HIDRÁULICO-TÉRMICA - CAPÍTULO 4
============================================================

Este arquivo integra a parte térmica e a parte hidráulica preservando os
nomes centrais já usados nos códigos originais:

PARTE TÉRMICA PRESERVADA
- ThermalPlateConfig
- BaseLinearSystemSolver
- ThermalPlateSolver
- ThermalPlateSolver.assembly
- ThermalPlateSolver.solve_system
- ij2n, n2ij
- constant_source, top_bottom_temperature_function

PARTE HIDRÁULICA PRESERVADA
- water_viscosity_pa_s
- empirical_viscosity
- edge_lengths
- build_incidence_matrix
- assembly                  -> montagem global hidráulica
- evaluate_flow_bc
- apply_pressure_bc
- hydraulic_conductivities
- solve_network
- compute_power
- nodal_mass_residual
- print_inputs_summary
- print_output_summary
- print_final_explanation

O QUE FOI ACRESCENTADO PARA O CAPÍTULO 4
- TemperatureInterpolator: interpolação 2D linear, cubic e nearest.
- Regras de quadratura em arestas: ponto médio e trapézio, simples e compostas.
- HydroThermalCoupledModel: integra placa térmica + rede hidráulica.
- NetworkInfluenceModel: efeito dos microcanais na condutividade térmica e no termo fonte/sumidouro.
- Funções de exercícios do PDF, focadas nas perguntas destacadas em amarelo:
  4.2.1 itens 1 a 5 e 4.3.3 itens 1 e 2.

DEPENDÊNCIAS
    pip install numpy scipy matplotlib pandas tabulate shapely

SAÍDAS GERADAS
- Pasta: resultados_acoplamento/
- Figuras PNG com mapas de temperatura, rede colorida por nós/arestas e perfis.
- CSVs com tabelas de comparação numérica.

OBSERVAÇÃO IMPORTANTE
O docente disponibiliza uma função externa generate_graph_arrays para montar a rede.
Se os arquivos gera_grafo.py e plota_rede.py estiverem na mesma pasta, o código usa a
função original. Caso contrário, há um gerador substituto compatível, para que o arquivo
rode de forma autônoma e ainda mantenha os nós 0 e 175 exigidos no enunciado.
"""

from __future__ import annotations

# ============================================================
# IMPORTS
# ============================================================

# pathlib organiza caminhos de saída sem depender do sistema operacional.
from pathlib import Path

# time permite medir o tempo computacional.
import time

# dataclasses concentram configurações físicas e numéricas em objetos claros.
from dataclasses import dataclass, field

# typing deixa claro o tipo esperado em cada função, ajudando na manutenção.
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# numpy é a base de vetores, matrizes, coordenadas, álgebra linear e quadratura.
import numpy as np

# pandas é usado apenas para salvar tabelas de resultados de forma organizada.
import pandas as pd

# backend Agg evita erro em ambientes sem janela gráfica. As figuras são salvas em PNG.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# scipy.sparse é essencial porque a matriz da placa térmica 2D é esparsa.
from scipy import sparse
from scipy.sparse.linalg import spsolve

# diferentes tipos de interpolação: linear, nearest e cubic.
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

# KDTree acelera buscas espaciais de proximidade entre pontos da malha e arestas da rede.
from scipy.spatial import cKDTree


# ============================================================
# CONFIGURAÇÃO GERAL DO ESTUDO DO PDF
# ============================================================

OUTPUT_DIR = Path("./resultados_acoplamento")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_INTEGRADO: Dict[str, Any] = {
    # Geometria da placa: 3 cm x 1,5 cm.
    "Lx": 0.03,
    "Ly": 0.015,

    # Propriedades térmicas da placa.
    "k_constant": 0.25,
    "source_value": 5.0e5,
    "TL": 10.0,
    "TR": 30.0,
    "TC": 35.0,
    "circle_radius": 0.0025,

    # O centro do círculo é (2 + R, 0.75) cm = (0.02 + R, 0.0075) m.
    "circle_center_x": 0.02 + 0.0025,
    "circle_center_y": 0.0075,

    # Rede hidráulica.
    "levels": 3,
    "spine_length": 6,
    "coord_scale_to_m": 0.001,
    "width": 500e-6,
    "height": 500e-6,
    "area_constant": 500e-6 * 500e-6,
    "inlet_0": 0,
    "inlet_175": 175,
    "Q0_in": 1.0e-7,
    "Q175_in": 1.0e-6,

    # Malhas citadas nos exercícios do PDF.
    "thermal_mesh_refined": (241, 121),
    "thermal_mesh_coarse": (61, 31),
    "secondary_grid_refined_case": (61, 31),
    "secondary_grid_coarse_case": (31, 16),

    # Opções de execução.
    "run_ex_4-2-1_item-1": True,      # Deduzir regra do trapézio
    "run_ex_4-2-1_item_2": True,     # Interpolação e malhas
    "run_ex_4-2-1_item_3-4-5": True, # Quadratura e hidráulica
    "run_ex_4-3-3_item-1": True,     # Condutividade modificada
    "run_ex_4-3-3_item-2": True,     # Fonte/Sumidouro

    # Modo de execução do item 4.3.
    # O PDF pede também a malha (241,121). Ela está implementada; para rodar tudo,
    # adicione (241,121) na lista abaixo. Por padrão deixei (61,31) e (121,61)
    # para que o script conclua em computadores comuns sem travar.
    "thermal_meshes_43": [(61, 31), (121, 61), (241,121)],

    "dmax_values": [0.00025, 0.0005, 0.001],
    "s0_values": [1e5, -1e5, 5e5, -5e5, 1e6, -1e6],
}


# ============================================================
# TIPOS AUXILIARES DA PARTE TÉRMICA
# ============================================================

Number = Union[int, float]
ScalarOrCallable = Union[Number, Callable[[float], float]]
FieldFunction2D = Callable[[float, float], float]


# ============================================================
# FUNÇÕES DE MAPEAMENTO DE ÍNDICES DA PLACA
# ============================================================

def ij2n(i: int, j: int, Nx: int) -> int:
    """
    Converte o índice 2D (i,j) da malha em índice global n.

    Fundamento matemático:
    A solução da equação de condução é armazenada em um vetor T, pois o sistema
    linear tem a forma A T = b. O mapeamento n = i + j*Nx empilha as linhas da
    malha em um único vetor.
    """
    return i + j * Nx


def n2ij(n: int, Nx: int) -> Tuple[int, int]:
    """Faz o caminho inverso: índice global n -> índices locais (i,j)."""
    j = n // Nx
    i = n % Nx
    return i, j


# ============================================================
# FUNÇÕES DE CAMPO, FONTE E CONTORNO TÉRMICO
# ============================================================

def to_callable(value: ScalarOrCallable) -> Callable[[float], float]:
    """Transforma número constante em função para padronizar o uso dos contornos."""
    if callable(value):
        return value
    return lambda _: float(value)


def constant_source(value: float) -> FieldFunction2D:
    """Retorna uma fonte volumétrica constante f(x,y)=value."""
    return lambda x, y: float(value)


def top_bottom_temperature_function(Lx: float) -> Callable[[float], float]:
    """
    Temperatura prescrita no topo e na base do PDF:
        T_B(x) = T_T(x) = 10 + 20*x/Lx.
    """
    return lambda x: 10.0 + 20.0 * x / Lx


def variable_k_function(Lx: float, Ly: float) -> FieldFunction2D:
    """
    Função mantida por compatibilidade com o código térmico original.
    Não é a função principal do acoplamento; a influência dos canais é tratada
    por NetworkInfluenceModel.
    """
    return lambda x, y: 0.2 + 0.05 * np.sin(3.0 * np.pi * x / Lx) * np.sin(3.0 * np.pi * y / Ly)


# ============================================================
# CONFIGURAÇÃO E SOLVER DA PLACA TÉRMICA
# ============================================================

@dataclass
class ThermalPlateConfig:
    """
    Configuração do problema térmico estacionário.

    Modelo matemático:
        -div(k grad T) = S
    com condições de Dirichlet nas bordas e, opcionalmente, no círculo interno.

    Discretização:
    Usa diferenças finitas/volumes finitos em malha retangular. Cada nó interno
    gera uma equação linear conectando o nó aos vizinhos leste, oeste, norte e sul.
    """

    Lx: float = 0.03
    Ly: float = 0.015
    Nx: int = 61
    Ny: int = 31

    TL: ScalarOrCallable = 10.0
    TR: ScalarOrCallable = 30.0
    TB: ScalarOrCallable = field(default_factory=lambda: top_bottom_temperature_function(0.03))
    TT: ScalarOrCallable = field(default_factory=lambda: top_bottom_temperature_function(0.03))

    source_function: FieldFunction2D = field(default_factory=lambda: constant_source(5.0e5))

    use_variable_k: bool = False
    k_constant: float = 0.25
    k_function: Optional[FieldFunction2D] = None

    use_circle_constraint: bool = True
    circle_center_x: float = 0.0225
    circle_center_y: float = 0.0075
    circle_radius: float = 0.0025
    TC: float = 35.0

    solver_mode: str = "sparse"
    preserve_symmetry: bool = True

    output_dir: Path = OUTPUT_DIR
    contour_levels: int = 24

    def __post_init__(self) -> None:
        # Padroniza contornos como funções de uma variável.
        self.TL = to_callable(self.TL)
        self.TR = to_callable(self.TR)
        self.TB = to_callable(self.TB)
        self.TT = to_callable(self.TT)

        # Se o usuário pedir k variável e não fornecer função, usa a função original.
        if self.use_variable_k and self.k_function is None:
            self.k_function = variable_k_function(self.Lx, self.Ly)


class BaseLinearSystemSolver:
    """Classe base preservada para concentrar métodos de solução linear."""

    def __init__(self) -> None:
        self.last_result: Optional[Dict[str, Any]] = None

    def solve_linear_system_dense(self, A_mod: np.ndarray, b_mod: np.ndarray) -> np.ndarray:
        # np.linalg.solve faz fatoração numérica; é preferível a calcular A^{-1}b.
        return np.linalg.solve(A_mod, b_mod)

    def solve_linear_system_sparse(self, A_mod_sparse: sparse.csr_matrix, b_mod: np.ndarray) -> np.ndarray:
        # spsolve explora que a matriz de uma malha 2D só tem poucos vizinhos por linha.
        return spsolve(A_mod_sparse, b_mod)


class ThermalPlateSolver(BaseLinearSystemSolver):
    """Solver térmico preservado e ampliado para aceitar k(x,y) e S(x,y)."""

    def __init__(self, cfg: ThermalPlateConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Coordenadas nodais da malha.
        self.x = np.linspace(0.0, cfg.Lx, cfg.Nx)
        self.y = np.linspace(0.0, cfg.Ly, cfg.Ny)

        # Passos espaciais usados na discretização.
        self.hx = cfg.Lx / (cfg.Nx - 1)
        self.hy = cfg.Ly / (cfg.Ny - 1)

        # Número total de incógnitas térmicas.
        self.nunk = cfg.Nx * cfg.Ny

        # Identifica antecipadamente todos os nós com temperatura imposta.
        self.dirichlet_ids, self.dirichlet_values = self._build_dirichlet_data()
        self.dirichlet_set = set(self.dirichlet_ids.tolist())
        self.dirichlet_value_by_id = {int(i): float(v) for i, v in zip(self.dirichlet_ids, self.dirichlet_values)}

    def _build_dirichlet_data(self) -> Tuple[np.ndarray, np.ndarray]:
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
        """Avalia k. Se houver acoplamento por microcanais, k_function representa k modificado."""
        if self.cfg.use_variable_k:
            if self.cfg.k_function is None:
                raise ValueError("use_variable_k=True, mas k_function não foi definida.")
            return float(self.cfg.k_function(x, y))
        return float(self.cfg.k_constant)

    def get_source(self, x: float, y: float) -> float:
        """Avalia a fonte térmica volumétrica S(x,y)."""
        return float(self.cfg.source_function(x, y))

    def is_inside_circle_constraint(self, x: float, y: float) -> bool:
        """Testa se o ponto pertence à região circular de temperatura prescrita."""
        if not self.cfg.use_circle_constraint:
            return False
        dx = x - self.cfg.circle_center_x
        dy = y - self.cfg.circle_center_y
        return dx * dx + dy * dy <= self.cfg.circle_radius ** 2

    def get_dirichlet_value_raw(self, i: int, j: int) -> Optional[float]:
        """Retorna temperatura prescrita no nó, se houver."""
        x = self.x[i]
        y = self.y[j]

        # Borda esquerda: T_L.
        if i == 0:
            return float(self.cfg.TL(y))

        # Borda direita: T_R.
        if i == self.cfg.Nx - 1:
            return float(self.cfg.TR(y))

        # Borda inferior: T_B(x).
        if j == 0:
            return float(self.cfg.TB(x))

        # Borda superior: T_T(x).
        if j == self.cfg.Ny - 1:
            return float(self.cfg.TT(x))

        # Inclusão circular interna: T_C.
        if self.is_inside_circle_constraint(x, y):
            return float(self.cfg.TC)

        # Caso contrário, o nó é uma incógnita real do problema.
        return None

    def get_dirichlet_value(self, i: int, j: int) -> Optional[float]:
        return self.get_dirichlet_value_raw(i, j)

    def assembly(self, matrix_mode: str = "sparse") -> Union[np.ndarray, sparse.csr_matrix]:
        """
        Monta a matriz térmica A.

        Fundamento numérico:
        Em cada nó interno, aproximamos o fluxo por interfaces:
            q_e = -k_e (T_E - T_P)/hx
        e fazemos balanço conservativo. Isso gera coeficientes positivos na diagonal
        e negativos nos vizinhos. Para k variável, k é avaliado no ponto médio das faces.
        """
        if matrix_mode not in ("dense", "sparse"):
            raise ValueError("matrix_mode deve ser 'dense' ou 'sparse'.")

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        # Fatores geométricos de volume finito: área da face/distância entre nós.
        rx = self.hy / self.hx
        ry = self.hx / self.hy

        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)

                # Pontos com temperatura prescrita serão impostos depois via Dirichlet.
                if Ic in self.dirichlet_set:
                    rows.append(Ic)
                    cols.append(Ic)
                    data.append(1.0)
                    continue

                Ie = ij2n(i + 1, j, self.cfg.Nx)
                Iw = ij2n(i - 1, j, self.cfg.Nx)
                In = ij2n(i, j + 1, self.cfg.Nx)
                Is = ij2n(i, j - 1, self.cfg.Nx)

                xc = self.x[i]
                yc = self.y[j]

                # k nas quatro faces do volume de controle.
                ke = self.get_k(xc + 0.5 * self.hx, yc)
                kw = self.get_k(xc - 0.5 * self.hx, yc)
                kn = self.get_k(xc, yc + 0.5 * self.hy)
                ks = self.get_k(xc, yc - 0.5 * self.hy)

                # Coeficientes conservativos. A diagonal é a soma dos fluxos que saem.
                aE = ke * rx
                aW = kw * rx
                aN = kn * ry
                aS = ks * ry
                aP = aE + aW + aN + aS

                rows.extend([Ic, Ic, Ic, Ic, Ic])
                cols.extend([Ic, Ie, Iw, In, Is])
                data.extend([aP, -aE, -aW, -aN, -aS])

        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(self.nunk, self.nunk)).tocsr()
        if matrix_mode == "sparse":
            return A_sparse
        return A_sparse.toarray()

    def build_rhs(self) -> np.ndarray:
        """Monta b. Para fonte volumétrica S, a contribuição nodal é S*hx*hy."""
        b = np.zeros(self.nunk, dtype=float)
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)
                if Ic in self.dirichlet_set:
                    # A linha de Dirichlet já foi montada como T_i = valor_prescrito.
                    b[Ic] = self.dirichlet_value_by_id[Ic]
                else:
                    b[Ic] = self.get_source(self.x[i], self.y[j]) * self.hx * self.hy
        return b

    def apply_dirichlet_bc_sparse(self, A: sparse.csr_matrix, b: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Impõe T = T_prescrita nos nós de Dirichlet.

        preserve_symmetry=True mantém a matriz simétrica: a coluna do nó prescrito é
        movida para o lado direito antes de zerar linha e coluna.
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

    def apply_dirichlet_bc_dense(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A_mod = A.copy()
        b_mod = b.copy()
        for Ic, Tpresc in zip(self.dirichlet_ids, self.dirichlet_values):
            if self.cfg.preserve_symmetry:
                b_mod -= A_mod[:, Ic] * Tpresc
                A_mod[:, Ic] = 0.0
                A_mod[Ic, :] = 0.0
                A_mod[Ic, Ic] = 1.0
                b_mod[Ic] = Tpresc
            else:
                A_mod[Ic, :] = 0.0
                A_mod[Ic, Ic] = 1.0
                b_mod[Ic] = Tpresc
        return A_mod, b_mod

    def solve_system(self, matrix_mode: Optional[str] = None) -> Dict[str, Any]:
        """Resolve A*T=b e mede o tempo de montagem, contorno e solução."""
        if matrix_mode is None:
            matrix_mode = self.cfg.solver_mode

        t0 = time.perf_counter()
        A = self.assembly(matrix_mode=matrix_mode)
        b = self.build_rhs()
        t1 = time.perf_counter()

        # As linhas de Dirichlet já foram inseridas diretamente na montagem
        # como equações T_i = valor_prescrito. Isso evita o custo alto de zerar
        # colunas em matrizes grandes e torna a execução dos exercícios bem mais rápida.
        A_mod, b_mod = A, b
        if matrix_mode == "dense":
            T_vec = self.solve_linear_system_dense(A_mod, b_mod)
        else:
            T_vec = self.solve_linear_system_sparse(A_mod, b_mod)
        t2 = time.perf_counter()

        T_grid = T_vec.reshape((self.cfg.Ny, self.cfg.Nx))
        result = {
            "A": A,
            "A_mod": A_mod,
            "b": b,
            "b_mod": b_mod,
            "T_vec": T_vec,
            "T_grid": T_grid,
            "x": self.x.copy(),
            "y": self.y.copy(),
            "Tmax": float(np.max(T_grid)),
            "Tmean": float(np.mean(T_grid)),
            "assembly_time": t1 - t0,
            "solve_time": t2 - t1,
            "total_time": t2 - t0,
        }
        self.last_result = result
        return result


# ============================================================
# FUNÇÕES HIDRÁULICAS
# ============================================================

def water_viscosity_pa_s(T_celsius: float) -> float:
    """
    Função preservada do código original.
    Neste trabalho, a viscosidade usada no acoplamento é empirical_viscosity,
    pois é exatamente a fórmula dada no PDF do capítulo 4.
    """
    A = 2.414e-5
    B = 247.8
    C = 140.0
    return A * 10 ** (B / ((T_celsius + 273.15) - C))


def empirical_viscosity(T_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Viscosidade dinâmica do fluido dada no PDF:
        mu(T) = 0.001791/(1 + 0.03368*T + 0.000221*T^2).
    T em °C e mu em Pa.s.
    """
    T = np.asarray(T_celsius, dtype=float)
    mu = 0.001791 / (1.0 + 0.03368 * T + 0.000221 * T ** 2)
    if np.isscalar(T_celsius):
        return float(mu)
    return mu


def rectangular_area(width: float, height: float) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("Largura e altura devem ser positivas.")
    return width * height


def equivalent_diameter_from_area(area: float) -> float:
    if area <= 0:
        raise ValueError("A área deve ser positiva.")
    return float(np.sqrt(4.0 * area / np.pi))


def edge_lengths(Xno: np.ndarray, conec: np.ndarray) -> np.ndarray:
    """Comprimento euclidiano de cada aresta/canal."""
    p0 = Xno[conec[:, 0], :]
    p1 = Xno[conec[:, 1], :]
    return np.linalg.norm(p1 - p0, axis=1)


def build_incidence_matrix(conec: np.ndarray, nv: Optional[int] = None) -> np.ndarray:
    """
    Matriz de incidência D. Cada linha é uma aresta i->j:
        D[k,i] = 1, D[k,j] = -1.
    Assim, a queda de pressão na aresta é dp = D @ p = p_i - p_j.
    """
    if nv is None:
        nv = int(np.max(conec)) + 1
    D = np.zeros((conec.shape[0], nv), dtype=float)
    for k, (i, j) in enumerate(conec.astype(int)):
        D[k, i] = 1.0
        D[k, j] = -1.0
    return D


def assembly(conec: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Monta a matriz global hidráulica.

    Fundamento:
    Para cada aresta k entre i e j, q_k = C_k(p_i - p_j).
    A conservação de massa em cada nó gera a matriz tipo grafo/laplaciana.
    """
    nv = int(np.max(conec)) + 1
    A = np.zeros((nv, nv), dtype=float)
    for k, (i, j) in enumerate(conec.astype(int)):
        ck = float(C[k])
        A[i, i] += ck
        A[j, j] += ck
        A[i, j] -= ck
        A[j, i] -= ck
    return A


def evaluate_flow_spec(spec: dict, t: float) -> float:
    """Avalia uma vazão prescrita constante, senoidal ou cossenoidal."""
    flow_type = spec["type"].lower()
    if flow_type == "constant":
        return float(spec["value"])
    if flow_type == "sin":
        return float(spec["mean"] + spec["amp"] * np.sin(2.0 * np.pi * spec["freq"] * t + spec.get("phase", 0.0)))
    if flow_type == "cos":
        return float(spec["mean"] + spec["amp"] * np.cos(2.0 * np.pi * spec["freq"] * t + spec.get("phase", 0.0)))
    raise ValueError(f"Tipo de vazão inválido: {flow_type}")


def evaluate_flow_bc(flow_bc: dict, t: float, nv: int) -> np.ndarray:
    """Monta o vetor de vazões nodais impostas. Entrada é positiva."""
    b = np.zeros(nv, dtype=float)
    for node, spec in flow_bc.items():
        if 0 <= int(node) < nv:
            b[int(node)] += evaluate_flow_spec(spec, t)
    return b


def apply_pressure_bc(A: np.ndarray, b: np.ndarray, pressure_bc: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Impõe pressões prescritas substituindo a equação do nó por p_i = valor."""
    if len(pressure_bc) == 0:
        raise ValueError("É necessário prescrever pelo menos uma pressão nodal.")
    A_mod = A.copy()
    b_mod = b.copy()
    for node, p_value in pressure_bc.items():
        node = int(node)
        A_mod[node, :] = 0.0
        A_mod[node, node] = 1.0
        b_mod[node] = float(p_value)
    return A_mod, b_mod


def get_area_per_edge(conec: np.ndarray, cfg: dict) -> np.ndarray:
    """Área de cada microcanal. Por padrão usa 500 µm x 500 µm, como no PDF."""
    nc = conec.shape[0]
    if cfg.get("area_per_edge") is not None:
        area_edge = np.array(cfg["area_per_edge"], dtype=float)
        if len(area_edge) != nc:
            raise ValueError("area_per_edge deve ter o mesmo tamanho do número de arestas.")
        return area_edge
    area = cfg.get("area_constant", cfg.get("width", 500e-6) * cfg.get("height", 500e-6))
    return np.full(nc, float(area), dtype=float)


def hydraulic_conductivities(Xno: np.ndarray, conec: np.ndarray, cfg: dict) -> dict:
    """
    Calcula condutância C de cada canal.

    No acoplamento, cfg pode conter mu_per_edge, que vem de mu(<T_k>) ou de
    <mu(T)>_k. Se não houver, usa uma temperatura global.
    """
    L = edge_lengths(Xno, conec)
    area_edge = get_area_per_edge(conec, cfg)
    D_eq = np.sqrt(4.0 * area_edge / np.pi)

    if "mu_per_edge" in cfg and cfg["mu_per_edge"] is not None:
        mu = np.array(cfg["mu_per_edge"], dtype=float)
        if len(mu) != len(L):
            raise ValueError("mu_per_edge deve ter o mesmo tamanho do número de arestas.")
    else:
        T_global = float(cfg.get("temperature_celsius", 25.0))
        mu = np.full_like(L, empirical_viscosity(T_global), dtype=float)

    # Modelo tipo Hagen-Poiseuille com diâmetro equivalente, preservando a ideia do código original.
    kappa = np.pi * D_eq ** 4 / (128.0 * mu)
    C = kappa / L

    return {
        "mu": mu,
        "lengths": L,
        "area_edge": area_edge,
        "diameter_eq_edge": D_eq,
        "kappa_edge": kappa,
        "conductance_edge": C,
    }


def solve_network(conec: np.ndarray, C: np.ndarray, pressure_bc: dict, flow_bc: dict, t: float) -> dict:
    """Resolve a rede hidráulica estacionária para as vazões prescritas no tempo t."""
    A = assembly(conec, C)
    nv = A.shape[0]
    b = evaluate_flow_bc(flow_bc, t, nv)
    A_mod, b_mod = apply_pressure_bc(A, b, pressure_bc)

    # Sistema linear da rede: A p = b, onde p são pressões nodais.
    p = np.linalg.solve(A_mod, b_mod)

    D = build_incidence_matrix(conec, nv)
    K = np.diag(C)
    dp_edge = D @ p
    q = C * dp_edge

    return {
        "A": A,
        "A_mod": A_mod,
        "b": b,
        "b_mod": b_mod,
        "p": p,
        "D": D,
        "K": K,
        "dp_edge": dp_edge,
        "q": q,
    }


def compute_power(p: np.ndarray, D: np.ndarray, K: np.ndarray) -> float:
    """
    Potência hidráulica dissipada/consumida pela rede:
        P = Δp^T K Δp = sum(C_k * Δp_k^2).
    """
    dp = D @ p
    return float(dp.T @ K @ dp)


def nodal_mass_residual(conec: np.ndarray, q: np.ndarray, imposed_b: np.ndarray) -> np.ndarray:
    """Resíduo de conservação de massa em cada nó."""
    nv = len(imposed_b)
    residual = -imposed_b.copy()
    for k, (i, j) in enumerate(conec.astype(int)):
        residual[i] += q[k]
        residual[j] -= q[k]
    return residual


def print_inputs_summary(cfg: dict, hydraulic_data: dict, Xno: np.ndarray, conec: np.ndarray) -> None:
    """Resumo preservado da parte hidráulica."""
    print("\n" + "=" * 88)
    print("ENTRADAS DA REDE HIDRÁULICA")
    print("=" * 88)
    print(f"nós = {Xno.shape[0]}, arestas = {conec.shape[0]}")
    print(f"área média = {np.mean(hydraulic_data['area_edge']):.6e} m²")
    print(f"comprimento médio = {np.mean(hydraulic_data['lengths']):.6e} m")
    print(f"mu média = {np.mean(hydraulic_data['mu']):.6e} Pa.s")
    print("pressões prescritas:", cfg.get("pressure_bc", {}))
    print("vazões prescritas:", cfg.get("flow_bc", {}))
    print("=" * 88)


def print_output_summary(cfg: dict, hydraulic_data: dict, result: dict, conec: np.ndarray) -> None:
    """Resumo preservado das saídas hidráulicas."""
    p = result["p"]
    q = result["q"]
    power = compute_power(p, result["D"], result["K"])
    residual = nodal_mass_residual(conec, q, result["b"])
    print("\n" + "=" * 88)
    print("SAÍDAS DA REDE HIDRÁULICA")
    print("=" * 88)
    print(f"pressão máxima = {np.max(p):.6e} Pa")
    print(f"pressão mínima = {np.min(p):.6e} Pa")
    print(f"potência total = {power:.6e} W")
    print(f"máximo resíduo de massa = {np.max(np.abs(residual)):.6e} m³/s")
    print("=" * 88)


def print_final_explanation(cfg: dict, hydraulic_data: dict, result: dict) -> None:
    """Explicação final preservada, agora resumida para o acoplamento."""
    print("\nINTERPRETAÇÃO FINAL")
    print("- A temperatura altera mu(T), que altera C_k e, por isso, muda as pressões.")
    print("- Como mu(T) diminui quando T aumenta, canais mais quentes tendem a conduzir melhor.")
    print("- A potência total é calculada por soma de C_k*(Δp_k)^2 em todas as arestas.")


# ============================================================
# GERAÇÃO DA REDE HIDRÁULICA
# ============================================================

try:
    from gera_grafo import generate_graph_arrays as _external_generate_graph_arrays
    print(">> SUCESSO: 'gera_grafo.py' do professor foi importado!")
except Exception as e:
    print(f"\n[ERRO] Não foi possível carregar 'gera_grafo.py': {e}")
    _external_generate_graph_arrays = None

try:
    from plota_rede import PlotaRede
    print(">> SUCESSO: 'plota_rede.py' do professor foi importado!")
except Exception as e:
    print(f"\n[ERRO] Não foi possível carregar 'plota_rede.py': {e}")
    PlotaRede = None


def generate_graph_arrays(complex_level: int = 3, spine_length: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper compatível com o PDF.

    Se gera_grafo.py existir, usa a função original. Caso contrário, cria uma rede
    retangular 16x11 = 176 nós, preservando os nós 0 e 175 citados no enunciado.
    As coordenadas retornam em milímetros, tal como o arquivo original parecia fazer.
    """
    if _external_generate_graph_arrays is not None:
        return _external_generate_graph_arrays(complex_level)

    # Gerador substituto: rede retangular incorporada em 20 mm x 10 mm.
    # 16*11 = 176 nós => índice máximo 175.
    nx_nodes = 16
    ny_nodes = 11
    x = np.linspace(0.0, (spine_length - 1) * 4.0, nx_nodes)  # 0 a 20 mm
    y = np.linspace(-5.0, 5.0, ny_nodes)                     # centralizada em y=0 mm

    X: List[Tuple[float, float]] = []
    for jj in range(ny_nodes):
        for ii in range(nx_nodes):
            X.append((x[ii], y[jj]))
    Xno = np.array(X, dtype=float)

    conec: List[Tuple[int, int]] = []
    for jj in range(ny_nodes):
        for ii in range(nx_nodes):
            n = ii + jj * nx_nodes
            if ii + 1 < nx_nodes:
                conec.append((n, n + 1))
            if jj + 1 < ny_nodes:
                conec.append((n, n + nx_nodes))
    return Xno, np.array(conec, dtype=int)


def build_pdf_network(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Monta a rede do PDF, converte mm->m e centraliza em y na placa."""
    Xno, conec = generate_graph_arrays(cfg["levels"], cfg["spine_length"])
    Xno = np.asarray(Xno, dtype=float) * cfg["coord_scale_to_m"]
    conec = np.asarray(conec, dtype=int)

    # O PDF desloca a rede em y por 0.5*Ly para embuti-la no centro vertical da placa.
    Xno[:, 1] += 0.5 * cfg["Ly"]

    # Saída hidráulica: nó mais próximo de xout=(spine_length-1)*4 mm e y=Ly/2.
    xout = cfg["coord_scale_to_m"] * (cfg["spine_length"] - 1) * 4.0
    target = np.array([xout, 0.5 * cfg["Ly"]])
    outlet_node = int(np.argmin(np.linalg.norm(Xno - target, axis=1)))
    return Xno, conec, outlet_node


# ============================================================
# INTERPOLAÇÃO BIDIMENSIONAL DE TEMPERATURA
# ============================================================

class TemperatureInterpolator:
    """
    Interpolador 2D do campo térmico, como pedido no PDF.

    Entrada: x, y e T_grid com shape (Ny, Nx).
    Saída: T(x,y) em pontos arbitrários, inclusive nós e pontos internos de arestas.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, T_grid: np.ndarray, method: str = "linear") -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.method = method.lower()

        if self.method in ("linear", "nearest"):
            # RegularGridInterpolator usa os eixos na mesma ordem do array; T.T tem shape (Nx,Ny).
            self._interp = RegularGridInterpolator(
                (self.x, self.y), self.T_grid.T,
                method=self.method,
                bounds_error=False,
                fill_value=None,
            )
            self._spline = None
        elif self.method == "cubic":
            # RectBivariateSpline fornece interpolação cúbica suave em grade retangular.
            self._interp = None
            self._spline = RectBivariateSpline(self.x, self.y, self.T_grid.T, kx=3, ky=3)
        else:
            raise ValueError("method deve ser 'linear', 'nearest' ou 'cubic'.")

    def __call__(self, pts_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xy, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        if self.method in ("linear", "nearest"):
            return np.asarray(self._interp(pts), dtype=float)
        return np.asarray(self._spline.ev(pts[:, 0], pts[:, 1]), dtype=float)


# ============================================================
# QUADRATURAS EM ARESTAS DA REDE
# ============================================================

def edge_quadrature_points(p0: np.ndarray, p1: np.ndarray, rule: str, subdivisions: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera pontos e pesos para integrar T(p(s)) ao longo de uma aresta.

    Ponto médio composto:
        s_n = (n - 1/2) Δs, n=1,...,N.
    Trapézio composto:
        usa os N+1 pontos das extremidades dos subintervalos, com pesos 1/2 nas pontas.
    """
    if subdivisions < 1:
        raise ValueError("subdivisions deve ser >= 1.")

    rule = rule.lower()
    if rule == "midpoint":
        alpha = (np.arange(subdivisions) + 0.5) / subdivisions
        weights = np.full(subdivisions, 1.0 / subdivisions, dtype=float)
    elif rule == "trapezoid":
        alpha = np.linspace(0.0, 1.0, subdivisions + 1)
        weights = np.full(subdivisions + 1, 1.0 / subdivisions, dtype=float)
        weights[0] *= 0.5
        weights[-1] *= 0.5
    else:
        raise ValueError("rule deve ser 'midpoint' ou 'trapezoid'.")

    pts = p0[None, :] + alpha[:, None] * (p1 - p0)[None, :]
    return pts, weights


def mean_edge_temperature(
    Xno: np.ndarray,
    conec: np.ndarray,
    interpolator: TemperatureInterpolator,
    rule: str,
    subdivisions: int,
) -> np.ndarray:
    """
    Calcula <T_k> em todas as arestas pela quadratura escolhida.

    Como <T_k> = (1/L_k) ∫ T(p(s)) ds, e ds = L_k dα, basta calcular
    a média ponderada de T nos pontos de quadratura em α∈[0,1].
    """
    values = np.zeros(conec.shape[0], dtype=float)
    for k, (i, j) in enumerate(conec.astype(int)):
        pts, weights = edge_quadrature_points(Xno[i], Xno[j], rule, subdivisions)
        T_pts = interpolator(pts)
        values[k] = float(np.sum(weights * T_pts))
    return values


def mean_edge_viscosity_direct(
    Xno: np.ndarray,
    conec: np.ndarray,
    interpolator: TemperatureInterpolator,
    rule: str,
    subdivisions: int,
) -> np.ndarray:
    """
    Resposta do item 4.2.1(5): em vez de calcular <T_k> e depois mu(<T_k>),
    pode-se integrar diretamente a viscosidade ao longo da aresta:
        <mu_k> = (1/L_k) ∫ mu(T(p(s))) ds.

    Essa estratégia é mais física quando mu(T) é não linear, porque em geral:
        mu(<T>) != <mu(T)>.
    """
    values = np.zeros(conec.shape[0], dtype=float)
    for k, (i, j) in enumerate(conec.astype(int)):
        pts, weights = edge_quadrature_points(Xno[i], Xno[j], rule, subdivisions)
        T_pts = interpolator(pts)
        values[k] = float(np.sum(weights * empirical_viscosity(T_pts)))
    return values


# ============================================================
# DISTÂNCIA PONTO-SEGMENTO E INFLUÊNCIA DA REDE NA PLACA
# ============================================================

def point_segment_distance_batch(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Distância de vários pontos a um segmento AB.

    Cálculo vetorial:
    projeta P-A sobre B-A, trunca o parâmetro t em [0,1] e mede |P - (A+t(B-A))|.
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)
    t = ((points - a) @ ab) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - closest, axis=1)


class NetworkInfluenceModel:
    """
    Modelo geométrico de proximidade entre malha térmica e rede hidráulica.

    Implementa o que o PDF chama de mapa de proximidade:
    para cada ponto p, identifica arestas dentro de d_max e calcula distâncias d_j.
    """

    def __init__(self, Xno: np.ndarray, conec: np.ndarray, d_max: float, k0: float = 0.25) -> None:
        self.Xno = np.asarray(Xno, dtype=float)
        self.conec = np.asarray(conec, dtype=int)
        self.d_max = float(d_max)
        self.k0 = float(k0)

        # Centros das arestas para uma primeira busca rápida de candidatos.
        self.edge_a = self.Xno[self.conec[:, 0]]
        self.edge_b = self.Xno[self.conec[:, 1]]
        self.edge_mid = 0.5 * (self.edge_a + self.edge_b)
        self.edge_half_length = 0.5 * np.linalg.norm(self.edge_b - self.edge_a, axis=1)
        self.tree = cKDTree(self.edge_mid)

        # Raio conservador: se o centro da aresta está muito longe, o segmento todo está longe.
        self.search_radius = self.d_max + float(np.max(self.edge_half_length))

    def nearby_edges_with_distances(self, point: np.ndarray) -> List[Tuple[int, float]]:
        """Retorna [(índice_aresta, distância)] para arestas dentro de d_max."""
        candidate_ids = self.tree.query_ball_point(point, self.search_radius)
        out: List[Tuple[int, float]] = []
        p = np.asarray(point, dtype=float).reshape(1, 2)
        for eid in candidate_ids:
            d = point_segment_distance_batch(p, self.edge_a[eid], self.edge_b[eid])[0]
            if d <= self.d_max:
                out.append((int(eid), float(d)))
        return out

    def CreateMapDistance(self, Lx: float, Ly: float, Nx: int, Ny: int) -> Dict[int, List[Tuple[int, float]]]:
        """
        Nome mantido conforme o PDF: cria o mapa de proximidade em todos os nós da grade.
        """
        x = np.linspace(0.0, Lx, Nx)
        y = np.linspace(0.0, Ly, Ny)
        mapa: Dict[int, List[Tuple[int, float]]] = {}
        for j, yy in enumerate(y):
            for i, xx in enumerate(x):
                idx = ij2n(i, j, Nx)
                mapa[idx] = self.nearby_edges_with_distances(np.array([xx, yy]))
        return mapa

    def k_modified_at_point(self, x: float, y: float) -> float:
        """
        Condutividade modificada do PDF:
            k_f = k0 * (1 + sum_{j in V_f} 1/(1+d_j)).
        """
        prox = self.nearby_edges_with_distances(np.array([x, y], dtype=float))
        return self.k0 * (1.0 + sum(1.0 / (1.0 + d) for _, d in prox))

    def source_extra_at_point(self, x: float, y: float, S0: float, intensity_edge: np.ndarray) -> float:
        """
        Fonte/sumidouro gaussiano do PDF:
            S_p = S0 * sum I_j * exp(-d_j²/(2σ²)), σ=d_max/2.
        """
        prox = self.nearby_edges_with_distances(np.array([x, y], dtype=float))
        if not prox:
            return 0.0
        sigma = self.d_max / 2.0
        total = 0.0
        for eid, d in prox:
            total += intensity_edge[eid] * np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        return float(S0 * total)


def CreateMapDistance(Lx: float, Ly: float, Nx: int, Ny: int, Xno: np.ndarray, conec: np.ndarray, d_max: float) -> Dict[int, List[Tuple[int, float]]]:
    """Função global com nome igual ao PDF, delegando para NetworkInfluenceModel."""
    return NetworkInfluenceModel(Xno, conec, d_max).CreateMapDistance(Lx, Ly, Nx, Ny)


# ============================================================
# CLASSE DE ACOPLAMENTO HIDRÁULICO-TÉRMICO
# ============================================================

class HydroThermalCoupledModel:
    """Organiza a integração one-way entre placa térmica e rede hidráulica."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.Xno, self.conec, self.outlet_node = build_pdf_network(cfg)

    def make_thermal_config(self, Nx: int, Ny: int, *, k_function: Optional[FieldFunction2D] = None,
                            source_function: Optional[FieldFunction2D] = None) -> ThermalPlateConfig:
        """Cria a configuração térmica padrão do PDF para uma malha específica."""
        return ThermalPlateConfig(
            Lx=self.cfg["Lx"],
            Ly=self.cfg["Ly"],
            Nx=Nx,
            Ny=Ny,
            TL=self.cfg["TL"],
            TR=self.cfg["TR"],
            TB=top_bottom_temperature_function(self.cfg["Lx"]),
            TT=top_bottom_temperature_function(self.cfg["Lx"]),
            source_function=source_function or constant_source(self.cfg["source_value"]),
            use_variable_k=(k_function is not None),
            k_constant=self.cfg["k_constant"],
            k_function=k_function,
            use_circle_constraint=True,
            circle_center_x=self.cfg["circle_center_x"],
            circle_center_y=self.cfg["circle_center_y"],
            circle_radius=self.cfg["circle_radius"],
            TC=self.cfg["TC"],
            solver_mode="sparse",
            output_dir=OUTPUT_DIR,
        )

    def solve_thermal(self, Nx: int, Ny: int, *, k_function: Optional[FieldFunction2D] = None,
                      source_function: Optional[FieldFunction2D] = None) -> Dict[str, Any]:
        cfg = self.make_thermal_config(Nx, Ny, k_function=k_function, source_function=source_function)
        solver = ThermalPlateSolver(cfg)
        return solver.solve_system()

    def hydraulic_cfg_from_mu(self, mu_per_edge: np.ndarray) -> Dict[str, Any]:
        """Config hidráulica com vazões e pressão do PDF."""
        return {
            "width": self.cfg["width"],
            "height": self.cfg["height"],
            "area_constant": self.cfg["area_constant"],
            "mu_per_edge": mu_per_edge,
            "pressure_bc": {self.outlet_node: 0.0},
            "flow_bc": {
                self.cfg["inlet_0"]: {"type": "constant", "value": self.cfg["Q0_in"]},
                self.cfg["inlet_175"]: {"type": "constant", "value": self.cfg["Q175_in"]},
            },
        }

    def solve_hydraulic_from_edge_temperatures(self, T_edge_mean: np.ndarray) -> Dict[str, Any]:
        """Calcula mu(<T>) em cada aresta, atualiza C e resolve a rede."""
        mu_edge = empirical_viscosity(T_edge_mean)
        hcfg = self.hydraulic_cfg_from_mu(mu_edge)
        hydraulic_data = hydraulic_conductivities(self.Xno, self.conec, hcfg)
        result = solve_network(self.conec, hydraulic_data["conductance_edge"], hcfg["pressure_bc"], hcfg["flow_bc"], t=0.0)
        return {
            "hydraulic_cfg": hcfg,
            "hydraulic_data": hydraulic_data,
            "result": result,
            "pmax": float(np.max(result["p"])),
            "pmin": float(np.min(result["p"])),
            "power": compute_power(result["p"], result["D"], result["K"]),
        }

    def solve_hydraulic_from_edge_viscosities(self, mu_edge_mean: np.ndarray) -> Dict[str, Any]:
        """Alternativa do item 5: usa <mu(T)> diretamente em vez de mu(<T>)."""
        hcfg = self.hydraulic_cfg_from_mu(mu_edge_mean)
        hydraulic_data = hydraulic_conductivities(self.Xno, self.conec, hcfg)
        result = solve_network(self.conec, hydraulic_data["conductance_edge"], hcfg["pressure_bc"], hcfg["flow_bc"], t=0.0)
        return {
            "hydraulic_cfg": hcfg,
            "hydraulic_data": hydraulic_data,
            "result": result,
            "pmax": float(np.max(result["p"])),
            "pmin": float(np.min(result["p"])),
            "power": compute_power(result["p"], result["D"], result["K"]),
        }


# ============================================================
# PLOTS
# ============================================================

def save_contour_plot(x: np.ndarray, y: np.ndarray, T: np.ndarray, title: str, filename: str,
                      Xno: Optional[np.ndarray] = None, conec: Optional[np.ndarray] = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cf = ax.contourf(x, y, T, levels=24)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label="T [°C]")

    # === SOBREPOSIÇÃO DA REDE HIDRÁULICA ===
    if Xno is not None and conec is not None:
        segments = [(Xno[i], Xno[j]) for i, j in conec.astype(int)]
        lc = LineCollection(segments, colors="black", linewidths=0.6, alpha=0.4, zorder=2)
        ax.add_collection(lc)
        ax.scatter(Xno[:, 0], Xno[:, 1], color="black", s=3, alpha=0.4, zorder=3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)

    # Sobreposição da rede hidráulica, se disponível, para mostrar a relação entre fluxo e temperatura.
    if Xno is not None and conec is not None:
        segments = [(Xno[i], Xno[j]) for i, j in conec.astype(int)]
        # Desenha as linhas das arestas em preto sutil por cima do contorno
        lc = LineCollection(segments, colors="black", linewidths=0.6, alpha=0.4, zorder=2)
        ax.add_collection(lc)
        # Desenha os pontos dos nós
        ax.scatter(Xno[:, 0], Xno[:, 1], color="black", s=3, alpha=0.4, zorder=3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)

def save_graph_nodes_temperature(Xno: np.ndarray, conec: np.ndarray, T_nodes: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    segments = [(Xno[i], Xno[j]) for i, j in conec.astype(int)]
    lc = LineCollection(segments, linewidths=0.7, alpha=0.45)
    ax.add_collection(lc)
    sc = ax.scatter(Xno[:, 0], Xno[:, 1], c=T_nodes, s=22)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="T nos nós [°C]")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)


def save_graph_edges_temperature(Xno: np.ndarray, conec: np.ndarray, T_edge: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    segments = [(Xno[i], Xno[j]) for i, j in conec.astype(int)]
    lc = LineCollection(segments, array=T_edge, linewidths=2.0)
    ax.add_collection(lc)
    ax.scatter(Xno[:, 0], Xno[:, 1], s=4, alpha=0.25)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.colorbar(lc, ax=ax, label="<T> na aresta [°C]")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)


def save_profile_plots(x: np.ndarray, y: np.ndarray, T: np.ndarray, title: str, filename: str) -> None:
    """Perfis horizontal em y=Ly/2 e vertical em x próximo do círculo."""
    jy = len(y) // 2
    ix = int(np.argmin(np.abs(x - CONFIG_INTEGRADO["circle_center_x"])))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, T[jy, :], marker="o", markersize=2)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("T [°C]")
    axes[0].set_title("perfil horizontal")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(y, T[:, ix], marker="o", markersize=2)
    axes[1].set_xlabel("y [m]")
    axes[1].set_ylabel("T [°C]")
    axes[1].set_title("perfil vertical")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)


# ============================================================
# EXERCÍCIOS 4.2.1 DO PDF
# ============================================================

def print_theoretical_4_2_1_item_1() -> None:
    """Resposta teórica para o Item 4.2.1 (1)"""
    print("\n" + "=" * 80)
    print("RESPOSTA TEÓRICA - EXERCÍCIO 4.2.1(1)")
    print("=" * 80)
    print("Dedução da Regra do Trapézio Composta para a temperatura média <T_k>:")
    print("\n1. Divisão do Domínio:")
    print("   A aresta de comprimento L_k é dividida em N subintervalos iguais.")
    print("   O passo espacial é Δs = L_k / N.")
    print("\n2. Aplicação do Trapézio Simples:")
    print("   Em cada subintervalo [s_{n-1}, s_n], a área é aproximada por um trapézio:")
    print("   Integral ≈ (Δs/2) * [T(s_{n-1}) + T(s_n)]")
    print("\n3. Composição (Soma dos Subintervalos):")
    print("   Ao somar todos os intervalos, os nós internos (compartilhados entre dois")
    print("   subintervalos vizinhos) são somados duas vezes. As pontas, apenas uma.")
    print("   Integral_Total ≈ (Δs/2) * [T(s_0) + 2*Σ_{n=1}^{N-1} T(s_n) + T(s_N)]")
    print("\n4. Cálculo da Média:")
    print("   Como a média é <T_k> = Integral_Total / L_k, e sabemos que Δs/L_k = 1/N:")
    print("   <T_k> ≈ (1 / 2N) * [T(s_0) + 2*Σ_{n=1}^{N-1} T(s_n) + T(s_N)]")
    print("=" * 80)

def print_theoretical_4_2_1_item_5() -> None:
    """Resposta teórica para o Item 4.2.1 (5)"""
    print("\n" + "=" * 80)
    print("RESPOSTA TEÓRICA - EXERCÍCIO 4.2.1(5)")
    print("=" * 80)
    print("Alternativa para o cálculo da viscosidade efetiva nas arestas:\n")
    print("Em vez de calcular a temperatura média <T> para depois aplicá-la na")
    print("fórmula da viscosidade -- ou seja, usar μ(<T>) --, o mais rigoroso é")
    print("integrar a PRÓPRIA VISCOSIDADE ao longo da aresta para achar <μ>:")
    print("\n   <μ_k> = (1/L_k) * Integral[0 -> L_k] μ(T(p(s))) ds")
    print("\nJustificativa Física e Matemática:")
    print("* A viscosidade da água μ(T) possui um comportamento fortemente NÃO-LINEAR.")
    print("* Em funções não-lineares, a função da média NÃO É IGUAL à média da função")
    print("  (ou seja, μ(<T>) ≠ <μ(T)>).")
    print("* Integrar diretamente a viscosidade ao longo da geometria capta com")
    print("  exatidão a real resistência ao escoamento distribuída pelo canal.")
    print("=" * 80)

def exercise_42_interpolation(model: HydroThermalCoupledModel) -> Dict[str, Dict[str, Any]]:
    """Item 4.2.1(2): interpolação 2D em malhas refinada e grosseira."""
    results: Dict[str, Dict[str, Any]] = {}
    for label, mesh, secondary_grid in [
        ("refinada_241x121", model.cfg["thermal_mesh_refined"], model.cfg["secondary_grid_refined_case"]),
        ("grosseira_61x31", model.cfg["thermal_mesh_coarse"], model.cfg["secondary_grid_coarse_case"]),
    ]:
        Nx, Ny = mesh
        print(f"\nEXERCÍCIO 4.2.1(2) - solução térmica base {label}")
        thermal = model.solve_thermal(Nx, Ny)
        results[label] = thermal
        print(f"Tmax = {thermal['Tmax']:.6f} °C | Tmean = {thermal['Tmean']:.6f} °C | tempo = {thermal['total_time']:.3f} s")

        # Campo original.
        save_contour_plot(thermal["x"], thermal["y"], thermal["T_grid"], f"Campo térmico base - {label}", f"ex-4-2-1_item-2_campo-termico-base_{label}.png", model.Xno, model.conec)

        # Grade secundária mais grosseira.
        xs = np.linspace(0.0, model.cfg["Lx"], secondary_grid[0])
        ys = np.linspace(0.0, model.cfg["Ly"], secondary_grid[1])
        XX, YY = np.meshgrid(xs, ys)
        pts = np.column_stack([XX.ravel(), YY.ravel()])

        for method in ["linear", "cubic", "nearest"]:
            interp = TemperatureInterpolator(thermal["x"], thermal["y"], thermal["T_grid"], method=method)
            Tsec = interp(pts).reshape((len(ys), len(xs)))
            save_contour_plot(xs, ys, Tsec, f"Interpolação {method} - {label}", f"ex-4-2-1_item-2_interpolacao-{method}_{label}.png", model.Xno, model.conec)

        # Temperatura nos nós da rede hidráulica e grafo colorido.
        interp_linear = TemperatureInterpolator(thermal["x"], thermal["y"], thermal["T_grid"], method="linear")
        T_nodes = interp_linear(model.Xno)
        save_graph_nodes_temperature(model.Xno, model.conec, T_nodes, f"Rede colorida por T nodal - {label}", f"ex-4-2-1_item-2_rede-temperatura-nodal_{label}.png")

    return results


def exercise_42_edge_means_and_hydraulics(model: HydroThermalCoupledModel, thermal_results: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Itens 4.2.1(3), 4.2.1(4) e 4.2.1(5)."""
    quad_rows: List[Dict[str, Any]] = []
    hydro_rows: List[Dict[str, Any]] = []

    print("\nEXERCÍCIO 4.2.1(3) - TEMPERATURA MÉDIA NAS ARESTAS")
    for mesh_label, thermal in thermal_results.items():
        interp = TemperatureInterpolator(thermal["x"], thermal["y"], thermal["T_grid"], method="linear")

        # Referência para estimativa de erro: trapézio com N=1000.
        T_ref = mean_edge_temperature(model.Xno, model.conec, interp, "trapezoid", 1000)

        for rule in ["midpoint", "trapezoid"]:
            for N in [1, 10, 100, 1000]:
                t0 = time.perf_counter()
                T_edge = mean_edge_temperature(model.Xno, model.conec, interp, rule, N)
                dt = time.perf_counter() - t0

                # Estimativa prática de erro: diferença contra uma quadratura mais refinada.
                err_inf = float(np.max(np.abs(T_edge - T_ref)))
                err_l2 = float(np.sqrt(np.mean((T_edge - T_ref) ** 2)))

                quad_rows.append({
                    "mesh": mesh_label,
                    "rule": rule,
                    "subdivisions": N,
                    "T_edge_min": float(np.min(T_edge)),
                    "T_edge_mean": float(np.mean(T_edge)),
                    "T_edge_max": float(np.max(T_edge)),
                    "erro_inf_vs_trap1000": err_inf,
                    "erro_l2_vs_trap1000": err_l2,
                    "time_s": dt,
                })

                # Item 4.2.1(4): resolve hidráulica atualizando condutância por mu(<T>).
                hydro = model.solve_hydraulic_from_edge_temperatures(T_edge)
                hydro_rows.append({
                    "mesh": mesh_label,
                    "rule": rule,
                    "subdivisions": N,
                    "pmax_Pa": hydro["pmax"],
                    "pmin_Pa": hydro["pmin"],
                    "power_W": hydro["power"],
                    "mu_edge_mean_Pa_s": float(np.mean(hydro["hydraulic_data"]["mu"])),
                })

                print(f"{mesh_label:18s} | {rule:9s} | N={N:4d} | <T>med={np.mean(T_edge):.4f} °C | erro∞={err_inf:.3e} | tempo={dt:.3f}s | pmax={hydro['pmax']:.3e} Pa | P={hydro['power']:.3e} W")

                # Salva um grafo de arestas para um caso representativo.
                if mesh_label.startswith("refinada") and rule == "trapezoid" and N == 100:
                    save_graph_edges_temperature(model.Xno, model.conec, T_edge, "Rede colorida por temperatura média nas arestas", "ex-4-2-1_item-3_rede-temperatura-media-arestas.png")
                    
                    # === CHAMA A FUNÇÃO PLOTAREDE DO PROFESSOR ===
                    if PlotaRede is not None:
                        p_calc = hydro["result"]["p"]
                        q_calc = hydro["result"]["q"]
                        fig, ax = PlotaRede(model.conec, model.Xno, p_calc, q_calc, factor_units=0.001)
                        ax.set_title(f"Pressão e Fluxo na Rede (Professor) - {mesh_label}")
                        fig.savefig(OUTPUT_DIR / f"ex-4-2-1_item-4_pressao-fluxo-rede_{mesh_label}.png", dpi=300, bbox_inches="tight")
                        plt.close(fig)

        # Item 5: viscosidade média direta <mu(T)>.
        mu_edge_direct = mean_edge_viscosity_direct(model.Xno, model.conec, interp, "trapezoid", 1000)
        hydro_direct = model.solve_hydraulic_from_edge_viscosities(mu_edge_direct)
        hydro_rows.append({
            "mesh": mesh_label,
            "rule": "direct_mu_trapezoid",
            "subdivisions": 1000,
            "pmax_Pa": hydro_direct["pmax"],
            "pmin_Pa": hydro_direct["pmin"],
            "power_W": hydro_direct["power"],
            "mu_edge_mean_Pa_s": float(np.mean(mu_edge_direct)),
        })
        print(f"{mesh_label:18s} | alternativa item 5: <mu(T)> direto | pmax={hydro_direct['pmax']:.3e} Pa | P={hydro_direct['power']:.3e} W")

    quad_df = pd.DataFrame(quad_rows)
    hydro_df = pd.DataFrame(hydro_rows)
    quad_df.to_csv(OUTPUT_DIR / "42_quadraturas_temperatura_media_arestas.csv", index=False)
    hydro_df.to_csv(OUTPUT_DIR / "42_hidraulica_acoplada_resultados.csv", index=False)
    return quad_df, hydro_df


# ============================================================
# EXERCÍCIOS 4.3.3 DO PDF
# ============================================================

def exercise_43_conductivity(model: HydroThermalCoupledModel) -> pd.DataFrame:
    """Item 4.3.3(1): efeito da rede na condutividade térmica."""
    rows: List[Dict[str, Any]] = []
    print("\nEXERCÍCIO 4.3.3(1) - CONDUTIVIDADE MODIFICADA PELA REDE")

    for mesh in model.cfg.get("thermal_meshes_43", [(61, 31), (121, 61)]):
        for dmax in model.cfg["dmax_values"]:
            influence = NetworkInfluenceModel(model.Xno, model.conec, dmax, k0=model.cfg["k_constant"])
            k_func = lambda x, y, infl=influence: infl.k_modified_at_point(x, y)

            t0 = time.perf_counter()
            result = model.solve_thermal(mesh[0], mesh[1], k_function=k_func)
            dt = time.perf_counter() - t0

            rows.append({
                "Nx": mesh[0],
                "Ny": mesh[1],
                "dmax": dmax,
                "Tmax_C": result["Tmax"],
                "Tmean_C": result["Tmean"],
                "time_s": dt,
            })
            print(f"malha={mesh} | dmax={dmax:.5g} | Tmax={result['Tmax']:.5f} °C | Tmean={result['Tmean']:.5f} °C | tempo={dt:.3f}s")

            # Para cada dmax, salva o mapa e perfis na malha principal refinada e na intermediária.
            if mesh in [(61, 31), (241, 121)]:
                suffix = f"{mesh[0]}x{mesh[1]}_dmax_{dmax:g}".replace(".", "p")
                save_contour_plot(result["x"], result["y"], result["T_grid"], f"k modificado - malha {mesh}, dmax={dmax}", f"ex-4-3-3_item-1_campo-termico-k-modificado_{suffix}.png", model.Xno, model.conec)
                save_profile_plots(result["x"], result["y"], result["T_grid"], f"Perfis - k modificado - dmax={dmax}", f"ex-4-3-3_item-1_perfis-temperatura-k-modificado_{suffix}.png")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "43_condutividade_modificada_resultados.csv", index=False)
    return df


def central_spine_intensity(Xno: np.ndarray, conec: np.ndarray) -> np.ndarray:
    """
    Define I_j = 100 nas arestas da espinha principal e I_j=0.1 nas demais.
    Como a rede foi centralizada em y=Ly/2, a espinha é aproximada pelas arestas
    cujo ponto médio tem y próximo de Ly/2 e que são predominantemente horizontais.
    """
    mid = 0.5 * (Xno[conec[:, 0]] + Xno[conec[:, 1]])
    vec = Xno[conec[:, 1]] - Xno[conec[:, 0]]
    y_center = 0.5 * CONFIG_INTEGRADO["Ly"]
    horizontal = np.abs(vec[:, 0]) >= np.abs(vec[:, 1])
    near_center = np.abs(mid[:, 1] - y_center) < 0.00075
    I = np.full(conec.shape[0], 0.1, dtype=float)
    I[horizontal & near_center] = 100.0
    return I


def exercise_43_source_sink(model: HydroThermalCoupledModel) -> pd.DataFrame:
    """Item 4.3.3(2): fonte/sumidouro gaussiano gerado pela rede."""
    rows: List[Dict[str, Any]] = []
    print("\nEXERCÍCIO 4.3.3(2) - TERMO FONTE/SUMIDOURO DA REDE")

    # O enunciado não fixa dmax aqui; usamos o valor intermediário do item anterior.
    dmax = 0.0005
    influence = NetworkInfluenceModel(model.Xno, model.conec, dmax, k0=model.cfg["k_constant"])
    intensity_cases = {
        "homogenea_I1": np.ones(model.conec.shape[0], dtype=float),
        "espinha_100_resto_0p1": central_spine_intensity(model.Xno, model.conec),
    }

    # Para não gerar dezenas de imagens repetidas, calcula todos os S0 nas duas distribuições
    # e salva mapas/perfis para os extremos ±1e6 e para +5e5.
    for intensity_label, I_edge in intensity_cases.items():
        for S0 in model.cfg["s0_values"]:
            def src_func(x: float, y: float, infl=influence, S0_=S0, I_=I_edge) -> float:
                return model.cfg["source_value"] + infl.source_extra_at_point(x, y, S0_, I_)

            t0 = time.perf_counter()
            result = model.solve_thermal(121, 61, source_function=src_func)
            dt = time.perf_counter() - t0

            rows.append({
                "intensity_case": intensity_label,
                "S0": S0,
                "dmax": dmax,
                "Nx": 121,
                "Ny": 61,
                "Tmax_C": result["Tmax"],
                "Tmean_C": result["Tmean"],
                "time_s": dt,
            })
            print(f"I={intensity_label:22s} | S0={S0: .1e} | Tmax={result['Tmax']:.5f} °C | Tmean={result['Tmean']:.5f} °C | tempo={dt:.3f}s")

            if S0 in (5e5, 1e6):
                suffix = f"{intensity_label}_S0_{S0:.0e}".replace("+", "").replace("-", "menos_").replace(".", "p")
                save_contour_plot(result["x"], result["y"], result["T_grid"], f"Fonte/sumidouro - {intensity_label}, S0={S0:.1e}", f"ex-4-3-3_item-2_campo-termico-fonte-sumidouro_{suffix}.png", model.Xno, model.conec)
                save_profile_plots(result["x"], result["y"], result["T_grid"], f"Perfis - {intensity_label}, S0={S0:.1e}", f"ex-4-3-3_item-2_perfis-temperatura-fonte-sumidouro_{suffix}.png")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "43_fonte_sumidouro_resultados.csv", index=False)
    return df


# ============================================================
# RELATÓRIO TEXTUAL GERADO JUNTO COM OS RESULTADOS
# ============================================================

def write_markdown_report(quad_df: pd.DataFrame, hydro_df: pd.DataFrame,
                          k_df: Optional[pd.DataFrame], source_df: Optional[pd.DataFrame]) -> None:
    """Cria um relatório curto com as principais respostas numéricas."""
    lines: List[str] = []
    lines.append("# Relatório de resultados - Acoplamento hidráulico-térmico\n")
    lines.append("## Regra do trapézio composta\n")
    lines.append("Para uma aresta de comprimento L dividida em N subintervalos, Δs=L/N:\n")
    lines.append("`integral ≈ Δs[0.5*T(s0) + Σ T(sn) + 0.5*T(sN)]`.\n")
    lines.append("Logo, a média na aresta é essa integral dividida por L.\n")

    lines.append("## Quadratura de temperatura média nas arestas\n")
    lines.append(quad_df.sort_values(["mesh", "rule", "subdivisions"]).to_markdown(index=False))
    lines.append("\n\n## Rede hidráulica acoplada\n")
    lines.append(hydro_df.sort_values(["mesh", "rule", "subdivisions"]).to_markdown(index=False))

    if k_df is not None:
        lines.append("\n\n## Condutividade térmica modificada pela rede\n")
        lines.append(k_df.to_markdown(index=False))

    if source_df is not None:
        lines.append("\n\n## Fonte/sumidouro da rede de microcanais\n")
        lines.append(source_df.to_markdown(index=False))

    lines.append("\n\n## Resposta conceitual do item 4.2.1(5)\n")
    lines.append("Em vez de calcular a temperatura média da aresta e depois usar μ(<T>), pode-se calcular diretamente a viscosidade média da aresta: <μ> = (1/L)∫μ(T(p(s)))ds. Como μ(T) é não linear, geralmente μ(<T>) não é igual a <μ(T)>.\n")

    (OUTPUT_DIR / "relatorio_resultados_acoplamento.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 100)
    print("ACOPLAMENTO HIDRÁULICO-TÉRMICO - EXECUÇÃO DOS EXERCÍCIOS DO PDF")
    print("=" * 100)
    print(f"Saídas serão salvas em: {OUTPUT_DIR}")

    model = HydroThermalCoupledModel(CONFIG_INTEGRADO)
    
    if CONFIG_INTEGRADO.get("run_ex_4-2-1_item-1"):
        print_theoretical_4_2_1_item_1()

    if CONFIG_INTEGRADO.get("run_ex_4-2-1_item_2"):
        thermal_results = exercise_42_interpolation(model)
        
    if CONFIG_INTEGRADO.get("run_ex_4-2-1_item_3-4-5"):
        # Garante que os dados térmicos existam caso o item 2 esteja 'False'
        if not CONFIG_INTEGRADO.get("run_ex_4-2-1_item_2"):
             thermal_results = exercise_42_interpolation(model) 
        quad_df, hydro_df = exercise_42_edge_means_and_hydraulics(model, thermal_results)
        # Exibe a resposta teórica do item 5 ao final deste bloco
        print_theoretical_4_2_1_item_5()

    if CONFIG_INTEGRADO.get("run_ex_4-3-3_item-1"):
        k_df = exercise_43_conductivity(model)

    if CONFIG_INTEGRADO.get("run_ex_4-3-3_item-2"):
        source_df = exercise_43_source_sink(model)

    print("\n" + "=" * 100)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 100)

if __name__ == "__main__":
    main()
