import numpy as np
import matplotlib.pyplot as plt

from gera_grafo import GeraGrafo
from plota_rede import PlotaRede


# ============================================================
# CONFIGURAÇÃO DO USUÁRIO
# ============================================================
#
# EDITE APENAS ESTA PARTE PARA TESTAR NOVOS CASOS
#
# Convenções:
# - pressão em Pa
# - vazão em m³/s
# - temperatura em °C
# - dimensões em m
# - vazão positiva  -> entra no nó
# - vazão negativa  -> sai do nó
#
# Para a rede funcionar, é necessário impor pelo menos uma pressão.
# ============================================================

CONFIG = {
    # ---------------------------
    # Geometria da rede
    # ---------------------------
    "levels": 1,              # aumenta ou diminui o tamanho da rede
    "coord_scale_to_m": 1e-3, # GeraGrafo fornece coordenadas em mm; aqui convertemos para m

    # ---------------------------
    # Modelo temporal
    # ---------------------------
    "time_mode": "transient",   # "steady" ou "transient"
    "t0": 0.0,
    "tf": 10.0,
    "dt": 0.2,

    # ---------------------------
    # Propriedades físicas dos canais
    # ---------------------------
    # geometry_mode:
    #   "area"        -> usa area_constant
    #   "diameter"    -> usa diameter_constant
    #   "rectangular" -> usa width e height
    "geometry_mode": "rectangular",

    "area_constant": 500e-6 * 500e-6,
    "diameter_constant": 800e-6,
    "width": 500e-6,
    "height": 500e-6,

    # Se quiser propriedades diferentes por canal, coloque aqui.
    # Ex.: [2.5e-7, 2.5e-7, 3.0e-7, ...]
    # Se None, o código usa o modo global acima.
    "area_per_edge": None,

    # ---------------------------
    # Temperatura e viscosidade
    # ---------------------------
    "temperature_celsius": 25.0,

    # Se quiser FORÇAR a viscosidade, coloque um valor aqui.
    # Se None, a viscosidade será calculada pela temperatura.
    "mu_override": None,

    # ---------------------------
    # Pressões prescritas
    # ---------------------------
    # Exemplo:
    # nó 5 em 0 Pa
    # nó 0 em 1500 Pa
    #
    # Você pode colocar quantos nós quiser.
    "pressure_bc": {
        5: 0.0,
        10: 0.1,
    },

    # ---------------------------
    # Vazões prescritas
    # ---------------------------
    # Cada nó pode receber:
    # - valor constante
    # - sinal senoidal
    # - sinal cossenoidal
    #
    # Formato:
    # "flow_bc": {
    #     0: {"type": "constant", "value": 1e-7},
    #     3: {"type": "sin", "mean": 1e-7, "amp": 2e-8, "freq": 0.5, "phase": 0.0},
    #     4: {"type": "cos", "mean": -2e-8, "amp": 1e-8, "freq": 0.25, "phase": 0.0},
    # }
    #
    "flow_bc": {
        0: {"type": "sin", "mean": 1.0e-7, "amp": 0.3e-7, "freq": 0.5, "phase": 0.0},
        3: {"type": "cos", "mean": -0.2e-7, "amp": 0.1e-7, "freq": 0.25, "phase": 0.0},
    },

    # ---------------------------
    # Saídas gráficas
    # ---------------------------
    "plot_network": True,
    "plot_time_series": True,

    # Escolha quais nós e arestas quer acompanhar no tempo
    "nodes_to_plot": [0, 2, 5],
    "edges_to_plot": [0, 1, 2],

    # Mostra tabela detalhada dos canais no terminal
    "print_edge_table": True,
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def water_viscosity_pa_s(T_celsius: float) -> float:
    """
    Viscosidade dinâmica aproximada da água [Pa.s]
    em função da temperatura [°C].
    """
    A = 2.414e-5
    B = 247.8
    C = 140.0
    return A * 10 ** (B / ((T_celsius + 273.15) - C))


def circular_area_from_diameter(diameter: float) -> float:
    if diameter <= 0:
        raise ValueError("O diâmetro deve ser positivo.")
    return np.pi * diameter**2 / 4.0


def rectangular_area(width: float, height: float) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("Largura e altura devem ser positivas.")
    return width * height


def equivalent_diameter_from_area(area: float) -> float:
    if area <= 0:
        raise ValueError("A área deve ser positiva.")
    return np.sqrt(4.0 * area / np.pi)


def edge_lengths(Xno: np.ndarray, conec: np.ndarray) -> np.ndarray:
    x1 = Xno[conec[:, 0], :]
    x2 = Xno[conec[:, 1], :]
    return np.linalg.norm(x2 - x1, axis=1)


def build_incidence_matrix(conec: np.ndarray, nv: int | None = None) -> np.ndarray:
    if nv is None:
        nv = int(np.max(conec)) + 1

    nc = conec.shape[0]
    D = np.zeros((nc, nv), dtype=float)

    for k in range(nc):
        i = int(conec[k, 0])
        j = int(conec[k, 1])
        D[k, i] = 1.0
        D[k, j] = -1.0

    return D


def assembly(conec: np.ndarray, C: np.ndarray) -> np.ndarray:
    nv = int(np.max(conec)) + 1
    nc = conec.shape[0]

    A = np.zeros((nv, nv), dtype=float)

    for k in range(nc):
        i = int(conec[k, 0])
        j = int(conec[k, 1])
        ck = float(C[k])

        A[i, i] += ck
        A[i, j] -= ck
        A[j, i] -= ck
        A[j, j] += ck

    return A


def get_time_vector(cfg: dict) -> np.ndarray:
    if cfg["time_mode"] == "steady":
        return np.array([cfg["t0"]], dtype=float)

    t0 = cfg["t0"]
    tf = cfg["tf"]
    dt = cfg["dt"]

    n = int(np.floor((tf - t0) / dt)) + 1
    return t0 + np.arange(n) * dt


def evaluate_flow_spec(spec: dict, t: float) -> float:
    flow_type = spec["type"].lower()

    if flow_type == "constant":
        return float(spec["value"])

    if flow_type == "sin":
        mean = float(spec["mean"])
        amp = float(spec["amp"])
        freq = float(spec["freq"])
        phase = float(spec.get("phase", 0.0))
        return mean + amp * np.sin(2.0 * np.pi * freq * t + phase)

    if flow_type == "cos":
        mean = float(spec["mean"])
        amp = float(spec["amp"])
        freq = float(spec["freq"])
        phase = float(spec.get("phase", 0.0))
        return mean + amp * np.cos(2.0 * np.pi * freq * t + phase)

    raise ValueError(f"Tipo de vazão inválido: {flow_type}")


def evaluate_flow_bc(flow_bc: dict, t: float, nv: int) -> np.ndarray:
    b = np.zeros(nv, dtype=float)

    for node, spec in flow_bc.items():
        if not (0 <= node < nv):
            raise ValueError(f"Nó {node} fora do intervalo válido.")
        b[node] += evaluate_flow_spec(spec, t)

    return b


def apply_pressure_bc(A: np.ndarray, b: np.ndarray, pressure_bc: dict) -> tuple[np.ndarray, np.ndarray]:
    if len(pressure_bc) == 0:
        raise ValueError("É necessário prescrever pelo menos uma pressão nodal.")

    A_mod = A.copy()
    b_mod = b.copy()

    for node, p_value in pressure_bc.items():
        if not (0 <= node < A.shape[0]):
            raise ValueError(f"Nó {node} fora do intervalo válido.")

        A_mod[node, :] = 0.0
        A_mod[node, node] = 1.0
        b_mod[node] = float(p_value)

    return A_mod, b_mod


def get_mu(cfg: dict) -> float:
    if cfg["mu_override"] is not None:
        mu = float(cfg["mu_override"])
        if mu <= 0:
            raise ValueError("A viscosidade deve ser positiva.")
        return mu

    return water_viscosity_pa_s(float(cfg["temperature_celsius"]))


def get_area_per_edge(conec: np.ndarray, cfg: dict) -> np.ndarray:
    nc = conec.shape[0]

    if cfg["area_per_edge"] is not None:
        area_edge = np.array(cfg["area_per_edge"], dtype=float)
        if len(area_edge) != nc:
            raise ValueError("area_per_edge deve ter o mesmo tamanho do número de arestas.")
        if np.any(area_edge <= 0):
            raise ValueError("Todas as áreas por aresta devem ser positivas.")
        return area_edge

    mode = cfg["geometry_mode"].lower()

    if mode == "area":
        area = float(cfg["area_constant"])
        if area <= 0:
            raise ValueError("A área constante deve ser positiva.")
        return np.full(nc, area, dtype=float)

    if mode == "diameter":
        diameter = float(cfg["diameter_constant"])
        area = circular_area_from_diameter(diameter)
        return np.full(nc, area, dtype=float)

    if mode == "rectangular":
        area = rectangular_area(float(cfg["width"]), float(cfg["height"]))
        return np.full(nc, area, dtype=float)

    raise ValueError("geometry_mode inválido. Use 'area', 'diameter' ou 'rectangular'.")


def hydraulic_conductivities(Xno: np.ndarray, conec: np.ndarray, cfg: dict) -> dict:
    mu = get_mu(cfg)
    L = edge_lengths(Xno, conec)

    if np.any(L <= 0):
        raise ValueError("Há canais com comprimento nulo ou negativo.")

    area_edge = get_area_per_edge(conec, cfg)
    D_eq = np.sqrt(4.0 * area_edge / np.pi)
    kappa = np.pi * D_eq**4 / (128.0 * mu)
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
    A = assembly(conec, C)
    nv = A.shape[0]

    b = evaluate_flow_bc(flow_bc, t, nv)
    A_mod, b_mod = apply_pressure_bc(A, b, pressure_bc)

    p = np.linalg.solve(A_mod, b_mod)

    D = build_incidence_matrix(conec, nv)
    K = np.diag(C)
    q = K @ (D @ p)
    dp = p[conec[:, 0]] - p[conec[:, 1]]

    return {
        "time": t,
        "A": A,
        "A_mod": A_mod,
        "b": b,
        "b_mod": b_mod,
        "p": p,
        "q": q,
        "dp_edge": dp,
        "D": D,
        "K": K,
    }


def compute_power(p: np.ndarray, D: np.ndarray, K: np.ndarray) -> float:
    return float(p.T @ (D.T @ K @ D) @ p)


def nodal_mass_residual(conec: np.ndarray, q: np.ndarray, imposed_b: np.ndarray) -> np.ndarray:
    nv = int(np.max(conec)) + 1
    residual = np.zeros(nv, dtype=float)

    for k, (i, j) in enumerate(conec):
        residual[i] += q[k]
        residual[j] -= q[k]

    residual -= imposed_b
    return residual


def print_inputs_summary(cfg: dict, hydraulic_data: dict, Xno: np.ndarray, conec: np.ndarray) -> None:
    print("=" * 88)
    print("ENTRADAS UTILIZADAS NA SIMULAÇÃO")
    print("=" * 88)
    print(f"Níveis da rede (levels): {cfg['levels']}")
    print(f"Número de nós gerados: {Xno.shape[0]}")
    print(f"Número de canais gerados: {conec.shape[0]}")
    print(f"Modo temporal: {cfg['time_mode']}")
    print(f"Intervalo de tempo: [{cfg['t0']}, {cfg['tf']}] s")
    print(f"Passo de tempo: {cfg['dt']} s")
    print()

    print("Propriedades físicas:")
    print(f"  geometry_mode: {cfg['geometry_mode']}")
    print(f"  temperature_celsius: {cfg['temperature_celsius']}")
    print(f"  mu_override: {cfg['mu_override']}")
    print(f"  viscosidade usada [Pa.s]: {hydraulic_data['mu']:.6e}")

    if cfg["geometry_mode"] == "area":
        print(f"  area_constant [m²]: {cfg['area_constant']:.6e}")
    elif cfg["geometry_mode"] == "diameter":
        print(f"  diameter_constant [m]: {cfg['diameter_constant']:.6e}")
    elif cfg["geometry_mode"] == "rectangular":
        print(f"  width [m]: {cfg['width']:.6e}")
        print(f"  height [m]: {cfg['height']:.6e}")

    print()
    print("Pressões prescritas:")
    if len(cfg["pressure_bc"]) == 0:
        print("  nenhuma")
    else:
        for node, value in cfg["pressure_bc"].items():
            print(f"  nó {node}: {value:.6e} Pa")

    print()
    print("Vazões prescritas:")
    if len(cfg["flow_bc"]) == 0:
        print("  nenhuma")
    else:
        for node, spec in cfg["flow_bc"].items():
            print(f"  nó {node}: {spec}")

    print("=" * 88)


def print_output_summary(cfg: dict, hydraulic_data: dict, result: dict, conec: np.ndarray) -> None:
    p = result["p"]
    q = result["q"]
    b = result["b"]
    dp = result["dp_edge"]
    D = result["D"]
    K = result["K"]

    lengths = hydraulic_data["lengths"]
    area_edge = hydraulic_data["area_edge"]
    diameter_eq = hydraulic_data["diameter_eq_edge"]
    C = hydraulic_data["conductance_edge"]

    power = compute_power(p, D, K)
    residual = nodal_mass_residual(conec, q, b)

    print()
    print("=" * 88)
    print("SAÍDAS DA SIMULAÇÃO")
    print("=" * 88)

    print("Pressões nodais [Pa]:")
    for i, pi in enumerate(p):
        print(f"  nó {i}: {pi:.6e}")

    print()
    print("Vazões impostas avaliadas no último instante [m³/s]:")
    for i, bi in enumerate(b):
        if abs(bi) > 0:
            print(f"  nó {i}: {bi:.6e}")

    if cfg["print_edge_table"]:
        print()
        print("Dados dos canais:")
        print("canal | i -> j | comprimento [m] | área [m²] | D_eq [m] | C [m³/(Pa.s)] | Δp [Pa] | q [m³/s]")
        for k, (i, j) in enumerate(conec):
            print(
                f"{k:5d} | {i:3d}->{j:<3d} | "
                f"{lengths[k]:.6e} | {area_edge[k]:.6e} | {diameter_eq[k]:.6e} | "
                f"{C[k]:.6e} | {dp[k]:.6e} | {q[k]:.6e}"
            )

    print()
    print(f"Potência hidráulica [W]: {power:.6e}")

    print()
    print("Resíduo de conservação de massa por nó:")
    for i, ri in enumerate(residual):
        print(f"  nó {i}: {ri:.6e}")

    print("=" * 88)


def print_final_explanation(cfg: dict, hydraulic_data: dict, result: dict) -> None:
    print()
    print("=" * 88)
    print("COMENTÁRIO FINAL AO USUÁRIO")
    print("=" * 88)

    print("O código acabou de:")
    print("1. Gerar a rede hidráulica a partir do parâmetro 'levels'.")
    print("2. Converter as coordenadas geométricas para metros.")
    print("3. Calcular os comprimentos dos canais.")
    print("4. Definir a viscosidade do fluido a partir da temperatura ou da viscosidade imposta.")
    print("5. Calcular a condutância hidráulica de cada canal.")
    print("6. Aplicar pressões prescritas e vazões prescritas nos nós.")
    print("7. Resolver o sistema linear para obter as pressões nodais.")
    print("8. Calcular as vazões em todos os canais e a potência hidráulica da rede.")
    print()

    print("Influência das entradas na saída:")
    print("- 'levels' altera o tamanho da rede, o número de nós e o número de canais.")
    print("- a geometria dos canais altera a área e o diâmetro equivalente, influenciando diretamente a condutância.")
    print("- canais com maior área ou diâmetro tendem a oferecer menor resistência ao escoamento.")
    print("- a temperatura altera a viscosidade; viscosidade maior tende a reduzir vazões para a mesma diferença de pressão.")
    print("- pressões prescritas fixam o nível hidráulico em nós específicos.")
    print("- vazões prescritas injetam ou retiram fluido dos nós, modificando o campo de pressões da rede.")
    print("- se a vazão varia no tempo, as pressões e vazões da rede também passam a variar no tempo.")
    print()

    if cfg["time_mode"] == "steady":
        print("Nesta execução, a análise foi estacionária: o sistema foi resolvido em um único instante.")
    else:
        print("Nesta execução, a análise foi quase-estática no tempo: o sistema foi resolvido em vários instantes.")
        print("Em cada instante, as condições de contorno de vazão foram reavaliadas e a rede foi resolvida novamente.")

    print()
    print(f"Viscosidade usada nesta execução [Pa.s]: {hydraulic_data['mu']:.6e}")
    print(f"Último instante resolvido [s]: {result['time']:.6f}")
    print("=" * 88)


def plot_time_series(results: list[dict], node_indices: list[int] | None = None, edge_indices: list[int] | None = None) -> None:
    times = np.array([r["time"] for r in results], dtype=float)
    p_hist = np.array([r["p"] for r in results], dtype=float)
    q_hist = np.array([r["q"] for r in results], dtype=float)

    if node_indices:
        plt.figure(figsize=(10, 5))
        for node in node_indices:
            if 0 <= node < p_hist.shape[1]:
                plt.plot(times, p_hist[:, node], label=f"p nó {node}")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Pressão [Pa]")
        plt.title("Histórico de pressões nodais")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if edge_indices:
        plt.figure(figsize=(10, 5))
        for edge in edge_indices:
            if 0 <= edge < q_hist.shape[1]:
                plt.plot(times, q_hist[:, edge], label=f"q aresta {edge}")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Vazão [m³/s]")
        plt.title("Histórico de vazões nos canais")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    # 1) Gera a rede
    Xno, conec = GeraGrafo(levels=CONFIG["levels"])
    Xno = Xno * CONFIG["coord_scale_to_m"]

    # 2) Propriedades hidráulicas
    hydraulic_data = hydraulic_conductivities(Xno, conec, CONFIG)
    C = hydraulic_data["conductance_edge"]

    # 3) Vetor de tempos
    times = get_time_vector(CONFIG)

    # 4) Resolve a rede ao longo do tempo
    results = []
    for t in times:
        result_t = solve_network(
            conec=conec,
            C=C,
            pressure_bc=CONFIG["pressure_bc"],
            flow_bc=CONFIG["flow_bc"],
            t=float(t),
        )
        results.append(result_t)

    # 5) Último estado
    last_result = results[-1]

    # 6) Resumos impressos
    print_inputs_summary(CONFIG, hydraulic_data, Xno, conec)
    print_output_summary(CONFIG, hydraulic_data, last_result, conec)
    print_final_explanation(CONFIG, hydraulic_data, last_result)

    # 7) Plot da rede no último instante
    if CONFIG["plot_network"]:
        p_last = last_result["p"]
        q_last = last_result["q"]
        fig, ax = PlotaRede(conec, Xno, p_last, q_last, factor_units=CONFIG["coord_scale_to_m"])
        plt.title(f"Rede hidráulica - levels = {CONFIG['levels']} - t = {last_result['time']:.3f} s")
        plt.show()

    # 8) Séries temporais
    if CONFIG["plot_time_series"] and len(results) > 1:
        plot_time_series(
            results,
            node_indices=CONFIG["nodes_to_plot"],
            edge_indices=CONFIG["edges_to_plot"],
        )


if __name__ == "__main__":
    main()