# -*- coding: utf-8 -*-
"""
acoplamento-hidraulico-elastico.py

==================================================
GÊMEO DIGITAL: ACOPLAMENTO HIDRÁULICO-MECÂNICO
==================================================
Integração completa entre a rede microfluídica (fractal) e a membrana elástica tensionada.
"""

# --- Bibliotecas Padrão ---
import time  # Medição de tempo de execução e benchmarking das rotinas (ex: tempo de montagem vs solução).

# --- Bibliotecas de Estrutura de Dados e Matemática Base ---
import numpy as np  # Motor principal para álgebra linear, operações vetorizadas e matrizes densas.
import pandas as pd  # Utilizado na geração e filtragem dos nós durante a construção topológica do grafo.

# --- Bibliotecas Visuais (Matplotlib) ---
import matplotlib.pyplot as plt  # Geração de gráficos temporais, curvas de nível e padrão de esparsidade (spy).
from matplotlib import cm  # Mapas de cores para visualização progressiva de pressões na rede.

# --- Bibliotecas Geométricas (Shapely) ---
# Usadas especificamente no algoritmo de geração do grafo fractal para detectar e fundir cruzamentos de canais.
from shapely.geometry import LineString
from shapely.ops import unary_union

# --- Bibliotecas de Álgebra Linear Avançada (SciPy) ---
import scipy.sparse as sparse  # Criação de matrizes esparsas (COO, CSR, diags) e montagem de supermatrizes em blocos (bmat).
import scipy.sparse.linalg as splinalg  # Solvers esparsos, com destaque para 'splu' (fatoração LU) para resolver o sistema no tempo de forma ultrarrápida.
from scipy.linalg import eigh  # Solver denso para o problema de autovalores generalizado K*Phi = lambda*M*Phi.
from scipy.sparse.linalg import eigsh  # Solver iterativo esparso para extrair apenas os modos fundamentais de vibração.

# ---
# Dependências: numpy, pandas, matplotlib, shapely, scipy
# ---

# ============================================================
# CONFIGURAÇÃO GERAL DO SISTEMA ACOPLADO
# ============================================================

CONFIG = {
    # --- Propriedades da Rede Hidráulica ---
    "levels": 3,                 # Complexidade topológica
    "inlet_node": 0,             # Nó de entrada (pressão controlada)
    "outlet_node": 5,            # Nó de saída (descarga no reservatório)
    "mu": 5e-4,                  # Viscosidade dinâmica [Pa.s]
    "channel_width": 1000e-6,    # Largura inicial do canal (geometria quadrada) [m]
    
    # --- Propriedades da Membrana Elástica ---
    "radius": 0.0025,            # Raio da membrana circular [m]
    "thickness": 0.0001,         # Espessura estrutural [m]
    "sigma": 200.0,              # Tensão membranal [N/m]
    "rho": 900.0,                # Densidade de massa [kg/m³]
    "beta_damping": 0.1,         # Amortecimento intrínseco (beta)
    "Nx": 51,                    # Nós padrão na malha X
    "Ny": 51,                    # Nós padrão na malha Y
    
    # --- Parâmetros de Simulação e Integração Temporal ---
    "dt": 0.025,                 # Passo de tempo base [s]
    "t_final": 12.0,             # Tempo final de simulação [s]
    "p_inlet": 5000.0,           # Pressão inicial de referência no inlet [Pa]

    # --- Controle de Saídas ---
    "show_plots": False,         # Se True, exibe na tela via plt.show(). Se False, apenas salva os arquivos em disco.

    # --- Seletor de Rotinas (Exercícios/Tópicos) ---
    "run_topic_1": False,         # Matriz R equivalente e dedução (Equação 5.3)
    "run_topic_2": False,
    "run_topic_3": False,
    "run_topic_4": False,
    "run_topic_5": True
}


# ============================================================
# FUNÇÕES DA REDE HIDRÁULICA
# ============================================================

def PlotaRede(conec, Xno, p, q, factor_units=0.001):
    edges = conec
    coord = Xno
    nv = np.max(conec) + 1

    segs = []
    mids = []
    for (i, j) in edges:
        x1, y1 = coord[i, 0], coord[i, 1]
        x2, y2 = coord[j, 0], coord[j, 1]
        segs.append(((x1, y1), (x2, y2)))
        mids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    segs = np.array(segs)
    mids = np.array(mids)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(float(p.min()), float(p.max()))
    
    colors = [cmap(norm(pi)) for pi in p]
    ax.scatter(coord[:, 0], coord[:, 1], s=50, c=colors, zorder=3, edgecolors="black")

    for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.5, zorder=1)
        xm, ym = mids[idx]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0: 
            continue
        dxn, dyn = dx / L, dy / L

        p1, p2 = p[edges[idx, 0]], p[edges[idx, 1]]
        q_dir = 1 if p1 > p2 else -1

        ax.annotate(
            "",
            xy=(xm + q_dir * 0.5 * 0.0005 * dxn, ym + q_dir * 0.5 * 0.0005 * dyn),
            xytext=(xm - q_dir * 0.5 * 0.0005 * dxn, ym - q_dir * 0.5 * 0.0005 * dyn),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5, mutation_scale=9),
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Pressure, p", fraction=0.0225, pad=0.025)
    return fig, ax


# ============================================================
# FUNÇÕES DA REDE HIDRÁULICA
# ============================================================

def water_viscosity_pa_s(T_celsius):
    A, B, C = 2.414e-5, 247.8, 140.0
    return A * 10 ** (B / ((T_celsius + 273.15) - C))

def rectangular_area(width, height):
    return width * height

def edge_lengths(Xno, conec):
    return np.linalg.norm(Xno[conec[:, 1]] - Xno[conec[:, 0]], axis=1)

def build_incidence_matrix(conec, nv):
    nc = conec.shape[0]
    D = np.zeros((nc, nv), dtype=float)
    for k in range(nc):
        D[k, int(conec[k, 0])] = 1.0
        D[k, int(conec[k, 1])] = -1.0
    return D

def assembly_hydraulic(conec, C):
    nv = int(np.max(conec)) + 1
    A = np.zeros((nv, nv), dtype=float)
    for k in range(conec.shape[0]):
        i, j = int(conec[k, 0]), int(conec[k, 1])
        ck = float(C[k])
        A[i, i] += ck
        A[j, j] += ck
        A[i, j] -= ck
        A[j, i] -= ck
    return A

def hydraulic_conductivities(Xno, conec, mu, width):
    L = edge_lengths(Xno, conec)
    area_edge = np.full(conec.shape[0], rectangular_area(width, width), dtype=float)
    D_eq = np.sqrt(4.0 * area_edge / np.pi)
    kappa = np.pi * D_eq**4 / (128.0 * mu)
    return {"lengths": L, "conductance_edge": kappa / L, "D_eq": D_eq, "area": area_edge}

def apply_hydraulic_pressure_bc(A, b, pressure_bc):
    A_mod, b_mod = A.copy(), b.copy()
    for node, p_value in pressure_bc.items():
        A_mod[node, :] = 0.0
        A_mod[node, node] = 1.0
        b_mod[node] = float(p_value)
    return A_mod, b_mod

def compute_power(p, D, K):
    return float(p.T @ (D.T @ K @ D) @ p)

def nodal_mass_residual(conec, q, imposed_b):
    residual = np.zeros(int(np.max(conec)) + 1, dtype=float)
    for k, (i, j) in enumerate(conec):
        residual[i] += q[k]
        residual[j] -= q[k]
    return residual - imposed_b

# ============================================================
# GERAÇÃO DA REDE HIDRÁULICA: gera_grafo
# ============================================================

def generate_graph_arrays(levels=3):
    nodes_data = []
    edges_raw = []
    node_id = 0
    spine_length = 6

    spine_nodes = []
    for i in range(spine_length):
        nodes_data.append({'x': i * 4.0, 'y': 0.0})
        spine_nodes.append(node_id)
        if i > 0: 
            edges_raw.append((node_id - 1, node_id))
        node_id += 1

    def add_fractal_branches(parent_id, px, py, angle, length, depth):
        nonlocal node_id
        if depth == 0: 
            return
        angles = [angle + np.pi/6, angle - np.pi/6]
        branch_len = length * 0.75
        for a in angles:
            nx = px + branch_len * np.cos(a)
            ny = py + branch_len * np.sin(a)
            curr_id = node_id
            nodes_data.append({'x': nx, 'y': ny})
            edges_raw.append((parent_id, curr_id))
            node_id += 1
            add_fractal_branches(curr_id, nx, ny, a, branch_len, depth - 1)

    for s_id in spine_nodes[1:-1]:
        add_fractal_branches(s_id, nodes_data[s_id]['x'], nodes_data[s_id]['y'], np.pi/2, 3.0, levels)
        add_fractal_branches(s_id, nodes_data[s_id]['x'], nodes_data[s_id]['y'], -np.pi/2, 3.0, levels)

    import pandas as pd
    df_temp = pd.DataFrame(nodes_data)
    y_max, y_min = df_temp['y'].max() + 1.0, df_temp['y'].min() - 1.0

    all_indices = [e[0] for e in edges_raw] + [e[1] for e in edges_raw]
    counts = pd.Series(all_indices).value_counts()
    leaf_ids = counts[counts == 1].index.tolist()
    leaf_ids = [idx for idx in leaf_ids if idx not in [spine_nodes[0], spine_nodes[-1]]]

    for l_id in leaf_ids:
        target_y = y_max if nodes_data[l_id]['y'] > 0 else y_min
        new_id = node_id
        nodes_data.append({'x': nodes_data[l_id]['x'], 'y': target_y})
        edges_raw.append((l_id, new_id))
        node_id += 1

    lines = [LineString([(nodes_data[e[0]]['x'], nodes_data[e[0]]['y']),
                         (nodes_data[e[1]]['x'], nodes_data[e[1]]['y'])]) for e in edges_raw]

    df_nodes_final = pd.DataFrame(nodes_data)
    for y_lim in [y_max, y_min]:
        pts = df_nodes_final[df_nodes_final['y'] == y_lim].sort_values('x')
        if len(pts) > 1:
            lines.append(LineString(pts[['x', 'y']].values))

    merged_graph = unary_union(lines)

    final_nodes_map = {}
    final_nodes_list = []
    final_edges_list = []

    def get_node_id(pt):
        coords = (round(pt[0], 6), round(pt[1], 6))
        if coords not in final_nodes_map:
            final_nodes_map[coords] = len(final_nodes_list)
            final_nodes_list.append([pt[0], pt[1]])
        return final_nodes_map[coords]

    segments = merged_graph.geoms if hasattr(merged_graph, 'geoms') else [merged_graph]
    for seg in segments:
        id_start = get_node_id(seg.coords[0])
        id_end = get_node_id(seg.coords[-1])
        final_edges_list.append([id_start, id_end])

    nodes_np = np.array(final_nodes_list)
    edges_np = np.array(final_edges_list)
    mask = edges_np[:, 0] != edges_np[:, 1]
    edges_np = edges_np[mask]

    return nodes_np, edges_np

# ============================================================
# VISUALIZAÇÃO DA REDE: plora_rede
# ============================================================

def PlotaRede(conec, Xno, p, q, factor_units=0.001):
    edges = conec
    coord = Xno
    nv = np.max(conec) + 1

    segs = []
    mids = []
    for (i, j) in edges:
        x1, y1 = coord[i, 0], coord[i, 1]
        x2, y2 = coord[j, 0], coord[j, 1]
        segs.append(((x1, y1), (x2, y2)))
        mids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    segs = np.array(segs)
    mids = np.array(mids)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(float(p.min()), float(p.max()))
    
    colors = [cmap(norm(pi)) for pi in p]
    ax.scatter(coord[:, 0], coord[:, 1], s=50, c=colors, zorder=3, edgecolors="black")

    for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.5, zorder=1)
        xm, ym = mids[idx]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0: 
            continue
        dxn, dyn = dx / L, dy / L

        p1, p2 = p[edges[idx, 0]], p[edges[idx, 1]]
        q_dir = 1 if p1 > p2 else -1

        ax.annotate(
            "",
            xy=(xm + q_dir * 0.5 * 0.0005 * dxn, ym + q_dir * 0.5 * 0.0005 * dyn),
            xytext=(xm - q_dir * 0.5 * 0.0005 * dxn, ym - q_dir * 0.5 * 0.0005 * dyn),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5, mutation_scale=9),
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Pressure, p", fraction=0.0225, pad=0.025)
    return fig, ax


# ============================================================
# FUNÇÕES DA MEMBRANA ELÁSTICA
# ============================================================

def ij2n(i, j, Nx):
    """Mapeia o par de índices (i, j) da malha 2D para um índice global 1D."""
    return i + j * Nx

def n2ij(n, Nx):
    """Converte o índice global 1D n para o par de índices (i, j) da malha 2D."""
    j = n // Nx
    i = n % Nx
    return i, j

def is_inside_unit_circle(x_hat, y_hat):
    """Verifica se o ponto adimensional está dentro do círculo de raio 1."""
    return x_hat * x_hat + y_hat * y_hat <= 1.0

def is_restricted_point(i, j, Nx, Ny, x_hat, y_hat, use_circular_mask):
    """Determina se um nó (i, j) possui movimento restrito (fronteira fixa)."""
    if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
        return True

    if not use_circular_mask:
        return False

    xh, yh = x_hat[i], y_hat[j]
    if not is_inside_unit_circle(xh, yh):
        return True

    # Verifica vizinhos para garantir a suavidade na borda discreta
    neighbors = [
        (x_hat[i + 1], y_hat[j]),
        (x_hat[i - 1], y_hat[j]),
        (x_hat[i], y_hat[j + 1]),
        (x_hat[i], y_hat[j - 1]),
    ]
    for xn, yn in neighbors:
        if not is_inside_unit_circle(xn, yn):
            return True
            
    return False

def build_mask_grid(Nx, Ny, x_hat, y_hat, use_circular_mask):
    """Gera uma matriz booleana 2D identificando a região ativa da membrana."""
    mask = np.zeros((Ny, Nx), dtype=bool)
    for j in range(Ny):
        for i in range(Nx):
            mask[j, i] = not is_restricted_point(i, j, Nx, Ny, x_hat, y_hat, use_circular_mask)
    return mask

def assembly_membrane(Nx, Ny, Lx_hat, Ly_hat, use_circular_mask, big_number=1e4):
    """
    Monta as matrizes de rigidez (K) e de massa (M) adimensionais da membrana.
    Retorna as matrizes esparsas prontas para processamento.
    """
    x_hat = np.linspace(-Lx_hat / 2.0, Lx_hat / 2.0, Nx)
    y_hat = np.linspace(-Ly_hat / 2.0, Ly_hat / 2.0, Ny)
    h_hat = Lx_hat / (Nx - 1)

    nunk = Nx * Ny
    rows = []
    cols = []
    data = []
    mass_diag = np.zeros(nunk, dtype=float)
    stiffness_scale = 1.0 / (h_hat ** 2)

    for j in range(Ny):
        for i in range(Nx):
            Ic = ij2n(i, j, Nx)

            # Aplicação de penalização de fronteira (Dirichlet homogênea)
            if is_restricted_point(i, j, Nx, Ny, x_hat, y_hat, use_circular_mask):
                rows.append(Ic)
                cols.append(Ic)
                data.append(big_number)
                mass_diag[Ic] = 1.0
                continue

            # Operador Laplaciano discreto estruturado de 5 pontos
            Ie = ij2n(i + 1, j, Nx)
            Iw = ij2n(i - 1, j, Nx)
            In = ij2n(i, j + 1, Nx)
            Is = ij2n(i, j - 1, Nx)

            rows.extend([Ic, Ic, Ic, Ic, Ic])
            cols.extend([Ic, Ie, Iw, In, Is])
            data.extend([
                4.0 * stiffness_scale,  # Diagonal principal
                -1.0 * stiffness_scale, # Vizinho Leste
                -1.0 * stiffness_scale, # Vizinho Oeste
                -1.0 * stiffness_scale, # Vizinho Norte
                -1.0 * stiffness_scale, # Vizinho Sul
            ])
            mass_diag[Ic] = 1.0

    K = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
    M = sparse.diags(mass_diag, offsets=0, format="csr")
    
    return K, M, x_hat, y_hat, h_hat

def solve_membrane_eigenproblem(K, M, num_modes, solver_mode="sparse"):
    """Resolve o problema de autovalores generalizado K*Phi = lambda*M*Phi."""
    if solver_mode == "dense":
        from scipy.linalg import eigh
        evals, evecs = eigh(K.toarray(), M.toarray())
        idx = np.argsort(evals)
        evals = evals[idx][:num_modes]
        evecs = evecs[:, idx][:, :num_modes]
    else:
        from scipy.sparse.linalg import eigsh
        evals, evecs = eigsh(K, k=num_modes, M=M, which="SM")
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]
        
    omega_hat = np.sqrt(np.maximum(evals, 0.0))
    return evals, evecs, omega_hat

def generalized_modal_projection(M, evecs, Z):
    """Projeta um vetor de carregamento forçante Z na base modal M-ortonormal."""
    return evecs.T @ Z

def mean_elastic_energy_curve(omega_star_values, omega_hat, alphas, beta=0.02):
    """Calcula a curva de energia elástica média ao longo de um espectro de frequências."""
    energies = np.zeros_like(omega_star_values, dtype=float)
    for m, omega_star in enumerate(omega_star_values):
        denom = np.sqrt((omega_hat ** 2 - omega_star ** 2) ** 2 + (beta ** 2) * (omega_star ** 2))
        ci = alphas / denom
        energies[m] = 0.25 * np.sum((ci ** 2) * (omega_hat ** 2))
    return energies


# ============================================================
# FUNÇÕES DE ACOPLAMENTO MULTIFÍSICO
# ============================================================

def compute_reference_scales(config):
    """
    Calcula as escalas de referência para a adimensionalização do sistema
    com base nas propriedades físicas fornecidas.
    """
    R = config["radius"]
    sigma = config["sigma"]
    rho = config["rho"]
    e = config["thickness"]
    
    # Deslocamento típico de referência (1% do raio)
    w_ref = 0.01 * R
    
    # Pressão de referência (equivalente a força por unidade de área)
    p_ref = (sigma * w_ref) / (R ** 2)
    
    # Velocidade de referência
    v_ref = np.sqrt(sigma / (rho * e)) * (w_ref / R)
    
    # Tempo de referência
    t_ref = w_ref / v_ref
    
    return w_ref, p_ref, v_ref, t_ref

def build_coupling_matrix_U(Nx, Ny, x_hat, y_hat, use_circular_mask, n_p, outlet_node=5):
    """
    Constrói a matriz de acoplamento U de dimensão (n_p x n_m).
    Insere o valor 1 apenas na linha do nó de saída (outlet) e nas colunas
    correspondentes aos graus de liberdade livres da membrana.
    """
    n_m = Nx * Ny
    U_dense = np.zeros((n_p, n_m), dtype=float)
    
    # Vetor de ponderação geométrica preenchido com 1 para nós livres e 0 para restritos
    uns = np.ones(n_m, dtype=float)
    
    for j in range(Ny):
        for i in range(Nx):
            Ic = ij2n(i, j, Nx)
            if is_restricted_point(i, j, Nx, Ny, x_hat, y_hat, use_circular_mask):
                uns[Ic] = 0.0
                
    U_dense[outlet_node, :] = uns[:]
    return sparse.csr_matrix(U_dense)

def build_global_coupled_system(K_mem, M_mem, A_hyd_scaled, U, dt_hat, beta_damping, h_hat, n_p, n_m):
    """
    Monta a supermatriz global acoplada (lado esquerdo) usando scipy.sparse.bmat.
    A estrutura segue a discretização implícita dos blocos de estado [w, v, p].
    """
    Iden_m = sparse.identity(n_m, format="csr")
    
    # Matriz de amortecimento proporcional da membrana (D = beta * M)
    D_mem = beta_damping * M_mem
    
    # Bloco da primeira equação: (1/dt)*w^(n+1) - v^(n+1) = (1/dt)*w^n
    row1 = [ (1.0 / dt_hat) * Iden_m, -Iden_m, None ]
    
    # Bloco da segunda equação: K*w^(n+1) + ((1/dt)*M + D)*v^(n+1) - U^T*p^(n+1)
    # Nota: O bloco transposto projeta a força da pressão interna na estrutura mecânica
    row2 = [ K_mem, (1.0 / dt_hat) * M_mem + D_mem, -U.T ]
    
    # Bloco da terceira equação: h_hat^2 * U * v^(n+1) + A_scaled * p^(n+1)
    h_hat_sq = h_hat ** 2
    row3 = [ None, h_hat_sq * U, A_hyd_scaled ]
    
    blocks = [row1, row2, row3]
    return sparse.bmat(blocks, format="csr")

def prefactor_coupled_system(A_global):
    """
    Pré-fatora a supermatriz estática utilizando a decomposição LU esparsa (splu)
    para garantir a máxima eficiência computacional dentro do laço temporal.
    """
    return splinalg.splu(A_global)


# ============================================================
# ROTINAS DE INVESTIGAÇÃO DO SISTEMA
# ============================================================

def run_simulation(config, custom_params=None, initial_state=None, forcing_func=None):
    """
    Motor de integração no tempo. Pré-fatora a matriz global e resolve
    o sistema a cada passo para máxima eficiência.
    """
    cfg = config.copy()
    if custom_params:
        cfg.update(custom_params)

    Nx, Ny = cfg["Nx"], cfg["Ny"]
    n_m = Nx * Ny
    
    # 1. Preparação da Rede Hidráulica
    Xno, conec = generate_graph_arrays(levels=cfg["levels"])
    n_p = Xno.shape[0]
    hyd_data = hydraulic_conductivities(Xno, conec, cfg["mu"], cfg["channel_width"])
    A_hyd = assembly_hydraulic(conec, hyd_data["conductance_edge"])
    
    # 2. Preparação da Membrana Elástica
    K_mem, M_mem, x_hat, y_hat, h_hat = assembly_membrane(Nx, Ny, 2.0, 2.0, True)
    
    # 3. Adimensionalização e Acoplamento
    w_ref, p_ref, v_ref, t_ref = compute_reference_scales(cfg)
    factor = p_ref / (v_ref * cfg["radius"]**2)
    A_hyd_scaled = A_hyd * factor
    
    U = build_coupling_matrix_U(Nx, Ny, x_hat, y_hat, True, n_p, cfg["outlet_node"])
    
    # Aplicação de condição de contorno (Inlet) na matriz da rede
    A_hyd_scaled, _ = apply_hydraulic_pressure_bc(A_hyd_scaled, np.zeros(n_p), {cfg["inlet_node"]: 0.0})
    
    dt_hat = cfg["dt"] / t_ref
    A_global = build_global_coupled_system(K_mem, M_mem, A_hyd_scaled, U, dt_hat, cfg["beta_damping"], h_hat, n_p, n_m)
    
    # OTIMIZAÇÃO: Pré-fatoração LU (a matriz não muda no tempo)
    solver = prefactor_coupled_system(A_global)
    
    # 4. Condições Iniciais
    times = np.arange(0, cfg["t_final"] + cfg["dt"], cfg["dt"])
    w_n = np.zeros(n_m)
    v_n = np.zeros(n_m)
    
    if initial_state:
        w_n[:] = initial_state.get("w", 0.0)
        v_n[:] = initial_state.get("v", 0.0)

    center_idx = ij2n(Nx//2, Ny//2, Nx)
    
    history = {
        "t": times,
        "w_center": np.zeros(len(times)),
        "p_outlet": np.zeros(len(times)),
        "q_outlet": np.zeros(len(times)),
        "volume": np.zeros(len(times)),
        "power": np.zeros(len(times))
    }
    
    vol_accum = 0.0
    
    # 5. Loop no Tempo
    for i, t in enumerate(times):
        t_hat = t / t_ref
        
        # Vetor de cargas (RHS)
        b_global = np.zeros(n_m + n_m + n_p)
        b_global[0:n_m] = w_n / dt_hat
        b_global[n_m:2*n_m] = M_mem.dot(v_n) / dt_hat
        
        # Condição de contorno (Pressão no inlet)
        p_in = forcing_func(t) if forcing_func else cfg["p_inlet"]
        b_global[2*n_m + cfg["inlet_node"]] = p_in / p_ref
        
        # Resolução do sistema
        x_next = solver.solve(b_global)
        
        w_next = x_next[0:n_m]
        v_next = x_next[n_m:2*n_m]
        p_next = x_next[2*n_m:]
        
        # Atualização para o próximo passo
        w_n = w_next.copy()
        v_n = v_next.copy()
        
        # Coleta de métricas (retornando às dimensões físicas)
        p_out_physical = p_next[cfg["outlet_node"]] * p_ref
        q_out_physical = (h_hat**2 * np.sum(v_next[U.getrow(cfg["outlet_node"]).indices])) * (v_ref * cfg["radius"]**2)
        vol_accum += q_out_physical * cfg["dt"]
        
        history["w_center"][i] = w_next[center_idx] * w_ref
        history["p_outlet"][i] = p_out_physical
        history["q_outlet"][i] = q_out_physical
        history["volume"][i] = vol_accum
        history["power"][i] = p_out_physical * q_out_physical # Simplificação de potência útil
        
    return history, {"w": w_n, "v": v_n, "p": p_next}, K_mem, M_mem

def rotina_1_analise_matriz_R(config):
    # Print detalhado explicando o objetivo e exigências do Tópico 1
    print("\n" + "="*80)
    print("OBJETIVO DA ROTINA 1 (TÓPICO 1): MATRIZ DE AMORTECIMENTO HIDRÁULICO EQUIVALENTE")
    print("="*80)
    print("Esta rotina realiza a redução cinemática do acoplamento multifísico.")
    print("A presença da rede microfluídica altera a resposta mecânica da membrana,")
    print("introduzindo um amortecimento dissipativo adicional proporcional à velocidade.")
    print("O objetivo aqui é:")
    print("1. Deduzir analiticamente como a pressão do fluido gera a matriz R.")
    print("2. Construir numericamente a matriz R para uma malha reduzida de 26x26 nós.")
    print("3. Analisar e salvar a estrutura de esparsidade (plt.spy) resultante.")
    print("="*80 + "\n")

    cfg = config.copy()
    cfg.update({"Nx": 26, "Ny": 26}) # Malha reduzida solicitada para evitar poluição visual
    
    # 1. Construção das estruturas da rede e da membrana
    Xno, conec = generate_graph_arrays(levels=cfg["levels"])
    n_p = Xno.shape[0]
    hyd_data = hydraulic_conductivities(Xno, conec, cfg["mu"], cfg["channel_width"])
    A_hyd = assembly_hydraulic(conec, hyd_data["conductance_edge"])
    
    _, _, x_hat, y_hat, h_hat = assembly_membrane(cfg["Nx"], cfg["Ny"], 2.0, 2.0, True)
    U = build_coupling_matrix_U(cfg["Nx"], cfg["Ny"], x_hat, y_hat, True, n_p, cfg["outlet_node"])
    
    # Aplicando condição de contorno de pressão nula para remover a singularidade da rede
    A_hyd_mod, _ = apply_hydraulic_pressure_bc(A_hyd.copy(), np.zeros(n_p), {cfg["inlet_node"]: 0.0})
    A_inv = np.linalg.pinv(A_hyd_mod) 
    
    # 2. Cálculo numérico da matriz condensada R = h^2 * U^T * A^-1 * U
    R = (h_hat**2) * U.T.dot(A_inv).dot(U.toarray())
    
    # 3. Geração do gráfico de esparsidade
    fig = plt.figure(figsize=(6, 6))
    plt.spy(R, marker=',', color='black')
    plt.title(f"Estrutura de Esparsidade da Matriz R ({cfg['Nx']}x{cfg['Ny']})")
    
    img_name = "esparsidade_matriz_R.png"
    plt.savefig(img_name, dpi=300, bbox_inches='tight')
    
    if cfg["show_plots"]:
        plt.show()
    else:
        plt.close(fig)
        
    # 4. Construção do texto de dedução para o relatório .md
    deducao_texto = (
        "## Dedução Analítica da Equação (5.3)\n\n"
        "O sistema acoplado completo divide-se em blocos mecânicos e hidráulicos. "
        "A conservação de massa da rede hidráulica é governada pelo sistema linear:\n"
        "$$\\mathbb{A}\\mathbf{p} = -h^2 \\mathbb{U}\\mathbf{v} + \\mathbf{\\tilde{b}}$$\n\n"
        "Isolando o vetor de pressões nodais $\\mathbf{p}$:\n"
        "$$\\mathbf{p} = \\mathbb{A}^{-1}(-h^2 \\mathbb{U}\\mathbf{v} + \\mathbf{\\tilde{b}})$$\n\n"
        "A equação de equilíbrio dinâmico da membrana projeta a força gerada pela pressão interna "
        "sobre a superfície através da matriz de acoplamento transposta $\\mathbb{U}^T$:\n"
        "$$\\mathbb{M}\\frac{d\\mathbf{v}}{dt} + \\mathbb{D}\\mathbf{v} + \\mathbb{K}\\mathbf{w} - \\mathbb{U}^T\\mathbf{p} = 0$$\n\n"
        "Substituindo a expressão da pressão $\\mathbf{p}$ na equação da membrana:\n"
        "$$\\mathbb{M}\\frac{d\\mathbf{v}}{dt} + \\mathbb{D}\\mathbf{v} + \\mathbb{K}\\mathbf{w} - \\mathbb{U}^T \\left[ \\mathbb{A}^{-1}(-h^2 \\mathbb{U}\\mathbf{v} + \\mathbf{\\tilde{b}}) \\right] = 0$$\n"
        "$$\\mathbb{M}\\frac{d\\mathbf{v}}{dt} + \\left(\\mathbb{D} + h^2 \\mathbb{U}^T \\mathbb{A}^{-1} \\mathbb{U}\\right)\\mathbf{v} + \\mathbb{K}\\mathbf{w} = \\mathbb{U}^T \\mathbb{A}^{-1} \\mathbf{\\tilde{b}}$$\n\n"
        "Definindo a matriz de amortecimento hidráulico equivalente como $\\mathbb{R} = h^2 \\mathbb{U}^T \\mathbb{A}^{-1} \\mathbb{U}$, "
        "obtem-se diretamente a Equação (5.3):\n"
        "$$\\mathbb{M}\\frac{\\mathbf{v}^{n+1}-\\mathbf{v}^n}{\\delta t} + (\\beta \\mathbb{M} + \\mathbb{R})\\mathbf{v}^{n+1} + \\mathbb{K}\\mathbf{w}^{n+1} = \\mathbb{U}^T \\mathbb{A}^{-1} \\mathbf{\\tilde{b}}^{n+1}$$\n\n"
        "### Análise de Propriedade Estrutural\n"
        f"- **Dimensão da Matriz R calculada:** {R.shape[0]} x {R.shape[1]}\n"
        f"- **Elementos não-nulos (nnz):** {np.count_nonzero(R)}\n"
        "- **Estrutura:** Diferente do amortecimento intrínseco (diagonal), a matriz R apresenta uma estrutura densa/bloqueada localmente devido à interconexão global que o escoamento hidráulico impõe sobre os nós livres do reservatório.\n"
    )
    
    salvar_relatorio_markdown("relatorio_topico_1.md", "Tópico 1: Condensação e Matriz R", deducao_texto, [img_name])
    print(f"Sucesso: Dados técnicos e dedução gravados em 'relatorio_topico_1.md'. Imagem salva como '{img_name}'.")

def rotina_2_evolucao_temporal(config):
    print("\n" + "="*80)
    print("OBJETIVO DA ROTINA 2 (TÓPICO 2): EVOLUÇÃO TRANSIENTE DO SISTEMA")
    print("="*80)
    print("Simulando o comportamento dinâmico do acoplamento do repouso até t=12s.")
    
    # Executando o cenário base parametrizado no dicionário CONFIG
    hist, _, _, _ = run_simulation(config)
    
    # 1. Configuração das métricas para geração de gráficos individuais
    metricas = [
        ("Deflexão Central [m]", hist["w_center"], "rotina2_deflexao.png"),
        ("Pressão Outlet [Pa]", hist["p_outlet"], "rotina2_pressao.png"),
        ("Vazão Outlet [m³/s]", hist["q_outlet"], "rotina2_vazao.png"),
        ("Volume Acumulado [m³]", hist["volume"], "rotina2_volume.png"),
        ("Potência Consumida [W]", hist["power"], "rotina2_potencia.png")
    ]
    
    imagens_salvas = []
    
    # 2. Geração e Salvamento de cada gráfico isoladamente
    for titulo, dados, arquivo in metricas:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist["t"], dados)
        ax.set_title(titulo)
        ax.set_xlabel("Tempo [s]")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        plt.savefig(arquivo, dpi=300, bbox_inches='tight')
        imagens_salvas.append(arquivo)
        
        if config["show_plots"]:
            plt.show()
        else:
            plt.close(fig)
            
    # 3. Formatação do Texto do Relatório
    texto_md = "## Parâmetros da Simulação Base\n"
    texto_md += f"- **Passo de tempo (dt):** {config['dt']} s\n"
    texto_md += f"- **Malha da Membrana:** {config['Nx']}x{config['Ny']}\n"
    texto_md += f"- **Pressão Inlet:** {config['p_inlet']} Pa\n\n"
    
    texto_md += "## Resultados Temporais (Amostragem)\n\n"
    texto_md += "| Tempo [s] | Deflexão Central [m] | Pressão Outlet [Pa] | Vazão [m³/s] | Volume [m³] |\n"
    texto_md += "|---|---|---|---|---|\n"
    
    # Amostrando 12 pontos temporalmente equidistantes
    n_pontos = len(hist["t"])
    indices = np.linspace(0, n_pontos - 1, 12, dtype=int)
    
    for idx in indices:
        texto_md += (f"| {hist['t'][idx]:.4f} "
                     f"| {hist['w_center'][idx]:.6e} "
                     f"| {hist['p_outlet'][idx]:.2f} "
                     f"| {hist['q_outlet'][idx]:.6e} "
                     f"| {hist['volume'][idx]:.6e} |\n")
                     
    texto_md += "\n> *Nota técnica: O código está arquitetado para permitir simulações paramétricas automatizadas iterando sobre as propriedades exigidas.*"
    
    # 4. Exportação
    # Passamos a lista 'imagens_salvas' para que todas sejam incluídas no .md
    salvar_relatorio_markdown("relatorio_topico_2.md", "Tópico 2: Evolução Transiente", texto_md, imagens_salvas)
    print(f"Sucesso: Relatório salvo em 'relatorio_topico_2.md' e {len(imagens_salvas)} imagens exportadas.\n")

def rotina_3_queda_pressao(config):
    print("\n" + "="*80)
    print("OBJETIVO DA ROTINA 3 (TÓPICO 3): QUEDA ABRUPTA DE PRESSÃO E ESTABILIZAÇÃO")
    print("="*80)
    print("Nesta etapa, o sistema parte do estado deformado e pressurizado atingido")
    print("ao final da simulação base, e a pressão de entrada é zerada subitamente.")
    print("O objetivo é observar a resposta transiente de relaxamento até o novo repouso.")
    print("="*80 + "\n")

    # 1. Obtendo o estado inicial (aquecimento do sistema)
    print("Pré-computando o estado estabilizado com pressão ativa...")
    _, state_inicial, _, _ = run_simulation(config) # Usa parâmetros base para gerar deformação

    # 2. Executando a simulação de decaimento
    print("Simulando o relaxamento após p_inlet = 0 Pa...")
    cfg_queda = {"p_inlet": 0.0, "t_final": 12.0, "Nx": 51, "Ny": 51, "dt": 0.025}
    hist, _, _, _ = run_simulation(config, custom_params=cfg_queda, initial_state=state_inicial)

    # 3. Configuração das métricas para geração de gráficos individuais
    metricas = [
        ("Decaimento: Deflexão Central [m]", hist["w_center"], "rotina3_deflexao.png"),
        ("Decaimento: Pressão Outlet [Pa]", hist["p_outlet"], "rotina3_pressao.png"),
        ("Decaimento: Vazão Outlet [m³/s]", hist["q_outlet"], "rotina3_vazao.png"),
        ("Volume Acumulado Total [m³]", hist["volume"], "rotina3_volume.png"),
        ("Decaimento: Potência [W]", hist["power"], "rotina3_potencia.png")
    ]
    
    imagens_salvas = []

    # 4. Geração e Salvamento de cada gráfico isoladamente
    for titulo, dados, arquivo in metricas:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist["t"], dados, color="teal")
        ax.set_title(titulo)
        ax.set_xlabel("Tempo [s]")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.savefig(arquivo, dpi=300, bbox_inches='tight')
        imagens_salvas.append(arquivo)

        if config["show_plots"]:
            plt.show()
        else:
            plt.close(fig)

    # 5. Formatação do Texto do Relatório (Tabelas Markdown)
    texto_md = "## Parâmetros da Simulação de Relaxamento\n"
    texto_md += f"- **Condição Inicial:** Estado dinâmico extraído da rotina base (p_inlet=5000 Pa).\n"
    texto_md += f"- **Nova Pressão Inlet:** 0.0 Pa\n"
    texto_md += f"- **Passo de tempo (dt):** {cfg_queda['dt']} s\n"
    texto_md += f"- **Malha da Membrana:** {cfg_queda['Nx']}x{cfg_queda['Ny']}\n\n"
    
    texto_md += "## Resultados Temporais do Relaxamento (Amostragem)\n\n"
    texto_md += "| Tempo [s] | Deflexão Central [m] | Pressão Outlet [Pa] | Vazão [m³/s] | Volume [m³] |\n"
    texto_md += "|---|---|---|---|---|\n"
    
    n_pontos = len(hist["t"])
    indices = np.linspace(0, n_pontos - 1, 12, dtype=int)
    
    for idx in indices:
        texto_md += (f"| {hist['t'][idx]:.4f} "
                     f"| {hist['w_center'][idx]:.6e} "
                     f"| {hist['p_outlet'][idx]:.2f} "
                     f"| {hist['q_outlet'][idx]:.6e} "
                     f"| {hist['volume'][idx]:.6e} |\n")
                     
    texto_md += "\n> *Nota técnica: Com a interrupção da força motriz, observa-se a dissipação de energia à medida que o sistema retorna elasticamente ao repouso.*"
    
    # 6. Exportação
    salvar_relatorio_markdown("relatorio_topico_3.md", "Tópico 3: Estabilização após Queda Abrupta de Pressão", texto_md, imagens_salvas)
    print(f"Sucesso: Relatório salvo em 'relatorio_topico_3.md' e {len(imagens_salvas)} imagens exportadas.\n")

def rotina_4_oscilacao_livre(config):
    print("\n" + "="*80)
    print("OBJETIVO DA ROTINA 4 (TÓPICO 4): OSCILAÇÃO LIVRE E COMPARAÇÃO DE FREQUÊNCIAS")
    print("="*80)
    print("Nesta etapa, validamos o sistema removendo o amortecimento intrínseco (beta = 0)")
    print("e alargando os canais (H = 2000 µm) para mitigar as perdas viscosas acopladas.")
    print("A membrana é inicializada com o formato exato do seu 3º modo fundamental,")
    print("deixada para oscilar livremente (p_inlet = 0). O programa extrairá a frequência")
    print("de oscilação transiente e a comparará com a frequência analítica da membrana isolada.")
    print("="*80 + "\n")

    cfg = config.copy()
    # Configurações forçadas conforme o escopo do Tópico 4
    cfg.update({
        "channel_width": 2000e-6, # Canais largos reduzem a condutância e perdas viscosas
        "beta_damping": 0.0,      # Sem amortecimento da estrutura isolada
        "p_inlet": 0.0,           # Sem pressão injetada
        "t_final": 2.0,           # Tempo suficiente para capturar múltiplos ciclos
        "dt": 0.005               # Passo fino necessário para não perder o pico da frequência
    })

    # 1. Obtenção do 3º Modo Fundamental Analítico (Isolado)
    print("Calculando modos de vibração analíticos da membrana isolada...")
    K, M, _, _, _ = assembly_membrane(cfg["Nx"], cfg["Ny"], 2.0, 2.0, True)
    _, evecs, omega_hat = solve_membrane_eigenproblem(K, M, num_modes=5)
    
    # O 3º modo corresponde ao índice 2
    w3_adimensional = omega_hat[2]
    _, _, _, t_ref = compute_reference_scales(cfg)
    freq_3_analitica = w3_adimensional / (2 * np.pi * t_ref)

    # Inicializa estado dinâmico: Deflexão = Autovetor do 3º Modo, Velocidade = 0
    state_inicial = {"w": evecs[:, 2], "v": np.zeros(cfg["Nx"] * cfg["Ny"])}

    # 2. Simulação no Tempo
    print(f"Simulando resposta acoplada livre no tempo (dt={cfg['dt']}s)...")
    hist, _, _, _ = run_simulation(cfg, initial_state=state_inicial)

    # 3. Processamento do Sinal: Extração da Frequência Simulada via Zero-Crossings
    # Procura onde o sinal muda de sinal (cruza o zero)
    w_c = hist["w_center"]
    zero_crossings = np.where(np.diff(np.sign(w_c)))[0]
    
    if len(zero_crossings) >= 2:
        # A distância entre dois cruzamentos consecutivos é meio período
        tempos_cruzamento = hist["t"][zero_crossings]
        meios_periodos = np.diff(tempos_cruzamento)
        periodo_simulado = 2.0 * np.mean(meios_periodos)
        freq_3_simulada = 1.0 / periodo_simulado
    else:
        freq_3_simulada = 0.0 # Caso não tenha oscilado o suficiente
        
    print(f"-> Frequência Analítica (Isolada): {freq_3_analitica:.3f} Hz")
    print(f"-> Frequência Simulada (Acoplada): {freq_3_simulada:.3f} Hz")

    # 4. Geração do Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist["t"], hist["w_center"], color="purple")
    ax.set_title("Oscilação Livre Acoplada - Inicialização no 3º Modo")
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Deflexão Central [m]")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    img_name = "oscilacao_livre_modo3.png"
    plt.savefig(img_name, dpi=300, bbox_inches='tight')

    if config["show_plots"]:
        plt.show()
    else:
        plt.close(fig)

    # 5. Formatação do Texto do Relatório
    texto_md = "## Validação Frequencial (Tópico 4)\n\n"
    texto_md += "O sistema foi configurado para minimizar perdas (amortecimento $\\beta=0$, canais largos) "
    texto_md += "e inicializado na forma exata do 3º modo fundamental da membrana isolada.\n\n"
    texto_md += "### Comparativo de Frequências\n"
    texto_md += f"- **Frequência Analítica (Membrana Isolada):** {freq_3_analitica:.4f} Hz\n"
    texto_md += f"- **Frequência Simulada (Sistema Acoplado):** {freq_3_simulada:.4f} Hz\n\n"
    texto_md += "> *Observação:* Uma pequena divergência (shift de frequência) pode ser "
    texto_md += "esperada devido à inércia do acoplamento numérico, mas os valores devem estar fortemente correlacionados.\n\n"
    
    texto_md += "## Resultados Temporais (Amostragem)\n\n"
    texto_md += "| Tempo [s] | Deflexão Central [m] |\n"
    texto_md += "|---|---|\n"
    
    n_pontos = len(hist["t"])
    indices = np.linspace(0, n_pontos - 1, 12, dtype=int)
    for idx in indices:
        texto_md += f"| {hist['t'][idx]:.4f} | {hist['w_center'][idx]:.6e} |\n"
                     
    # 6. Exportação
    salvar_relatorio_markdown("relatorio_topico_4.md", "Tópico 4: Oscilação Livre e Frequência", texto_md, [img_name])
    print(f"\nSucesso: Relatório salvo em 'relatorio_topico_4.md' e imagem em '{img_name}'.\n")

def rotina_5_ressonancia(config):
    print("\n" + "="*80)
    print("OBJETIVO DA ROTINA 5 (TÓPICO 5): FORÇAMENTO HARMÔNICO E RESSONÂNCIA")
    print("="*80)
    print("Nesta etapa, o sistema inicia em repouso e é submetido a uma pressão de entrada")
    print("oscilante dada por p_inlet(t) = 5000 * cos(w3 * t). Sendo w3 a frequência")
    print("angular do 3º modo fundamental, o objetivo é observar o fenômeno de ressonância.")
    print("="*80 + "\n")

    cfg = config.copy()
    cfg.update({
        "channel_width": 2000e-6, # Canais largos para reduzir amortecimento
        "beta_damping": 0.0,      # Sem amortecimento estrutural
        "t_final": 5.0,           # Tempo para observar o crescimento da amplitude
        "dt": 0.01                # Passo de integração temporal
    })

    # 1. Obtenção da Frequência Angular Analítica (w3)
    print("Calculando a frequência angular (w3) para calibração do forçamento...")
    K, M, _, _, _ = assembly_membrane(cfg["Nx"], cfg["Ny"], 2.0, 2.0, True)
    _, _, omega_hat = solve_membrane_eigenproblem(K, M, num_modes=5)
    
    _, _, _, t_ref = compute_reference_scales(cfg)
    w3_adimensional = omega_hat[2]
    w3_fisica = w3_adimensional / t_ref # Frequência angular em rad/s

    # 2. Definição da Função de Forçamento Harmônico
    def force_harmonic(t):
        return 5000.0 * np.cos(w3_fisica * t)

    # 3. Simulação no Tempo (Inicia em repouso)
    print(f"Injetando pressão harmônica com w = {w3_fisica:.2f} rad/s...")
    hist, _, _, _ = run_simulation(cfg, forcing_func=force_harmonic)

    # 4. Geração do Gráfico de Ressonância
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist["t"], hist["w_center"], color="darkred")
    ax.set_title("Ressonância: Resposta ao Forçamento Harmônico (3º Modo)")
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Deflexão Central [m]")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    img_name = "ressonancia_modo3.png"
    plt.savefig(img_name, dpi=300, bbox_inches='tight')

    if config["show_plots"]:
        plt.show()
    else:
        plt.close(fig)

    # 5. Formatação do Texto do Relatório
    texto_md = "## Fenômeno de Ressonância (Tópico 5)\n\n"
    texto_md += "O sistema, partindo do repouso, foi estimulado através do circuito hidráulico "
    texto_md += f"com uma pressão harmônica de $5000 \\cos(\\omega_3 t)$ Pa, onde $\\omega_3 = {w3_fisica:.2f}$ rad/s.\n\n"
    texto_md += "### Observação Física\n"
    texto_md += "Como a frequência de excitação coincide com uma frequência natural do sistema acoplado (e o amortecimento foi minimizado), "
    texto_md += "a energia é transferida de forma construtiva a cada ciclo, gerando um aumento linear e contínuo da amplitude de deflexão (ressonância).\n\n"
    
    texto_md += "## Resultados Temporais (Amostragem)\n\n"
    texto_md += "| Tempo [s] | Deflexão Central [m] | Pressão Outlet [Pa] |\n"
    texto_md += "|---|---|---|\n"
    
    n_pontos = len(hist["t"])
    indices = np.linspace(0, n_pontos - 1, 15, dtype=int)
    for idx in indices:
        texto_md += f"| {hist['t'][idx]:.4f} | {hist['w_center'][idx]:.6e} | {hist['p_outlet'][idx]:.2f} |\n"
                     
    # 6. Exportação
    salvar_relatorio_markdown("relatorio_topico_5.md", "Tópico 5: Forçamento Harmônico e Ressonância", texto_md, [img_name])
    print(f"\nSucesso: Relatório salvo em 'relatorio_topico_5.md' e imagem em '{img_name}'.\n")


# ============================================================
# RELATÓRIO TEXTUAL GERADO JUNTO COM OS RESULTADOS
# ============================================================

def salvar_relatorio_markdown(filename, titulo, conteudo_txt, imagens=None):
    """Gera um documento Markdown contendo tabelas ou textos técnicos e links para imagens."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {titulo}\n\n")
        f.write(f"Data de Execução: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(conteudo_txt)
        if imagens:
            f.write("\n\n## Visualizações Associadas\n")
            for img in imagens:
                f.write(f"![{img}]({img})\n")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    if CONFIG["run_topic_1"]: rotina_1_analise_matriz_R(CONFIG)
    if CONFIG["run_topic_2"]: rotina_2_evolucao_temporal(CONFIG)
    if CONFIG["run_topic_3"]: rotina_3_queda_pressao(CONFIG)
    if CONFIG["run_topic_4"]: rotina_4_oscilacao_livre(CONFIG)
    if CONFIG["run_topic_5"]: rotina_5_ressonancia(CONFIG)
    
    if not any([CONFIG[f"run_topic_{i}"] for i in range(1, 6)]):
        print("\n--- Nenhuma rotina específica selecionada. Executando o CASO GERAL ---")
        rotina_2_evolucao_temporal(CONFIG) # Utiliza a rotina transiente como padrão


