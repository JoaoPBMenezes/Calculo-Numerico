"""
Monta o vetor de vazões nodais impostas no instante t.

    Parâmetros
    ----------
    flow_bc : dict
        Dicionário com as vazões prescritas nos nós.
        Cada entrada tem o formato:
            no: especificacao_da_vazao

    t : float
        Instante de tempo em que as vazões devem ser avaliadas.

    nv : int
        Número total de nós da rede.
 Retorna
    -------
    b : np.ndarray
        Vetor de vazões nodais impostas.
        Cada posição b[i] representa a vazão prescrita no nó i.           
"""
def evaluate_flow_bc(flow_bc: dict, t: float, nv: int) -> np.ndarray:
     # Cria o vetor de vazões nodais, inicialmente todo zero.
    b = np.zeros(nv, dtype=float)
 # Percorre cada nó que possui vazão prescrita.
    for node, spec in flow_bc.items():
        if not (0 <= node < nv):
            raise ValueError(f"Nó {node} fora do intervalo válido.")
        b[node] += evaluate_flow_spec(spec, t)# Avalia a vazão especificada para esse nó no instante t
        # e soma no vetor b.

    return b
"""Aplica condições de contorno de pressão ao sistema linear.

    Parâmetros
    ----------
    A : np.ndarray
        Matriz global do sistema hidráulico.

    b : np.ndarray
        Vetor do lado direito do sistema.
        Em geral, contém as vazões impostas nos nós.

    pressure_bc : dict
        Dicionário com pressões prescritas.
        Formato:
            {indice_do_no: valor_da_pressao}

        Exemplo:
            {5: 0.0, 0: 1500.0}

       Retorna
    -------
    A_mod : np.ndarray
        Matriz do sistema modificada para impor as pressões.

    b_mod : np.ndarray
        Vetor do lado direito modificado com os valores de pressão prescritos. """
def apply_pressure_bc(A: np.ndarray, b: np.ndarray, pressure_bc: dict) -> tuple[np.ndarray, np.ndarray]:
    if len(pressure_bc) == 0:
        raise ValueError("É necessário prescrever pelo menos uma pressão nodal.")
# Faz cópias para não alterar diretamente a matriz e o vetor originais.
    A_mod = A.copy()
    b_mod = b.copy()
# Percorre todos os nós onde a pressão foi prescrita.
    for node, p_value in pressure_bc.items():
        if not (0 <= node < A.shape[0]):
            raise ValueError(f"Nó {node} fora do intervalo válido.")
# Zera toda a linha correspondente ao nó.
        # Isso remove a equação original daquele nó.
        A_mod[node, :] = 0.0
         # Coloca 1 na diagonal.
        # Assim, a nova equação passa a ser:
        #     1 * p_node = valor_prescrito
        A_mod[node, node] = 1.0
         # Coloca no vetor b o valor da pressão desejada.
        b_mod[node] = float(p_value)

    return A_mod, b_mod

"""  Determina a viscosidade dinâmica do fluido.

    Regras:
    - se o usuário fornecer mu_override, esse valor será usado
    - caso contrário, a viscosidade será calculada pela temperatura

    Parâmetros
    ----------
    cfg : dict
        Dicionário de configuração do problema.

    Retorna
    -------
    mu : float
        Viscosidade dinâmica do fluido [Pa.s]"""
def get_mu(cfg: dict) -> float:
     # Se o usuário definiu diretamente a viscosidade, usa esse valor.
    if cfg["mu_override"] is not None:
        mu = float(cfg["mu_override"])
        if mu <= 0:
            raise ValueError("A viscosidade deve ser positiva.")
        return mu
# Se não houver viscosidade imposta,
    # calcula a viscosidade com base na temperatura.
    return water_viscosity_pa_s(float(cfg["temperature_celsius"]))

"""  Determina a área de seção transversal de cada canal da rede.

    Pode funcionar de duas formas:
    1. usando um vetor fornecido pelo usuário, com uma área por aresta
    2. usando um modelo global:
       - área constante
       - diâmetro constante
       - seção retangular constante

    Parâmetros
    ----------
    conec : np.ndarray
        Matriz de conectividade da rede.

    cfg : dict
        Dicionário de configuração do problema.
         Retorna
    -------
    area_edge : np.ndarray
        Vetor com a área de cada aresta/canal.
"""

def get_area_per_edge(conec: np.ndarray, cfg: dict) -> np.ndarray:
    nc = conec.shape[0]  # Número de canais da rede.

    if cfg["area_per_edge"] is not None: # Caso o usuário tenha fornecido diretamente uma área para cada aresta.
        area_edge = np.array(cfg["area_per_edge"], dtype=float)
        if len(area_edge) != nc:  # Verifica se o número de áreas coincide com o número de canais.
            raise ValueError("area_per_edge deve ter o mesmo tamanho do número de arestas.")
        if np.any(area_edge <= 0): 
            raise ValueError("Todas as áreas por aresta devem ser positivas.")
        return area_edge

    mode = cfg["geometry_mode"].lower() # Se não houver área por aresta, usa o modo geométrico global.

    if mode == "area":  # Modo 1: área constante para todos os canais.
        area = float(cfg["area_constant"])
        if area <= 0:
            raise ValueError("A área constante deve ser positiva.")
        return np.full(nc, area, dtype=float)

    if mode == "diameter": # Modo 2: diâmetro constante para todos os canais.
        diameter = float(cfg["diameter_constant"])
        area = circular_area_from_diameter(diameter)
        return np.full(nc, area, dtype=float)

    if mode == "rectangular": 
    # Modo 3: seção retangular constante para todos os canais.
        area = rectangular_area(float(cfg["width"]), float(cfg["height"]))
        return np.full(nc, area, dtype=float)
# Se o modo não for reconhecido, gera erro.
    raise ValueError("geometry_mode inválido. Use 'area', 'diameter' ou 'rectangular'.")

"""Calcula as propriedades hidráulicas dos canais da rede.

    Parâmetros
    ----------
    Xno : np.ndarray
        Coordenadas dos nós da rede.

    conec : np.ndarray
        Matriz de conectividade das arestas/canais.

    cfg : dict
        Dicionário com as configurações físicas e geométricas.

    Retorna
      -------
    dict
        Dicionário contendo:
        - mu: viscosidade dinâmica
        - lengths: comprimento de cada canal
        - area_edge: área de cada canal
        - diameter_eq_edge: diâmetro equivalente de cada canal
        - kappa_edge: constante hidráulica por canal
        - conductance_edge: condutância hidráulica por canal """
def hydraulic_conductivities(Xno: np.ndarray, conec: np.ndarray, cfg: dict) -> dict:
    mu = get_mu(cfg)# Obtém a viscosidade do fluido.
    L = edge_lengths(Xno, conec)# Calcula o comprimento de cada canal.

    if np.any(L <= 0):
        raise ValueError("Há canais com comprimento nulo ou negativo.")

    area_edge = get_area_per_edge(conec, cfg)# Obtém a área de seção transversal de cada canal.
    D_eq = np.sqrt(4.0 * area_edge / np.pi)# Calcula o diâmetro equivalente a partir da área.
    kappa = np.pi * D_eq**4 / (128.0 * mu) # Calcula a constante hidráulica kappa de cada canal.
    C = kappa / L  # Calcula a condutância hidráulica de cada canal.

    return {
        "mu": mu,
        "lengths": L,
        "area_edge": area_edge,
        "diameter_eq_edge": D_eq,
        "kappa_edge": kappa,
        "conductance_edge": C,
    }

"""Resolve a rede hidráulica para um dado instante de tempo.

    Parâmetros
    ----------
    conec : np.ndarray
        Matriz de conectividade das arestas.

    C : np.ndarray
        Vetor de condutâncias hidráulicas das arestas.

    pressure_bc : dict
        Dicionário com pressões prescritas nos nós.

    flow_bc : dict
        Dicionário com vazões prescritas nos nós.

    t : float
        Instante de tempo em que o sistema será resolvido.
        Retorna
    -------
    dict
        Dicionário contendo matrizes, vetores e resultados da solução."""
def solve_network(conec: np.ndarray, C: np.ndarray, pressure_bc: dict, flow_bc: dict, t: float) -> dict:
    A = assembly(conec, C)# Monta a matriz global da rede hidráulica.
    nv = A.shape[0]  # Número de nós da rede.

    b = evaluate_flow_bc(flow_bc, t, nv)  # Monta o vetor de vazões nodais impostas no instante t.
    A_mod, b_mod = apply_pressure_bc(A, b, pressure_bc)# Aplica as pressões prescritas ao sistema.

    p = np.linalg.solve(A_mod, b_mod)# Resolve o sistema linear modificado para encontrar as pressões nodais.

    D = build_incidence_matrix(conec, nv) # Monta a matriz de incidência da rede.
    K = np.diag(C)# Monta a matriz diagonal das condutâncias.
    q = K @ (D @ p) # Calcula as vazões nas arestas:
    # q = K D p
    dp = p[conec[:, 0]] - p[conec[:, 1]]  # Calcula a queda de pressão em cada aresta.

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
