# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def water_viscosity_pa_s(T_celsius: float) -> float:
    """
    Calcula a viscosidade dinâmica aproximada da água em função da temperatura.

    Parâmetros
    ----------
    T_celsius : float
        Temperatura da água em graus Celsius [°C].

    Retorna
    -------
    float
        Viscosidade dinâmica do fluido [Pa.s].
    """
    A = 2.414e-5
    B = 247.8
    C = 140.0
    return A * 10 ** (B / ((T_celsius + 273.15) - C))

def circular_area_from_diameter(diameter: float) -> float:
    """
    Calcula a área da seção transversal para canais de geometria circular.

    Parâmetros
    ----------
    diameter : float
        Diâmetro do canal [m].

    Retorna
    -------
    float
        Área da seção transversal circular [m²].
    """
    if diameter <= 0:
        raise ValueError("O diâmetro deve ser positivo.")
    return np.pi * diameter**2 / 4.0

def rectangular_area(width: float, height: float) -> float:
    """
    Calcula a área da seção transversal para microcanais retangulares.

    Parâmetros
    ----------
    width : float
        Largura do canal [m].
    height : float
        Altura do canal [m].

    Retorna
    -------
    float
        Área da seção transversal retangular [m²].
    """
    if width <= 0 or height <= 0:
        raise ValueError("Largura e altura devem ser positivas.")
    return width * height

def equivalent_diameter_from_area(area: float) -> float:
    """
    Calcula o diâmetro hidráulico equivalente a partir de uma área genérica.

    Parâmetros
    ----------
    area : float
        Área da seção transversal do canal [m²].

    Retorna
    -------
    float
        Diâmetro hidráulico equivalente [m].
    """
    if area <= 0:
        raise ValueError("A área deve ser positiva.")
    return np.sqrt(4.0 * area / np.pi)

def edge_lengths(Xno: np.ndarray, conec: np.ndarray) -> np.ndarray:
    """
    Calcula o comprimento geométrico de cada canal da rede.

    Parâmetros
    ----------
    Xno : np.ndarray
        Matriz com as coordenadas espaciais (x, y) de cada nó.
    conec : np.ndarray
        Matriz de conectividade das arestas (nó de origem, nó de destino).

    Retorna
    -------
    np.ndarray
        Vetor contendo o comprimento (distância euclidiana) de cada aresta [m].
    """
    x1 = Xno[conec[:, 0], :]
    x2 = Xno[conec[:, 1], :]
    return np.linalg.norm(x2 - x1, axis=1)

def build_incidence_matrix(conec: np.ndarray, nv: int | None = None) -> np.ndarray:
    """
    Monta a matriz de incidência topológica do grafo da rede.

    Parâmetros
    ----------
    conec : np.ndarray
        Matriz de conectividade das arestas.
    nv : int | None
        Número total de nós (inferido automaticamente se omitido).

    Retorna
    -------
    D : np.ndarray
        Matriz de incidência (n_canais x n_nós). 
        Valores: 1 (sai do nó), -1 (chega no nó), 0 (sem conexão).
    """
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
    """
    Monta a matriz global de condutância da rede hidráulica.

    Parâmetros
    ----------
    conec : np.ndarray
        Matriz de conectividade identificando os nós interligados por cada canal.
    C : np.ndarray
        Vetor contendo a condutância hidráulica de cada canal.

    Retorna
    -------
    A : np.ndarray
        Matriz quadrada global do sistema, simétrica, cujas somas das linhas 
        e colunas são nulas (conservação de massa).
    """
    nv = int(np.max(conec)) + 1
    nc = conec.shape[0]

    A = np.zeros((nv, nv), dtype=float)

    for k in range(nc):
        i = int(conec[k, 0])
        j = int(conec[k, 1])
        ck = float(C[k])

        A[i, i] += ck
        A[j, j] += ck
        A[i, j] -= ck
        A[j, i] -= ck

    return A

def get_time_vector(cfg: dict) -> np.ndarray:
    """
    Gera o vetor de tempo discretizado para a simulação transiente ou estacionária.

    Parâmetros
    ----------
    cfg : dict
        Dicionário de configuração geral contendo os parâmetros temporais 
        (modo, tempo inicial, tempo final e passo).

    Retorna
    -------
    np.ndarray
        Vetor unidimensional contendo os instantes de tempo da simulação [s].
    """
    if cfg["time_mode"] == "steady":
        return np.array([cfg["t0"]], dtype=float)

    t0 = cfg["t0"]
    tf = cfg["tf"]
    dt = cfg["dt"]

    n = int(np.floor((tf - t0) / dt)) + 1
    return t0 + np.arange(n) * dt

def evaluate_flow_spec(spec: dict, t: float) -> float:
    """
    Calcula a vazão prescrita em um determinado instante t.

    Parâmetros
    ----------
    spec : dict
        Dicionário com a especificação matemática do sinal de vazão 
        (tipo: constante, senoidal ou cossenoidal, além de amplitude e frequência).
    t : float
        Instante de tempo atual para avaliação do sinal [s].

    Retorna
    -------
    float
        Valor instantâneo da vazão prescrita [m³/s].
    """
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