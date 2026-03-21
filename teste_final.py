import numpy as np #importa uma biblioteca no python para manipular eficientemente os dados e matrizes da rede hidráulica
import matplotlib.pyplot as plt # importa uma biblioteca para plotar graficos de linhas simples da rede hidráulica

from gera_grafo import GeraGrafo #Do arquivo chamado gera_grafo, já adicionado, importamos a biblioteca Geragrafo para gerar o grafo da rede hidráulica com os respectivos níveis de ligação indicados no código mais a frente
from plota_rede import PlotaRede #Do arquivo chamado plota_rede, já adicionado, importamos a biblioteca Plotarede, para gerar a rede hidráulica em si, o esquema com os canos interligados formando a rede propriamente dita


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

#As convenções utilizadas para cada unidade de medida estão no SI. A vazão que entra no nó é definida como positiva e a vazão que sai do nó é definida como negativa
#No código, podemos editar essas configurações iniciais para testar novos casos
CONFIG = {
    #Configurações iniciais e padrão da rede hidráulica, todas essas configurações podem ser alteradas caso queira testar valores distintos de área, temperatura, pressão, nível da rede hidráulica, tempo, entre outras medidas'
    # ---------------------------
    # Geometria da rede
    # ---------------------------
    #Configurações da geometria da rede'

    "levels": 1,              #Pode ser editado para testar novos casos, é relacionado ao tamnho da rede, pode aumentar ou diminuir o tamanho da rede'
    "coord_scale_to_m": 1e-3, #Ajuste das coordenadas fornecidas pela função Geragrafo, que fornece as coordenadas dos nós e das conexões em mm no grafo, mas como está sendo utilizado o SI como convenção (como visto em cima), então aqui esses valores são convertidos para metros, por isso o 1e-3, significa o 10 elevado a -3 em python

    # ---------------------------
    # Modelo temporal
    # ---------------------------
    #Configurações do regime/modelo de tempo utilizado na rede hidráulica, para calcular vazões, mudanças de pressão com o tempo, entre outras medidas
    "time_mode": "transient",  #O regime temporal pode ser escolhido como steady (ou seja, estático ou parado), nesse caso as configurações seriam constantes com o tempo e as medidas não se alterariam, ficam estáveis em qualquer ponto da rede. Ou pode escolher o regime transient (escolhido nesse caso), em que os valores de Vazão e pressão são variáveis com o tempo
    "t0": 0.0,  #tempo inicial
    "tf": 10.0,   #tempo final
    "dt": 0.2,    #Variação de tempo para cada medição de Vazão e pressão, isto é, a cada dt de 0.2 segundos a pressão e a Vazão mudarão e serão medidas 

    # ---------------------------
    # Propriedades físicas dos canais
    # ---------------------------

    #Agora passamos de falar da rede para as propriedades físicas específicas dos canais (canos), como área e volume por exemplo, essenciais para calcular pressão e Vazão de cada cano da rede hidráulica'
    # geometry_mode: 
    #dependendo da geometria do modelo do canal o modo de calcular pode ser distinto, mas aqui são os valores da área do cano (se o programa quiser manter uma area fixa, aqui independe da geometria do cano), do diametro do cano (para caso seja um cano redondo/circular), da largura e altura do cano (para caso seja um cano retangular)
    #   "area"        -> usa area_constant 
    #se a preferencia for por manter uma area de cano constante para calcular a vazão da rede hidraulica'
    #   "diameter"    -> usa diameter_constant
    #em geral usado para canos circulares, mantem-se um valor constante de diametro da circunferencia do cano para calcular sua área, vazão e demais propriedades '
    #   "rectangular" -> usa width e height
    #Usa o comprimento e a largura definidos, em geral para canos retangulares, para cálculo de área'
    "geometry_mode": "rectangular",
    # O modo da geometria utilizado é o retangular, isto é, o cano deve ser retangular'
    "area_constant": 500e-6 * 500e-6,
    # A área foi definida com o valor de 500 vezes 10 elevado a -6 multiplicado por 500 vezes 10 a -6, entretanto, como o modo escolhido foi o retangular, então não será utilizada esssa área definida, mas se mudar o regime para a área constante então esse será o valor de área utilizado, mas que pode ser alterado caso queira'
    "diameter_constant": 800e-6,
    #O diametro foi definido com o valor de 800 vezes 10 a -6, entretanto, como o modo escolhido foi definido como retangular, então esse valor de diametro não será utilizado neste caso, mas caso o regime mude para circular, o diâmetro já estará definido com esse valor, que pode ser mudado caso queira'
    "width": 500e-6,
    #largura do retangulo da base do cano, ou seja, largura do cano para o cálculo da área no caso retangular, que será o utilizado nesse exemplo, o valor dele é de 500 vezes 10 elevado a -6'
    "height": 500e-6,
    #O comprimento do retangulo da base do cano, ou seja, comprimento do cano para o cálculo da área no caso retangular, que será o utilizado nesse exemplo, o valor dele é de 500 vezes 10 elevado a -6'
    # Se quiser propriedades diferentes por canal, coloque aqui.
    #Outras propriedades para além da área, diametro, comprimento e largura podem ser acrescentadas aqui, em forma de vetor'
    # Ex.: [2.5e-7, 2.5e-7, 3.0e-7, ...]
    #aqui exemplos de valores distintos que podem ser utilizados'
    # Se None, o código usa o modo global acima.
    #A área por curva, somente se quiser alterar o modo nos cantos do cano para algo diferente do já utilizado, nesse caso, o None indica que o modo geométrico dos cantos do cano é o mesmo do modo global já definido acima, que no caso foi o retangular, sendo assim, os cantos do cano tambem tem formato retangular'
    "area_per_edge": None,

    # ---------------------------
    # Temperatura e viscosidade
    # ---------------------------
    "temperature_celsius": 25.0,
    #A temperatura está em celsius e foi predeterminada com um valor escolhido de 25.0'
    #É possível colocar aqui um valor forçado para a viscosidade, para deixá-la predeterminada, evidentemente que nesse caso algumas das variáveis do problema já estarão predeterminadas e, dependendo da fórmula utilizada para relacionar temperatura e viscosidade pode ocorrer algum problema ou discrepancia, mas, no geral, é possível forçar um valor de viscosidade, pois nem sempre será utilizada uma fórmula'
    # Se quiser FORÇAR a viscosidade, coloque um valor aqui.
    # Se None, a viscosidade será calculada pela temperatura.
    #Aqui pode colocar um regime de viscosidade de preferencia, se estiver escrito None então a viscosidade será calculada a partir da temperatura predeterminada'
    "mu_override": None,

    # ---------------------------
    # Pressões prescritas
    # ---------------------------
    # Exemplo:
    # nó 5 em 0 Pa
    # nó 0 em 1500 Pa
    #
    # Você pode colocar quantos nós quiser.
    #Aqui as pressões em cada nó podem ser predeterminadas e definidas, para quantos nós quiser'
    "pressure_bc": {
        5: 0.0,
        10: 0.1,
    },
    #A pressão no cano bc foi definida pelas pressões nos nós, ou seja, a struct de pressão no cano bc é a pressão em cada nó que liga para gerar o cano, nesse caso, do cano bc, são os nós 5 e 10, os quais possuem valores de pressão, nesse caso, definidos como 0.0 e 0.1'
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
    #A struct da vazão é definida parecida com a pressão, a vazão em bc foi definida pelos nós 0 e 3'
    #O formato das vazões em cada nó é primeiro explicitado pelo tipo da vazão, que pode ser constante, senoidal ou cossenoidal. No caso constante define-se o valor para essa pressão, no exemplo acima no nó 0 o valor é de 1 vezes 10 elevado a -7. No tipo senoide coloca type:sin e explicita-se o valor médio da função seno que descreve a vazão, a amplitude dessa função, a frequencia e a fase inicial da onda senoide, assim definindo unicamente a onda senoide característica. No nó 3 o tipo definido é seno, o valor médio é de 1 vezes 10 elevado a -7, a amplitude é de 2 vezes 10 elevado a -8, a frequencia é de 0.5 e a fase inicial é 0.0. Para o caso do nó 4, que é cosseno, é análogo ao seno'
    "flow_bc": {
        0: {"type": "sin", "mean": 1.0e-7, "amp": 0.3e-7, "freq": 0.5, "phase": 0.0},
        3: {"type": "cos", "mean": -0.2e-7, "amp": 0.1e-7, "freq": 0.25, "phase": 0.0},
    },

    # ---------------------------
    # Saídas gráficas
    # ---------------------------
    "plot_network": True,
    #Aqui tem a saída da rede, plotada num gráfico'
    "plot_time_series": True,
    #Aqui tem a saída da série de tempo, ou seja, plota o gráfico em função do tempo, ambas dos arquivos externos importados'

    # Escolha quais nós e arestas quer acompanhar no tempo
    "nodes_to_plot": [0, 2, 5],
    "edges_to_plot": [0, 1, 2],
    #Aqui escolhe quais cantos e quais nós serão acompanhados ao longo do tempo e, consequentemente, plotados na função chamada acima plot_time_series do arquivo externo importado
    # Mostra tabela detalhada dos canais no terminal
    "print_edge_table": True,
    #Aqui printa a tabela com os valores das pressões nos cantos'
}


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


"""
    def compute_power  
        Calcula a potencia total dissipada pelo atrito do fluido nos canais
    atraves de um produto Matricial  p^t*A*p, no qual a Matriz A corresponde A: A=D^t*K*D (Matriz de Condutancia Global)

    Paramametros
    ----------
    p : np.ndarray
        Vetor com as pressoes calculadas em cada no

    D : np.ndarray
        Matriz de incidencia que mostra como os nos estao conectados, 

    K : np.ndarray
        Matriz de condutancia(relfete o quao dificil e deixar o fluido passar em cada canal)
    ----------
    Retorna 

    Float
        Um numero real indicando a potencia hidraulica total.
        
"""

def compute_power(p: np.ndarray, D: np.ndarray, K: np.ndarray) -> float:
    # O termo (D.T @ K @ D) calcula a matriz de condutancia da rede.
    # O produto p.T @ Matriz @ p calcula a forma quadratica que resulta na dissipacao de energia.
    return float(p.T @ (D.T @ K @ D) @ p)


"""
    def nodal_mass_residual  
        Essa funcao serve para a verificacao da consistencia do modelo implementado atraves da verificao da conservacao da Lei das massas,
    assim ela faz a somatoria das vazoes de cada no, e esse valor tem que ser igual ou aproximadamente igual a 0. 
    Se o valor for diferente de zero, aparentemente teve um erro numerico no sistema 

    Paramametros
    ----------
    conec : np.ndarray
        Matriz de conectividade que define os nós de origem e destino de cada canal.

    q : np.ndarray
        Vetor contendo as vazoes calculadas para cada canal [m³/s]

    imposed_b : np.ndarray
        Os vetores com as vazoes forcadas externamente na rede
    ----------
    Retorna 

    np.ndarray
        Vetor de residuos nodais. Valores proximos de zero (ex: 1e-16) indicando que a conservacao de massas foi satisfeita
        
"""

def nodal_mass_residual(conec: np.ndarray, q: np.ndarray, imposed_b: np.ndarray) -> np.ndarray:
    nv = int(np.max(conec)) + 1 #Numero totais de nos
    residual = np.zeros(nv, dtype=float) #Cria um vetor para guardar a respsota e operar sobre ele

    #Para cada canal k, que conecta os nos i e j
    for k, (i, j) in enumerate(conec): 
        # A vazao q[k] sai do no "i" da aresta
        residual[i] += q[k]
        # A vazao q[k] entra do no "j" da aresta
        residual[j] -= q[k]
    #Assim acumulando as vazoes da rede inteira

    #Desconsidera as vazoes aplicadas externamente no sistema
    residual -= imposed_b 
    return residual



"""
   def print_inputs_summary 
        Esta funcao exibe um resumo organizado dos dados de entrada da simulacao. 
    Sua principal utilidade e permitir a conferencia das unidades no Sistema 
    Internacional (SI), prevenindo erros de interpretacao de escala.

    Parametros
    ----------
    cfg : dict
        Dicionario de configuracoes da rede (CONFIG).
    hydraulic_data : dict
        Dicionario contendo propriedades fisicas e geometricas calculadas.
    Xno : np.ndarray
        Matriz de coordenadas dos nos[m].
    conec : np.ndarray
        Matriz de conectividade que define os nos de origem e destino de cada canal.
    
        ----------
        
"""
def print_inputs_summary(cfg: dict, hydraulic_data: dict, Xno: np.ndarray, conec: np.ndarray) -> None:
    #Prints das propriedades da simulacao em si
    print("=" * 88)
    print("ENTRADAS UTILIZADAS NA SIMULACAO")
    print("=" * 88)
    print(f"Níveis da rede (levels): {cfg['levels']}") 
    print(f"Número de nós gerados: {Xno.shape[0]}") 
    print(f"Número de canais gerados: {conec.shape[0]}") 
    print(f"Modo temporal: {cfg['time_mode']}")
    print(f"Intervalo de tempo: [{cfg['t0']}, {cfg['tf']}] s")
    print(f"Passo de tempo: {cfg['dt']} s")
    print()

    #Print das Propriedades Fisicas
    print("Propriedades fisicas:")
    print(f"  geometry_mode: {cfg['geometry_mode']}")
    print(f"  temperature_celsius: {cfg['temperature_celsius']}")
    print(f"  mu_override: {cfg['mu_override']}")
    print(f"  viscosidade usada [Pa.s]: {hydraulic_data['mu']:.6e}")

    # Bloco condicional : Seleciona qual propriedade geometrica 
    # exibir com base no modo escolhido CONFIG.
    #.6e->Formata a resposta para notacao cientifica
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
    print("Vazoes prescritas:")
    if len(cfg["flow_bc"]) == 0:
        print("  nenhuma")
    else:
        for node, spec in cfg["flow_bc"].items():
            print(f"  nó {node}: {spec}")

    print("=" * 88)


"""
    def print__output_summary 
            Esta funcao consolida e apresenta os resultados finais da simulacao de forma 
        estruturada utilizando o conceito de encapsulamento atraves do dicionario 
        chamado result, que centraliza o estado final da rede, incluindo as matrizes de 
        incidencia D e condutancia K. Alem de exibir pressoes e vazoes, a funcao 
        invoca calculos de verificacao potencia e residuo para validar a solucao.

    Paramametros
    ----------
    cfg : dictionary
        O dicionario de configuracaos da rede utilizados para resolver o sistema

    hydraulic_data : dictionary
        Dicionário contendo propriedades físicas e geométricas calculadas.

    result : dictionary
        Dicionário contendo os vetores de resposta (p, q, b) e matrizes do sistema.

    conec : np.ndarray
        Matriz de conectividade que define os nós de origem e destino de cada canal.
    
    ----------
        
"""

def print_output_summary(cfg: dict, hydraulic_data: dict, result: dict, conec: np.ndarray) -> None:
    # Pega os valores empacotados no dicionario result e coloca em variaveis locais
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

    #Chamando as funcoes compute_power(calcula a potencia total do atrito do fluido nos canais) e 
    #residual(verifica a conservacao de massa do sistema), para validacao do resultado
    power = compute_power(p, D, K)
    residual = nodal_mass_residual(conec, q, b)

    print()
    print("=" * 88)
    print("SAÍDAS DA SIMULAÇÃO")
    print("=" * 88)

    #Imprime as pressoes em todos os nos do sistema em Pascal [Pa](pelo menos 6 casas decimais)
    print("Pressoes nodais [Pa]:")
    for i, pi in enumerate(p):
        print(f"  nó {i}: {pi:.6e}")

    print()
    #Imprime as vazoes nos ultimo instante
    print("Vazões impostas avaliadas no ultimo instante [m^3/s]:")
    for i, bi in enumerate(b):
        if abs(bi) > 0:
            print(f"  nó {i}: {bi:.6e}")

    #Mostra o comportamento detalhado de cada trecho da rede
    if cfg["print_edge_table"]:
        print()
        #Printa os dados dos canais
        print("Dados dos canais:")
        print("canal | i -> j | comprimento [m] | area [m²] | D_eq [m] | C [m³/(Pa.s)] | Δp [Pa] | q [m³/s]")
        #Usa f-strings com especificadores de largura (ex: {k:5d})
        #para garantir que as colunas da tabela de canais fiquem 
        #perfeitamente alinhadas no console.
        for k, (i, j) in enumerate(conec):
            print(
                f"{k:5d} | {i:3d}->{j:<3d} | "
                f"{lengths[k]:.6e} | {area_edge[k]:.6e} | {diameter_eq[k]:.6e} | "
                f"{C[k]:.6e} | {dp[k]:.6e} | {q[k]:.6e}"
            )

    #Printa a potencia total consumida/dissipada pelo sistema
    print()
    print(f"Potencia hidraulica [W]: {power:.6e}")

    #Printa a conservacao de massa em cada no e O residuo deve tender a zero em uma solucao convergente 
    print()
    print("Residuo de conservacao de massa por no:")
    for i, ri in enumerate(residual):
        print(f"  nó {i}: {ri:.6e}")

    print("=" * 88)

"""

print_final_explanation(cfg: dict, hydraulic_data: dict, result: dict)
Esta função recebe 3 entradas na forma de dicionários nomeados:
cfg: contém o tipo do sistema em relação a vazão: “steady” corresponde a resolver o exercício
de uma vez só, enquanto “transient” corresponde a existência de várias instâncias de tempo
hydraulic_data: um dicionário contendo dados hidráulicos sobre o sistema
result: contém vários resultados do sistema. A função utiliza o “time” que corresponde a
instância de tempo atual
O objetivo dessa função é printar no terminal explicações e comentários para o usuário sobre o
que foi feito, variando de acordo com a entrada cfg[“time_mode”]

"""
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

"""

Esta função recebe 3 entradas, sendo que duas destas podem ser ausentes, ou seja, nulas.
results: um dicionário contendo resultados de varios processos ao longo da resolução do
sistema
node_indices: uma lista contendo os índices dos nós do sistema (em int)
edge_indices: uma lista contendo os índices das arestas/canos da rede (em int)
Caso node_indices e/ou edge_indices forem não nulas, a função irá imprimir os gráficos de
pressão por tempo em certos nós ou vazão por tempo em certos canos, respectivamente.

"""
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

"""
A função que irá iniciar todo o processo, contendo os parâmetros a serem utilizados para a
geração da rede, criação do vetor tempo e resolução do sistema.
"""
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