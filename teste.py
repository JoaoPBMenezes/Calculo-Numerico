# ============================================================
# ANÁLISE DE REDE HIDRÁULICA EM GRAFO
# ============================================================
#
# Este programa:
# 1. Gera uma rede de canais a partir de um grafo
# 2. Calcula os comprimentos dos canais
# 3. Calcula as condutâncias hidráulicas
# 4. Monta o sistema linear da rede
# 5. Resolve as pressões nos nós
# 6. Calcula as vazões em cada canal
# 7. Calcula a potência dissipada / potência da bomba
# 8. Plota a rede com pressões e sentido do escoamento
#
# ============================================================


# ------------------------------------------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# ------------------------------------------------------------

# numpy: usado para cálculos numéricos, vetores e matrizes
import numpy as np

# matplotlib: usado para exibir o gráfico da rede
import matplotlib.pyplot as plt

# Importa a função que gera o grafo da rede
from gera_grafo import GeraGrafo

# Importa a função que desenha a rede
from plota_rede import PlotaRede


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def edge_lengths(Xno: np.ndarray, conec: np.ndarray) -> np.ndarray:
    """
    Calcula o comprimento de cada canal da rede.

    Parâmetros:
    - Xno: matriz com as coordenadas dos nós
           cada linha representa um nó: [x, y]
    - conec: matriz de conectividade
             cada linha representa um canal ligando dois nós [i, j]

    Retorna:
    - vetor com o comprimento de cada canal
    """

    # Seleciona as coordenadas do primeiro nó de cada aresta
    x1 = Xno[conec[:, 0], :]

    # Seleciona as coordenadas do segundo nó de cada aresta
    x2 = Xno[conec[:, 1], :]

    # Calcula a distância euclidiana entre os dois extremos de cada canal
    return np.linalg.norm(x2 - x1, axis=1)


def hydraulic_diameter_from_area(area: float) -> float:
    """
    Calcula um diâmetro hidráulico equivalente a partir da área:
        D = sqrt(4A/pi)

    Parâmetro:
    - area: área da seção transversal do canal

    Retorna:
    - diâmetro hidráulico equivalente
    """

    # Verifica se a área é válida
    if area <= 0:
        raise ValueError("A área da seção transversal deve ser positiva.")

    # Fórmula do diâmetro equivalente
    return np.sqrt(4.0 * area / np.pi)


def hydraulic_conductivities(
    Xno: np.ndarray,
    conec: np.ndarray,
    area: float,
    mu: float,
) -> tuple[np.ndarray, float, float]:
    """
    Calcula a condutância hidráulica de cada canal.

    Fórmulas usadas:
        C_k = kappa / L_k
        kappa = pi * D^4 / (128 * mu)
        D = sqrt(4A/pi)

    Onde:
    - L_k = comprimento do canal k
    - mu = viscosidade dinâmica
    - A = área da seção do canal

    Retorna:
    - C: vetor de condutâncias de cada canal
    - D: diâmetro hidráulico equivalente
    - kappa: constante hidráulica comum
    """

    # A viscosidade precisa ser positiva
    if mu <= 0:
        raise ValueError("A viscosidade dinâmica deve ser positiva.")

    # Calcula o comprimento de cada canal
    L = edge_lengths(Xno, conec)

    # Verifica se algum canal ficou com comprimento inválido
    if np.any(L <= 0):
        raise ValueError("Há canais com comprimento nulo ou negativo.")

    # Calcula o diâmetro hidráulico equivalente
    D = hydraulic_diameter_from_area(area)

    # Calcula kappa pela fórmula de Poiseuille
    kappa = np.pi * D**4 / (128.0 * mu)

    # Condutância de cada canal = kappa / comprimento
    C = kappa / L

    return C, D, kappa


# ============================================================
# MONTAGEM DA MATRIZ GLOBAL DA REDE
# ============================================================

def assembly(conec: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Monta a matriz global A da rede hidráulica.

    Para cada canal entre nós i e j, a contribuição local é:
        [ c  -c ]
        [ -c  c ]

    Parâmetros:
    - conec: conectividade dos canais
    - C: condutância de cada canal

    Retorna:
    - matriz global A
    """

    # Garante que conec tenha 2 colunas: nó inicial e nó final
    if conec.ndim != 2 or conec.shape[1] != 2:
        raise ValueError("conec deve ter shape (nc, 2).")

    # Garante que há uma condutância para cada canal
    if len(C) != len(conec):
        raise ValueError("O vetor C deve ter um valor por aresta.")

    # Número de nós = maior índice de nó + 1
    nv = int(np.max(conec)) + 1

    # Número de canais
    nc = conec.shape[0]

    # Inicializa a matriz global A com zeros
    A = np.zeros((nv, nv), dtype=float)

    # Percorre todos os canais
    for k in range(nc):
        # Nó inicial do canal
        n1 = int(conec[k, 0])

        # Nó final do canal
        n2 = int(conec[k, 1])

        # Condutância do canal
        ck = float(C[k])

        # Soma a contribuição local na matriz global
        A[n1, n1] += ck
        A[n1, n2] -= ck
        A[n2, n1] -= ck
        A[n2, n2] += ck

    return A


def build_incidence_matrix(conec: np.ndarray, nv: int | None = None) -> np.ndarray:
    """
    Monta a matriz de incidência D.

    Para cada canal k ligando i -> j:
        D[k, i] =  1
        D[k, j] = -1

    Assim, D @ p fornece a queda de pressão orientada em cada canal.

    Retorna:
    - matriz D
    """

    # Se o número de nós não foi passado, calcula automaticamente
    if nv is None:
        nv = int(np.max(conec)) + 1

    # Número de canais
    nc = conec.shape[0]

    # Inicializa D com zeros
    D = np.zeros((nc, nv), dtype=float)

    # Preenche a matriz de incidência
    for k in range(nc):
        i = int(conec[k, 0])
        j = int(conec[k, 1])

        D[k, i] = 1.0
        D[k, j] = -1.0

    return D


# ============================================================
# RESOLUÇÃO DA REDE
# ============================================================

def solve_network(
    conec: np.ndarray,
    C: np.ndarray,
    natm: int,
    nB: int,
    QB: float,
    patm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve a rede hidráulica.

    Condições:
    - no nó nB entra uma vazão QB
    - no nó natm a pressão é fixa e vale patm

    Retorna:
    - p: vetor de pressões nodais
    - q: vetor de vazões nos canais
    - A: matriz global original
    - D: matriz de incidência
    """

    # Monta a matriz global da rede
    A = assembly(conec, C)

    # Número de nós
    nv = A.shape[0]

    # Verifica se os índices dos nós são válidos
    if not (0 <= natm < nv):
        raise ValueError("natm fora do intervalo de nós.")
    if not (0 <= nB < nv):
        raise ValueError("nB fora do intervalo de nós.")

    # Faz uma cópia da matriz A para aplicar a condição de contorno
    Atilde = A.copy()

    # Inicializa o vetor do lado direito do sistema
    b = np.zeros(nv, dtype=float)

    # Aplica vazão de entrada no nó da bomba
    b[nB] = QB

    # Impõe a pressão fixa no nó atmosférico:
    # substitui toda a linha por zero
    Atilde[natm, :] = 0.0

    # coloca 1 na diagonal dessa linha
    Atilde[natm, natm] = 1.0

    # força a pressão desse nó a ser patm
    b[natm] = patm

    # Resolve o sistema linear Atilde * p = b
    p = np.linalg.solve(Atilde, b)

    # Monta a matriz de incidência
    D = build_incidence_matrix(conec, nv)

    # Monta a matriz diagonal das condutâncias
    K = np.diag(C)

    # Calcula as vazões:
    # q = K D p
    q = K @ (D @ p)

    return p, q, A, D


# ============================================================
# PÓS-PROCESSAMENTO
# ============================================================

def compute_pump_power(p: np.ndarray, C: np.ndarray, D: np.ndarray) -> float:
    """
    Calcula a potência hidráulica dissipada / requerida:
        W = p^T (D^T K D) p
    """

    # Monta a matriz diagonal de condutâncias
    K = np.diag(C)

    # Aplica a fórmula da potência
    return float(p.T @ (D.T @ K @ D) @ p)


def compute_pressure_drops(p: np.ndarray, conec: np.ndarray) -> np.ndarray:
    """
    Calcula a queda de pressão em cada canal:
        Delta p = p_i - p_j
    seguindo a orientação do canal i -> j
    """

    return p[conec[:, 0]] - p[conec[:, 1]]


def check_mass_conservation(
    conec: np.ndarray,
    q: np.ndarray,
    injection_node: int,
    injection_flow: float,
) -> np.ndarray:
    """
    Verifica a conservação de massa em cada nó.

    Para cada canal orientado i -> j:
    - sai q do nó i
    - entra q no nó j

    O resíduo ideal dos nós internos deve ser próximo de zero.
    """

    # Número de nós
    nv = int(np.max(conec)) + 1

    # Inicializa vetor de resíduos
    residual = np.zeros(nv, dtype=float)

    # Percorre cada canal
    for k, (i, j) in enumerate(conec):
        # Vazão saindo de i
        residual[i] += q[k]

        # Vazão entrando em j
        residual[j] -= q[k]

    # Compensa a injeção externa no nó da bomba
    residual[injection_node] -= injection_flow

    return residual


# ============================================================
# RELATÓRIO DOS RESULTADOS
# ============================================================

def print_summary(
    Xno: np.ndarray,
    conec: np.ndarray,
    lengths: np.ndarray,
    C: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    dp: np.ndarray,
    power: float,
    mass_residual: np.ndarray,
    inlet_node: int,
    outlet_node: int,
):
    """
    Imprime um resumo organizado dos resultados no terminal.
    """

    # Ajusta a forma como o numpy mostra números
    np.set_printoptions(precision=6, suppress=False)

    print("=" * 78)
    print("RESUMO DA REDE HIDRÁULICA")
    print("=" * 78)
    print(f"Número de nós      : {Xno.shape[0]}")
    print(f"Número de canais   : {conec.shape[0]}")
    print(f"Inlet              : nó {inlet_node}")
    print(f"Outlet             : nó {outlet_node}")
    print()

    print("-" * 78)
    print("PRESSÕES NODAIS [Pa]")
    print("-" * 78)
    for i, pi in enumerate(p):
        print(f"Nó {i:3d}: {pi:.6e}")

    print()
    print("-" * 78)
    print("DADOS DOS CANAIS")
    print("-" * 78)
    print("canal |   i -> j | comprimento [m] | condutância [m^3/(Pa.s)] | Δp [Pa] | q [m^3/s]")
    for k, (i, j) in enumerate(conec):
        print(
            f"{k:5d} | {i:3d} -> {j:<3d} | "
            f"{lengths[k]:.6e} | {C[k]:.6e} | {dp[k]:.6e} | {q[k]:.6e}"
        )

    print()
    print("-" * 78)
    print("GRANDEZAS GLOBAIS")
    print("-" * 78)
    print(f"Potência hidráulica [W]: {power:.6e}")

    print()
    print("-" * 78)
    print("RESÍDUO DE CONSERVAÇÃO DE MASSA POR NÓ")
    print("(idealmente próximo de zero nos nós internos)")
    print("-" * 78)
    for i, ri in enumerate(mass_residual):
        print(f"Nó {i:3d}: {ri:.6e}")

    print("=" * 78)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    """
    Função principal do programa.
    """

    # --------------------------------------------------------
    # 1) DEFINIÇÃO DA REDE
    # --------------------------------------------------------

    # Número de níveis da árvore/rede gerada
    levels = 2

    # Gera os nós e a conectividade da rede
    Xno, conec = GeraGrafo(levels=levels)

    # Converte as coordenadas de milímetros para metros
    mm_to_m = 1e-3
    Xno = Xno * mm_to_m

    # Define o nó de entrada (inlet)
    inlet_node = 0

    # Define o nó de saída (outlet)
    outlet_node = 5

    # --------------------------------------------------------
    # 2) PARÂMETROS FÍSICOS
    # --------------------------------------------------------

    # Área da seção transversal do canal
    # 500 micrômetros x 500 micrômetros
    area = 500e-6 * 500e-6

    # Viscosidade dinâmica da água [Pa.s]
    mu = 1e-3

    # Vazão de entrada [m^3/s]
    Qin = 1e-7

    # Pressão atmosférica manométrica no outlet [Pa]
    patm = 0.0

    # --------------------------------------------------------
    # 3) CONDUTÂNCIAS HIDRÁULICAS
    # --------------------------------------------------------

    # Calcula:
    # - C: condutância de cada canal
    # - D_h: diâmetro hidráulico equivalente
    # - kappa: constante hidráulica
    C, D_h, kappa = hydraulic_conductivities(
        Xno=Xno,
        conec=conec,
        area=area,
        mu=mu,
    )

    # Calcula os comprimentos dos canais
    lengths = edge_lengths(Xno, conec)

    # --------------------------------------------------------
    # 4) RESOLVE A REDE
    # --------------------------------------------------------

    # Resolve as pressões e vazões da rede
    p, q, A, D = solve_network(
        conec=conec,
        C=C,
        natm=outlet_node,
        nB=inlet_node,
        QB=Qin,
        patm=patm,
    )

    # --------------------------------------------------------
    # 5) PÓS-PROCESSAMENTO
    # --------------------------------------------------------

    # Calcula queda de pressão em cada canal
    dp = compute_pressure_drops(p, conec)

    # Calcula potência hidráulica
    power = compute_pump_power(p, C, D)

    # Verifica conservação de massa
    mass_residual = check_mass_conservation(
        conec=conec,
        q=q,
        injection_node=inlet_node,
        injection_flow=Qin,
    )

    # --------------------------------------------------------
    # 6) IMPRESSÃO DOS RESULTADOS
    # --------------------------------------------------------

    print(f"Diâmetro hidráulico equivalente [m]: {D_h:.6e}")
    print(f"kappa [m^5/(Pa.s)]: {kappa:.6e}")

    print_summary(
        Xno=Xno,
        conec=conec,
        lengths=lengths,
        C=C,
        p=p,
        q=q,
        dp=dp,
        power=power,
        mass_residual=mass_residual,
        inlet_node=inlet_node,
        outlet_node=outlet_node,
    )

    # --------------------------------------------------------
    # 7) PLOT DA REDE
    # --------------------------------------------------------

    # Gera a figura da rede com pressões e setas de vazão
    fig, ax = PlotaRede(conec, Xno, p, q, factor_units=mm_to_m)

    # Título do gráfico
    plt.title(f"Rede hidráulica - levels = {levels}")

    # Exibe o gráfico
    plt.show()


# ============================================================
# EXECUÇÃO DO PROGRAMA
# ============================================================

# Esse trecho garante que o programa só roda quando este arquivo
# for executado diretamente.
if __name__ == "__main__":
    main()