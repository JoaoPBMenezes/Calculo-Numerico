
#--------------------------------------------------------------------------------------------

#def compute_power
#Calcula a potencia total dissipada pelo atrito do fluido nos canais
#atraves de um produto Matricial  p^t*A*p, no qual a Matriz A corresponde A: A=D^t*K*D (Matriz de Condutancia Global)
#Entradas: p->Vetor com as pressoes calculadas em cada no, D->Matriz de incidencia que mostra como os nos estao conectados, 
#K->Matriz de condutancia(relfete o quao dificil e deixar o fluido passar em cada canal)
#Saidas: Um numero real(float) indicando a potencia hidraulica total.

def compute_power(p: np.ndarray, D: np.ndarray, K: np.ndarray) -> float:
    # O termo (D.T @ K @ D) calcula a matriz de condutancia da rede.
    # O produto p.T @ Matriz @ p calcula a forma quadratica que resulta na dissipacao de energia.
    return float(p.T @ (D.T @ K @ D) @ p)

#def nodal_mass_residual
#Essa funcao serve para a verificacao da consistencia do modelo implementado atraves da verificao da conservacao da Lei das massas,
#assim ela faz a somatoria das vazoes de cada no, e esse valor tem que ser igual ou aproximadamente igual a 0. E o "balanco de massas do sistema" se for positivo ha, 
#um acumulo naquele no, se negativo perda se zero sistema estara em equilibrio
#Entradas: conec->Matriz com as conexoes dos canos do sistema, q->Vetor com as vazoes em cada canal, 
#imposed_b->Os vetores com as vazoes forcadas externamente na rede
#Saidas: residual-> Um vetor, em que cada posicao sua representa a somatoria das vazoes de cada no. 
#Se o sistema foi montado coerentemente, e para todas as posicoes desse vetor serem proximas de zero

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

    #Desconsidera as vazoes aplicadas externamente no sistema, verificando se o mesmo fecha em zero
    residual -= imposed_b 
    return residual

#def print_inputs_summary
#Essa funcao serve para, a fim de uma facil visibilidade dos dados que vao ser insiridos no sistema para simulacao
# Ela garante que o usuario visualize as unidades no SI antes de analisar os resultados,prevenindo erros de interpretacao de escala.
#Entradas: cfg->O dicionario de configuracao dos dados utilizados no sistema, hydraulic_data-> Um dicionario para os dados calculados(viscosidade e comprimentos), 
#Xno-> Coordenadas dos nos e conec-> e lista de conexoes do sistema.
#Saidas: -
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

#def print__output_summary
#Essa funcao serve para, a fim de uma facil visibilidade dos dados, da resolucao da rede, serem mostrados para o usuario.
## O parametro 'result' centraliza o estado final da rede (encapsulamento), facilitando o acesso as matrizes de incidencia D e condutancia K.
#Entradas: cfg->O dicionario de configuracao, hydraulic_data-> Um dicionario para os dados calculados(viscosidade e comprimentos), 
#result-> Um dicionario para as respostas dos sistema e conec-> e lista de conexoes da rede.
#Saidas: -
def print_output_summary(cfg: dict, hydraulic_data: dict, result: dict, conec: np.ndarray) -> None:
    # Pega os valores do dicionario result e coloca em variaveis locais
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

    #Chamando as funcoes compute_power(calcula a potencia total do atrito do fluido nos canais) e residual(verifica a conservacao de massa do sistema)
    power = compute_power(p, D, K)
    residual = nodal_mass_residual(conec, q, b)

    print()
    print("=" * 88)
    print("SAÍDAS DA SIMULAÇÃO")
    print("=" * 88)

    #Imprime as pressoes em todos os nos do sistema(pelo menos 6 casas decimais)
    print("Pressoes nodais [Pa]:")
    for i, pi in enumerate(p):
        print(f"  nó {i}: {pi:.6e}")

    print()
    #Imprime as vazoes nos ultimo instante
    print("Vazões impostas avaliadas no ultimo instante [m^3/s]:")
    for i, bi in enumerate(b):
        if abs(bi) > 0:
            print(f"  nó {i}: {bi:.6e}")

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

    #Printa a potencia hidraulica
    print()
    print(f"Potencia hidraulica [W]: {power:.6e}")

    #Printa a conservacao de massa em cada no
    print()
    print("Residuo de conservacao de massa por no:")
    for i, ri in enumerate(residual):
        print(f"  nó {i}: {ri:.6e}")

    print("=" * 88)
#-----------------------------------------------------------------------------------------------

