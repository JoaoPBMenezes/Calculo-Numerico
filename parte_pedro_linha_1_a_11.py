import numpy as np '#importa uma biblioteca no python para manipular eficientemente os dados e matrizes da rede hidráulica
import matplotlib.pyplot as plt '# importa uma biblioteca para plotar graficos de linhas simples da rede hidráulica

from gera_grafo import GeraGrafo '#Do arquivo chamado gera_grafo, já adicionado, importamos a biblioteca Geragrafo para gerar o grafo da rede hidráulica com os respectivos níveis de ligação indicados no código mais a frente
from plota_rede import PlotaRede '#Do arquivo chamado plota_rede, já adicionado, importamos a biblioteca Plotarede, para gerar a rede hidráulica em si, o esquema com os canos interligados formando a rede propriamente dita


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

'#As convenções utilizadas para cada unidade de medida estão no SI. A vazão que entra no nó é definida como positiva e a vazão que sai do nó é definida como negativa
'#No código, podemos editar essas configurações iniciais para testar novos casos
CONFIG = {
    '#Configurações iniciais e padrão da rede hidráulica, todas essas configurações podem ser alteradas caso queira testar valores distintos de área, temperatura, pressão, nível da rede hidráulica, tempo, entre outras medidas'
    # ---------------------------
    # Geometria da rede
    # ---------------------------
    '#Configurações da geometria da rede'

    "levels": 1,              '#Pode ser editado para testar novos casos, é relacionado ao tamnho da rede, pode aumentar ou diminuir o tamanho da rede'
    "coord_scale_to_m": 1e-3, '#Ajuste das coordenadas fornecidas pela função Geragrafo, que fornece as coordenadas dos nós e das conexões em mm no grafo, mas como está sendo utilizado o SI como convenção (como visto em cima), então aqui esses valores são convertidos para metros, por isso o 1e-3, significa o 10 elevado a -3 em python

    # ---------------------------
    # Modelo temporal
    # ---------------------------
    '#Configurações do regime/modelo de tempo utilizado na rede hidráulica, para calcular vazões, mudanças de pressão com o tempo, entre outras medidas'
    "time_mode": "transient",  '#O regime temporal pode ser escolhido como steady (ou seja, estático ou parado), nesse caso as configurações seriam constantes com o tempo e as medidas não se alterariam, ficam estáveis em qualquer ponto da rede. Ou pode escolher o regime transient (escolhido nesse caso), em que os valores de Vazão e pressão são variáveis com o tempo'
    "t0": 0.0,  '#tempo inicial'
    "tf": 10.0,   '#tempo final'
    "dt": 0.2,    '#Variação de tempo para cada medição de Vazão e pressão, isto é, a cada dt de 0.2 segundos a pressão e a Vazão mudarão e serão medidas '

    # ---------------------------
    # Propriedades físicas dos canais
    # ---------------------------

    '#Agora passamos de falar da rede para as propriedades físicas específicas dos canais (canos), como área e volume por exemplo, essenciais para calcular pressão e Vazão de cada cano da rede hidráulica'
    # geometry_mode: 
    '#dependendo da geometria do modelo do canal o modo de calcular pode ser distinto, mas aqui são os valores da área do cano (se o programa quiser manter uma area fixa, aqui independe da geometria do cano), do diametro do cano (para caso seja um cano redondo/circular), da largura e altura do cano (para caso seja um cano retangular)'
    #   "area"        -> usa area_constant 
    '#se a preferencia for por manter uma area de cano constante para calcular a vazão da rede hidraulica'
    #   "diameter"    -> usa diameter_constant
    '#em geral usado para canos circulares, mantem-se um valor constante de diametro da circunferencia do cano para calcular sua área, vazão e demais propriedades '
    #   "rectangular" -> usa width e height
    '#Usa o comprimento e a largura definidos, em geral para canos retangulares, para cálculo de área'
    "geometry_mode": "rectangular",
'# O modo da geometria utilizado é o retangular, isto é, o cano deve ser retangular'
    "area_constant": 500e-6 * 500e-6,
'# A área foi definida com o valor de 500 vezes 10 elevado a -6 multiplicado por 500 vezes 10 a -6, entretanto, como o modo escolhido foi o retangular, então não será utilizada esssa área definida, mas se mudar o regime para a área constante então esse será o valor de área utilizado, mas que pode ser alterado caso queira'
    "diameter_constant": 800e-6,
    '#O diametro foi definido com o valor de 800 vezes 10 a -6, entretanto, como o modo escolhido foi definido como retangular, então esse valor de diametro não será utilizado neste caso, mas caso o regime mude para circular, o diâmetro já estará definido com esse valor, que pode ser mudado caso queira'
    "width": 500e-6,
    '#largura do retangulo da base do cano, ou seja, largura do cano para o cálculo da área no caso retangular, que será o utilizado nesse exemplo, o valor dele é de 500 vezes 10 elevado a -6'
    "height": 500e-6,
'#O comprimento do retangulo da base do cano, ou seja, comprimento do cano para o cálculo da área no caso retangular, que será o utilizado nesse exemplo, o valor dele é de 500 vezes 10 elevado a -6'
    # Se quiser propriedades diferentes por canal, coloque aqui.
    '#Outras propriedades para além da área, diametro, comprimento e largura podem ser acrescentadas aqui, em forma de vetor'
    # Ex.: [2.5e-7, 2.5e-7, 3.0e-7, ...]
    '#aqui exemplos de valores distintos que podem ser utilizados'
    # Se None, o código usa o modo global acima.
    '#A área por curva, somente se quiser alterar o modo nos cantos do cano para algo diferente do já utilizado, nesse caso, o None indica que o modo geométrico dos cantos do cano é o mesmo do modo global já definido acima, que no caso foi o retangular, sendo assim, os cantos do cano tambem tem formato retangular'
    "area_per_edge": None,

    # ---------------------------
    # Temperatura e viscosidade
    # ---------------------------
    "temperature_celsius": 25.0,
'#A temperatura está em celsius e foi predeterminada com um valor escolhido de 25.0'
    '#É possível colocar aqui um valor forçado para a viscosidade, para deixá-la predeterminada, evidentemente que nesse caso algumas das variáveis do problema já estarão predeterminadas e, dependendo da fórmula utilizada para relacionar temperatura e viscosidade pode ocorrer algum problema ou discrepancia, mas, no geral, é possível forçar um valor de viscosidade, pois nem sempre será utilizada uma fórmula'
    # Se quiser FORÇAR a viscosidade, coloque um valor aqui.
    # Se None, a viscosidade será calculada pela temperatura.
'#Aqui pode colocar um regime de viscosidade de preferencia, se estiver escrito None então a viscosidade será calculada a partir da temperatura predeterminada'
    "mu_override": None,

    # ---------------------------
    # Pressões prescritas
    # ---------------------------
    # Exemplo:
    # nó 5 em 0 Pa
    # nó 0 em 1500 Pa
    #
    # Você pode colocar quantos nós quiser.
    '#Aqui as pressões em cada nó podem ser predeterminadas e definidas, para quantos nós quiser'
    "pressure_bc": {
        5: 0.0,
        10: 0.1,
    },
'#A pressão no cano bc foi definida pelas pressões nos nós, ou seja, a struct de pressão no cano bc é a pressão em cada nó que liga para gerar o cano, nesse caso, do cano bc, são os nós 5 e 10, os quais possuem valores de pressão, nesse caso, definidos como 0.0 e 0.1'
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
    '#A struct da vazão é definida parecida com a pressão, a vazão em bc foi definida pelos nós 0 e 3'
    '#O formato das vazões em cada nó é primeiro explicitado pelo tipo da vazão, que pode ser constante, senoidal ou cossenoidal. No caso constante define-se o valor para essa pressão, no exemplo acima no nó 0 o valor é de 1 vezes 10 elevado a -7. No tipo senoide coloca type:sin e explicita-se o valor médio da função seno que descreve a vazão, a amplitude dessa função, a frequencia e a fase inicial da onda senoide, assim definindo unicamente a onda senoide característica. No nó 3 o tipo definido é seno, o valor médio é de 1 vezes 10 elevado a -7, a amplitude é de 2 vezes 10 elevado a -8, a frequencia é de 0.5 e a fase inicial é 0.0. Para o caso do nó 4, que é cosseno, é análogo ao seno'
    "flow_bc": {
        0: {"type": "sin", "mean": 1.0e-7, "amp": 0.3e-7, "freq": 0.5, "phase": 0.0},
        3: {"type": "cos", "mean": -0.2e-7, "amp": 0.1e-7, "freq": 0.25, "phase": 0.0},
    },

    # ---------------------------
    # Saídas gráficas
    # ---------------------------
    "plot_network": True,
    '#Aqui tem a saída da rede, plotada num gráfico'
    "plot_time_series": True,
    '#Aqui tem a saída da série de tempo, ou seja, plota o gráfico em função do tempo, ambas dos arquivos externos importados'

    # Escolha quais nós e arestas quer acompanhar no tempo
    "nodes_to_plot": [0, 2, 5],
    "edges_to_plot": [0, 1, 2],
'#Aqui escolhe quais cantos e quais nós serão acompanhados ao longo do tempo e, consequentemente, plotados na função chamada acima plot_time_series do arquivo externo importado'
    # Mostra tabela detalhada dos canais no terminal
    "print_edge_table": True,
    '#Aqui printa a tabela com os valores das pressões nos cantos'
}
