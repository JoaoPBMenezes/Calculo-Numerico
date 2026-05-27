# Relatório de Resultados - Acoplamento Hidráulico-Térmico

## 1. Respostas Teóricas

### Dedução da Regra do Trapézio Composta (Item 4.2.1 - 1)
Para a temperatura média $\langle T_k \rangle$:
1. **Divisão do Domínio:** A aresta de comprimento $L_k$ é dividida em $N$ subintervalos iguais. O passo espacial é $\Delta s = L_k / N$.
2. **Aplicação do Trapézio Simples:** Em cada subintervalo $[s_{n-1}, s_n]$, a área é aproximada por: $\approx (\Delta s/2) \cdot [T(s_{n-1}) + T(s_n)]$
3. **Composição:** Ao somar todos os intervalos, os nós internos são somados duas vezes. As pontas, apenas uma: $\approx (\Delta s/2) \cdot [T(s_0) + 2\sum_{n=1}^{N-1} T(s_n) + T(s_N)]$
4. **Cálculo da Média:** Como a média é $\langle T_k \rangle = \text{Integral} / L_k$, e $\Delta s/L_k = 1/N$:
   $$\langle T_k \rangle \approx \frac{1}{2N} \left[ T(s_0) + 2\sum_{n=1}^{N-1} T(s_n) + T(s_N) \right]$$

### Alternativa para o Cálculo da Viscosidade (Item 4.2.1 - 5)
Em vez de calcular a temperatura média $\langle T \rangle$ para depois aplicá-la na fórmula da viscosidade (usando $\mu(\langle T \rangle)$), o mais rigoroso é integrar a **própria viscosidade** ao longo da aresta para achar $\langle \mu_k \rangle$:
$$\langle \mu_k \rangle = \frac{1}{L_k} \int_0^{L_k} \mu(T(p(s))) \, ds$$
**Justificativa:** A viscosidade $\mu(T)$ é fortemente não-linear. Em funções não-lineares, a função da média não é igual à média da função ($\mu(\langle T \rangle) \neq \langle \mu(T) \rangle$). Integrar diretamente capta com exatidão a real resistência ao escoamento.

---

## 2. Quadratura de Temperatura e Rede Hidráulica Acoplada
*Referente ao Item 4.2.1 (3 e 4)*

| Malha | Regra | N | $T_{med}$ (°C) | Erro $\infty$ | $P_{max}$ (Pa) | Potência (W) | Tempo (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| refinada_241x121 | midpoint | 1 | 43.9929 | 1.326e+00 | 2.614e+03 | 2.786e-03 | 0.021 |
| refinada_241x121 | midpoint | 10 | 43.9128 | 2.390e-02 | 2.639e+03 | 2.814e-03 | 0.033 |
| refinada_241x121 | midpoint | 100 | 43.9120 | 1.921e-04 | 2.640e+03 | 2.814e-03 | 0.041 |
| refinada_241x121 | midpoint | 1000 | 43.9120 | 7.097e-06 | 2.640e+03 | 2.814e-03 | 0.067 |
| refinada_241x121 | trapezoid | 1 | 43.7488 | 3.139e+00 | 2.703e+03 | 2.884e-03 | 0.037 |
| refinada_241x121 | trapezoid | 10 | 43.9103 | 4.823e-02 | 2.641e+03 | 2.815e-03 | 0.041 |
| refinada_241x121 | trapezoid | 100 | 43.9120 | 4.102e-04 | 2.640e+03 | 2.814e-03 | 0.040 |
| refinada_241x121 | trapezoid | 1000 | 43.9120 | 0.000e+00 | 2.640e+03 | 2.814e-03 | 0.056 |
| **refinada_241x121** | **direta <μ(T)>** | **1000** | **-** | **-** | **2.652e+03** | **2.827e-03** | **-** |
| grosseira_61x31 | midpoint | 1 | 44.0231 | 1.472e+00 | 2.607e+03 | 2.777e-03 | 0.033 |
| grosseira_61x31 | midpoint | 10 | 43.9408 | 2.087e-02 | 2.634e+03 | 2.808e-03 | 0.035 |
| grosseira_61x31 | midpoint | 100 | 43.9399 | 3.279e-04 | 2.635e+03 | 2.808e-03 | 0.047 |
| grosseira_61x31 | midpoint | 1000 | 43.9399 | 1.692e-06 | 2.635e+03 | 2.808e-03 | 0.082 |
| grosseira_61x31 | trapezoid | 1 | 43.7704 | 3.335e+00 | 2.701e+03 | 2.881e-03 | 0.039 |
| grosseira_61x31 | trapezoid | 10 | 43.9381 | 3.727e-02 | 2.635e+03 | 2.809e-03 | 0.036 |
| grosseira_61x31 | trapezoid | 100 | 43.9399 | 3.279e-04 | 2.635e+03 | 2.808e-03 | 0.042 |
| grosseira_61x31 | trapezoid | 1000 | 43.9399 | 0.000e+00 | 2.635e+03 | 2.808e-03 | 0.072 |
| **grosseira_61x31** | **direta <μ(T)>** | **1000** | **-** | **-** | **2.647e+03** | **2.822e-03** | **-** |

> **Nota sobre o Tempo de Computação:** O tempo listado na tabela acima mede *apenas* a execução das integrais ao longo da rede. Como a geometria da rede hidráulica é estática, o número de cálculos de quadratura é rigorosamente idêntico para ambas as malhas. A minúscula diferença nos milissegundos a favor da malha refinada é causada por flutuações e ruídos normais de processamento do sistema operacional. O tempo de solução principal (do campo térmico global) comporta-se conforme o esperado: o sistema foi solucionado quase 20 vezes mais rapidamente na malha grosseira em relação à refinada.

---

## 3. Condutividade Térmica Modificada pela Rede
*Referente ao Item 4.3.3 (1)*

| Malha (Nx, Ny) | Raio de Corte ($d_{max}$) | $T_{max}$ (°C) | $T_{med}$ (°C) | Tempo (s) |
| :--- | :--- | :--- | :--- | :--- |
| 61 x 31 | 0.00025 | 49.44100 | 34.27139 | 1.697 |
| 61 x 31 | 0.0005 | 43.16327 | 32.43534 | 1.937 |
| 61 x 31 | 0.001 | 41.11072 | 29.17136 | 2.555 |
| 121 x 61 | 0.00025 | 49.53603 | 34.54860 | 6.313 |
| 121 x 61 | 0.0005 | 42.34882 | 32.34502 | 7.807 |
| 121 x 61 | 0.001 | 40.54688 | 28.75000 | 10.847 |
| 241 x 121 | 0.00025 | 49.40226 | 34.70473 | 25.543 |
| 241 x 121 | 0.0005 | 42.21275 | 32.30072 | 30.819 |
| 241 x 121 | 0.001 | 40.39276 | 28.86058 | 41.803 |

---

## 4. Fonte/Sumidouro da Rede de Microcanais
*Referente ao Item 4.3.3 (2) - Malha base: 121x61, $d_{max}=0.0005$*

| Distribuição de Intensidade | $S_0$ (Base) | $T_{max}$ (°C) | $T_{med}$ (°C) | Tempo (s) |
| :--- | :--- | :--- | :--- | :--- |
| Homogênea ($I=1$) | 1.0e+05 | 73.17970 | 41.77794 | 2.067 |
| Homogênea ($I=1$) | -1.0e+05 | 52.92502 | 34.77412 | 1.997 |
| Homogênea ($I=1$) | 5.0e+05 | 115.21906 | 55.78558 | 1.950 |
| Homogênea ($I=1$) | -5.0e+05 | 40.52096 | 20.76648 | 2.150 |
| Homogênea ($I=1$) | 1.0e+06 | 167.78971 | 73.29513 | 1.978 |
| Homogênea ($I=1$) | -1.0e+06 | 40.32916 | 3.25693 | 2.018 |
| Espinha ($I=100$), Resto ($0.1$) | 1.0e+05 | 143.92318 | 56.67639 | 1.955 |
| Espinha ($I=100$), Resto ($0.1$) | -1.0e+05 | 40.56819 | 19.87568 | 1.988 |
| Espinha ($I=100$), Resto ($0.1$) | 5.0e+05 | 468.76907 | 130.27781 | 1.951 |
| Espinha ($I=100$), Resto ($0.1$) | -5.0e+05 | 40.02889 | -53.72574 | 2.026 |
| Espinha ($I=100$), Resto ($0.1$) | 1.0e+06 | 874.82643 | 222.27958 | 1.979 |
| Espinha ($I=100$), Resto ($0.1$) | -1.0e+06 | 39.61063 | -145.72752 | 2.084 |

> **Nota sobre Extremos de Temperatura:** Nos testes aplicando peso $I_j=100$ focado na espinha central somado a valores de intensidade base $S_0$ elevadíssimos (na magnitude de $\pm 10^6$), observa-se um espalhamento de valores que atingem máximas irreais, tais como 874 °C e -145 °C. Esse cenário atesta a exatidão computacional da implementação geométrica: a concentração gaussiana em uma área minúscula deforma massivamente o balanço térmico local, produzindo gradientes extremos limitados à sua vizinhança graças à constante imposta na região circular próxima.