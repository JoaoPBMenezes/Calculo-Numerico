import numpy as np

def create_B(list,nn):
    #essa funçao gera um vetor de tamanho nn contendo a vazão em cada ponto
    B = np.zeros(shape=(nn))
    for a in range(nn):    
        B[list[a][0]]= list[a][1]
    return B
def create_C_matrix(nn, C_list):
    #C_list é uma 'lista' nc (numero de canos) por 3.
    #Cada linha representa um cano. [0] da linha é o no inicial, [1] é no final e [2] o valor do cano
    #lista com os C dos canos diferentes. O item C_list[i][j] se refere ao cano conectando os nos i e j
    #caso não exista cano conectando nos i e j, retornará zero. caso i = j, retornara zero
    C_matrix = np.zeros(shape=(nn,nn))
    for k in range(len(C_list)):
        C_matrix[C_list[k][0]][C_list[k][1]] = C_list[k][2]
        C_matrix[C_list[k][1]][C_list[k][0]] = C_list[k][2]
    return C_matrix


def soma_matrix_no(C_matrix, no):
    #soma de todos os Cs correspondentes aos canos conectados no nó 'no'
    soma = 0
    n_size = len(C_matrix)
    for a in range(n_size):
        soma += C_matrix[a][no]
    return soma

def Assembly(C_matrix, nn):
    #monta a matriz correspondente ao sistema
    C_assembly = np.zeros(shape=(nn,nn))
    for i in range(nn):
        for j in range(nn):
            if (i == j):
                C_assembly[i][i] = soma_matrix_no(C_matrix,i)
            else:
                C_assembly[i][j] = -1*C_matrix[i][j]
    return C_assembly

def mod_atm(C_assembly,no):
    #dentro da matriz C_assembly correspondente ao sistema, altera o valor da linha/coluna correspondentes ao nó 'no' que esta conectado a atmosfera.
    n = len(C_assembly)
    for c in range(n):
        if c == no:
            C_assembly[no][no] = 1
        else:
            C_assembly[no][c] = 0
    return C_assembly

def solve(C_assembly,B):
    #resolve o sistema utilizando ferramentas do numpy e retorna um vetor contendo o valor da pressão em cada nó.
    n = len(C_assembly)
    
    pressure = np.linalg.solve(C_assembly, B)
    return pressure






