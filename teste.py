import arquivo_funcoes as af
list0 = [0,3]
B = af.create_B(list0, 5)
list = [[0,1,2],[1,2,2],[2,3,1],[3,4,2],[1,4,1],[2,4,2],[0,4,2]]
for a in range(5):
    print(a, af.solve(af.mod_atm(af.Assembly(af.create_C_matrix(5,list), 5),2), B)[a])
