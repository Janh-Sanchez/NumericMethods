def take_data(amount_of_x: int):
    x = []
    f_x = []
    derivate = []
    for k in range(amount_of_x):
        x_value = float(input("Enter X: "))
        x.append(x_value)
        f_x_value = float(input("Enter F(x): "))
        f_x.append(f_x_value)
        derivate_value = float(input("Enter F'(x): "))
        derivate.append(derivate_value)
    return x, f_x, derivate


def calc_z_k(amount_of_x, x, f_x):
    z_k = []
    f_z_k = []
    for k in range(amount_of_x):
        z_k.append(x[k])
        z_k.append(x[k])
        f_z_k.append(f_x[k])
        f_z_k.append(f_x[k])
    return z_k, f_z_k


def build_hermite_table(z_k, f_z_k, derivate):
    n = len(z_k)
    Q = [[0.0 for row in range(n)] for col in range(n)]

    # Primera columna: f(x)
    for i in range(n):
        Q[i][0] = f_z_k[i]

    # Segunda columna: derivadas si son duplicados
    # Si dos posiciones seguidas son iguales entonces es una derivada
    for i in range(1, n):
        if z_k[i] == z_k[i - 1]:
            Q[i][1] = derivate[i // 2]
    # Si no, se utiliza la formula de diferencias divididas
        else:
            Q[i][1] = (Q[i][0] - Q[i - 1][0]) / (z_k[i] - z_k[i - 1])

    # Resto de columnas
    for j in range(2, n):
        for i in range(j, n):
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (z_k[i] - z_k[i - j])

    return Q

def print_table(Q):
    for row in Q:
        print(["{:.2f}".format(value) for value in row])


def print_hermite_formula(z_k, Q):
    n = len(z_k)
    formula = f"H(x) = {Q[0][0]:.4f}"

    for i in range(1, n):
        coef = Q[i][i]
        term = f"{' + ' if coef >= 0 else ' - '}{abs(coef):.4f}"
        for j in range(i):
            term += f"(x - ({z_k[j]:.4f}))"
        formula += term

    print("\nPolinomio de Hermite (forma de Newton):")
    print(formula)



# Programa principal
amount_of_x = int(input("Enter the amount of x to use: "))
while amount_of_x > 10 or amount_of_x < 2:
    print("The amount of X can't be more than ten or less than 2, Try again")
    amount_of_x = int(input("Enter the amount of x to use: "))

x, f_x, derivate = take_data(amount_of_x)
z_k, f_z_k = calc_z_k(amount_of_x, x, f_x)

Q = build_hermite_table(z_k, f_z_k, derivate)

print("Tabla de diferencias divididas de Hermite:")
print_table(Q)

print_hermite_formula(z_k, Q)

