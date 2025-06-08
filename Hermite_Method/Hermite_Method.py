import matplotlib.pyplot as plt
import numpy as np

# Collect data: x values, f(x), and derivatives
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


# Generate repeated nodes and values for Hermite interpolation
def calc_z_k(amount_of_x, x, f_x):
    z_k = []
    f_z_k = []
    for k in range(amount_of_x):
        z_k.append(x[k])
        z_k.append(x[k])
        f_z_k.append(f_x[k])
        f_z_k.append(f_x[k])
    return z_k, f_z_k


# Build divided difference table for Hermite interpolation
def build_hermite_table(z_k, f_z_k, derivate):
    n = len(z_k)
    Q = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        Q[i][0] = f_z_k[i]

    for i in range(1, n):
        if z_k[i] == z_k[i - 1]:
            Q[i][1] = derivate[i // 2]
        else:
            Q[i][1] = (Q[i][0] - Q[i - 1][0]) / (z_k[i] - z_k[i - 1])

    for j in range(2, n):
        for i in range(j, n):
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (z_k[i] - z_k[i - j])

    return Q


def getPolynomsSplines(xp, Q):
    coefficientsP = []
    for i in range(len(xp) - 1):
        b0 = float(Q[2 * i][0])
        b1 = float(Q[2 * i + 1][1])  # Derivada en xi
        b2 = float(Q[2 * i + 2][2])
        b3 = float(Q[2 * i + 3][3])

        coefficientsP.append((b0, b1, b2, b3))

        print(f"---")
        print(f"P{i}(x) en [{xp[i]:.7f}, {xp[i+1]:.7f}]:")
        b1Str = f"{b1:.7f}(x - {xp[i]:.7f})"
        b2Str = f"{b2:.7f}(x - {xp[i]:.7f})^2"
        b3Str = f"{b3:.7f}(x - {xp[i]:.7f})^2(x - {xp[i+1]:.7f})"
        print(f"P{i}(x) = {b0:.7f} + {b1Str} + {b2Str} + {b3Str}")

        # Forma simplificada:
        A = b3
        B = b2 - 2 * b3 * xp[i] - b3 * xp[i + 1]
        C = b1 - 2 * b2 * xp[i] + b3 * xp[i] ** 2 + 2 * b3 * xp[i] * xp[i + 1]
        D = b0 - b1 * xp[i] + b2 * xp[i] ** 2 - b3 * xp[i] ** 2 * xp[i + 1]

        simplified_poly_parts = []
        if abs(A) > 1e-9:
            simplified_poly_parts.append(f"{A:.7f}x^3")
        if abs(B) > 1e-9:
            simplified_poly_parts.append(f"{B:+.7f}x^2")
        if abs(C) > 1e-9:
            simplified_poly_parts.append(f"{C:+.7f}x")
        if abs(D) > 1e-9:
            simplified_poly_parts.append(f"{D:+.7f}")

        simplified_poly_str = "".join(simplified_poly_parts)
        if simplified_poly_str.startswith("+"):
            simplified_poly_str = simplified_poly_str[1:]
        elif not simplified_poly_str:
            simplified_poly_str = "0.0000"

        print(f"P{i}(x) (Simplificado) = {simplified_poly_str}\n")

    return coefficientsP


# Print Hermite divided differences table
def print_table(Q):
    for row in Q:
        print(["{:.4f}".format(value) for value in row])


def evaluatePolynomsSplines(valX, xi, xiAux, coefficients):
    b0, b1, b2, b3 = coefficients
    term1 = b0
    term2 = b1 * (valX - xi)
    term3 = b2 * (valX - xi) ** 2
    term4 = b3 * (valX - xi) ** 2 * (valX - xiAux)
    return term1 + term2 + term3 + term4


# Display Hermite polynomial in factored form
def print_hermite_formula(z_k, Q):
    n = len(z_k)
    formula = f"H(x) = {Q[0][0]:.4f}"

    for i in range(1, n):
        coef = Q[i][i]
        term = f"{'+ ' if coef >= 0 else '- '}{abs(coef):.4f}"

        factors = []
        for j in range(i):
            factors.append(f"(x - ({z_k[j]:.4f}))")

        grouped = []
        count = 1
        for j in range(1, len(factors) + 1):
            if j < len(factors) and factors[j] == factors[j - 1]:
                count += 1
            else:
                if count == 1:
                    grouped.append(factors[j - 1])
                else:
                    grouped.append(f"{factors[j - 1]}^{count}")
                count = 1

        term += ''.join(grouped)
        formula += f" {term}"

    print("\nHermite Polynomial (factored form with powers):")
    print(formula)


# Evaluate Hermite polynomial at a given x
def hermite_polynomial(x_value, z_k, Q):
    n = len(z_k)
    result = Q[0][0]
    product_term = 1.0

    for i in range(1, n):
        product_term *= (x_value - z_k[i - 1])
        result += Q[i][i] * product_term

    return result


def graph_hermite_and_splines(x, f_x, z_k, Q, splines_funcs, nodos):
    # Hermite global
    x_min = min(x)
    x_max = max(x)
    x_vals_hermite = np.linspace(x_min, x_max, 500)
    y_vals_hermite = [hermite_polynomial(x_val, z_k, Q) for x_val in x_vals_hermite]

    # Plot setup
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals_hermite, y_vals_hermite, label='Hermite Global', color='blue')

    # Spline Hermite por tramos
    for i, spline_func in enumerate(splines_funcs):
        xs = np.linspace(nodos[i], nodos[i + 1], 100)
        ys = [spline_func(valx) for valx in xs]
        plt.plot(xs, ys, linestyle='--', color='green', label='Spline Hermite' if i == 0 else "")

    # Puntos originales
    plt.scatter(x, f_x, color='red', label='Puntos dados', zorder=5)

    # Límites del gráfico
    all_y_vals = y_vals_hermite
    for spline_func, a, b in zip(splines_funcs, nodos[:-1], nodos[1:]):
        all_y_vals += [spline_func(valx) for valx in np.linspace(a, b, 10)]

    y_min, y_max = min(all_y_vals), max(all_y_vals)
    y_margin = 0.05 * (y_max - y_min)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.xlim(x_min, x_max)

    # Estética
    plt.title("Interpolación: Hermite Global vs. Spline Hermite por tramos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()



# Main program
amount_of_x = int(input("Enter the amount of x to use: "))
while amount_of_x > 10 or amount_of_x < 2:
    print("The amount of x must be between 2 and 10. Try again.")
    amount_of_x = int(input("Enter the amount of x to use: "))

x, f_x, derivate = take_data(amount_of_x)
z_k, f_z_k = calc_z_k(amount_of_x, x, f_x)
Q = build_hermite_table(z_k, f_z_k, derivate)

print("Hermite Divided Differences Table:")
print_table(Q)

print_hermite_formula(z_k, Q)

# Obtener polinomios spline Hermite por tramos
splines_coeffs = getPolynomsSplines(x, Q)

# Crear funciones evaluables lambda para cada tramo
splines_functions = []
for i in range(len(splines_coeffs)):
    coeffs = splines_coeffs[i]
    xi = x[i]
    xiAux = x[i + 1]
    func = lambda valX, c=coeffs, xi=xi, xiAux=xiAux: evaluatePolynomsSplines(valX, xi, xiAux, c)
    splines_functions.append(func)

# Generar gráfica de comparación
graph_hermite_and_splines(x, f_x, z_k, Q, splines_functions, x)
