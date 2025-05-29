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

# Print Hermite divided differences table
def print_table(Q):
    for row in Q:
        print(["{:.4f}".format(value) for value in row])

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

# Construct natural cubic spline interpolation
def cubic_spline_natural(x, f_x):
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n - 1)]
    alpha = [0] + [3 * ((f_x[i+1] - f_x[i]) / h[i] - (f_x[i] - f_x[i-1]) / h[i-1]) for i in range(1, n - 1)]

    l = [1] + [0] * (n - 1)
    mu = [0] * n
    z = [0] * n

    for i in range(1, n - 1):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n - 1] = 1
    z[n - 1] = 0

    M = [0] * n
    for j in reversed(range(n - 1)):
        M[j] = z[j] - mu[j] * M[j + 1]

    splines = []
    for i in range(n - 1):
        def spline_factory(i=i):
            def S(x_val):
                hi = h[i]
                xi, xi1 = x[i], x[i+1]
                fi, fi1 = f_x[i], f_x[i+1]
                Mi, Mi1 = M[i], M[i+1]

                t1 = Mi * (xi1 - x_val)**3 / (6 * hi)
                t2 = Mi1 * (x_val - xi)**3 / (6 * hi)
                t3 = (fi / hi - Mi * hi / 6) * (xi1 - x_val)
                t4 = (fi1 / hi - Mi1 * hi / 6) * (x_val - xi)
                return t1 + t2 + t3 + t4
            return S
        splines.append(spline_factory())
    return splines, x

# Plot Hermite and spline curves with optional zoom on region of interest
def graph_hermite_and_splines(x, f_x, z_k, Q, splines, nodos):
    x_min = min(x)
    x_max = max(x)
    x_vals_hermite = np.linspace(x_min, x_max, 500)
    y_vals_hermite = [hermite_polynomial(x_val, z_k, Q) for x_val in x_vals_hermite]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals_hermite, y_vals_hermite, label='Hermite Polynomial', color='blue')

    x_vals_spline = []
    y_vals_spline = []
    for i in range(len(splines)):
        xs = np.linspace(nodos[i], nodos[i + 1], 100)
        ys = [splines[i](xv) for xv in xs]
        x_vals_spline.extend(xs)
        y_vals_spline.extend(ys)

    plt.plot(x_vals_spline, y_vals_spline, label='Cubic Spline', color='green', linestyle='--')
    plt.scatter(x, f_x, color='red', zorder=5, label='Input Points')

    plt.xlim(x_min, x_max)
    all_y_vals = y_vals_hermite + y_vals_spline
    y_min = min(all_y_vals)
    y_max = max(all_y_vals)
    y_margin = 0.05 * (y_max - y_min)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.title("Interpolation: Hermite vs. Cubic Spline (Zoomed Area)")
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

splines, nodos = cubic_spline_natural(x, f_x)
graph_hermite_and_splines(x, f_x, z_k, Q, splines, nodos)