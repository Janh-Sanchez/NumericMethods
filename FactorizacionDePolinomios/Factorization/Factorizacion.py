from sympy import symbols, sympify, factor, init_printing

def factorizar_polinomio(entrada):
    """
    Toma una cadena que representa un polinomio en términos de x,
    y devuelve el polinomio original junto con su factorización simbólica.
    """
    x = symbols('x')  # Variable principal
    polinomio = sympify(entrada)
    factorizado = factor(polinomio)
    return polinomio, factorizado

# Inicio del programa
init_printing(use_unicode=True)

print("""
#############################################
#####   Descomposicion de polinomios   ######
#############################################
""")

entrada = input("Introduce un polinomio usando 'x' como variable: ")

try:
    polinomio, factorizado = factorizar_polinomio(entrada)

    print("\nTu polinomio original es: ")
    print(f"  {polinomio}")

    print("\nSu factorización es: ")
    print(f"  {factorizado}")
except Exception as e:
    print("¡Ups! Algo salió mal al procesar el polinomio:", e)
