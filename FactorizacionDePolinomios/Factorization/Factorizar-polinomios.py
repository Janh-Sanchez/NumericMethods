from sympy import symbols, sympify, factor, init_printing

def main():
    init_printing(use_unicode=True)  # Para mostrar resultados bonitos si se usa en Jupyter o similar
    print("=== Factorizador simbólico de Polinomios ===")
    
    x = symbols('x')  # variable principal
    entrada = input("Ingresa un polinomio en terminos de x: ")

    try:
        polinomio = sympify(entrada)
        print("\nPolinomio original:")
        print(polinomio)

        factorizado = factor(polinomio)

        print("\nFactorización simbólica máxima: ")
        print(factorizado)
    except Exception as e:
        print("Ocurrió un error al procesar el polinomio: ", e)

if __name__ == "__main__":
    main()
