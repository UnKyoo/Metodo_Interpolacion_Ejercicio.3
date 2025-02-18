# Código que implementa el esquema numérico 
# de interpolación para determinar la raíz de
# una ecuación
#
#            Autor:
# Gilbert Alexander Mendez Cervera
# Versión 1.01 : 17/02/2025
#
import numpy as np  # Se importa la librería NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Se importa Matplotlib para graficar

# Nueva función f(x) = e^(-x) - x
def f(x):
    return np.exp(-x) - x  # Se define la nueva función cuya raíz queremos encontrar

# Interpolación de Lagrange
def lagrange_interpolation(x, x_puntos, y_puntos):
    P_Interpolacion = len(x_puntos)  # Corrección del error en la variable
    Resultado = 0  # Se inicializa el resultado en 0
    for Iteracion_i in range(P_Interpolacion):  # Se recorre cada punto de interpolación
        termino_Lag = y_puntos[Iteracion_i]  # Se inicializa el término de Lagrange
        for Iteracion_j in range(P_Interpolacion):  # Se calcula el producto en la fórmula de Lagrange
            if Iteracion_i != Iteracion_j:  # Se omite cuando i == j
                termino_Lag *= (x - x_puntos[Iteracion_j]) / (x_puntos[Iteracion_i] - x_puntos[Iteracion_j])  # Aplicación de la fórmula de Lagrange
        Resultado += termino_Lag  # Se acumula el resultado final del polinomio interpolante
    return Resultado  # Se retorna el valor interpolado

# Método de Bisección para encontrar la raíz de una función
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:  # Se verifica si hay un cambio de signo en el intervalo
        raise ValueError("El intervalo no contiene una raíz")  # Se lanza un error si no hay una raíz en el intervalo

    for _ in range(max_iter):  # Se itera hasta el máximo permitido
        c = (a + b) / 2  # Se calcula el punto medio del intervalo
        if abs(func(c)) < tol or (b - a) / 2 < tol:  # Se verifica si la aproximación cumple con la tolerancia
            return c  # Se devuelve la raíz aproximada
        if func(a) * func(c) < 0:  # Si hay un cambio de signo en [a, c], se ajusta el intervalo
            b = c
        else:  # Si no, la raíz está en [c, b]
            a = c
    return (a + b) / 2  # Se retorna la mejor estimación de la raíz después de iterar

# Selección de cuatro puntos de interpolación en el intervalo [0,1]
x0 = 0.0  # Primer punto
x1 = 0.25  # Segundo punto
x2 = 0.5   # Tercer punto
x3 = 1.0   # Cuarto punto
x_points = np.array([x0, x1, x2, x3])  # Se almacenan los puntos de x en un array
y_points = f(x_points)  # Se evalúa la función en los puntos seleccionados

# Construcción del polinomio interpolante
x_vals = np.linspace(x0, x3, 100)  # Se generan 100 puntos entre x0 y x3 para graficar
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]  # Se evalúa la interpolación en los puntos generados

# Encontrar raíz del polinomio interpolante usando bisección
root = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x3)  # Se busca la raíz del polinomio interpolante en el intervalo dado

# Calcular errores
errores_absolutos = np.abs(y_interp - f(x_vals))  # Se calcula el error absoluto
errores_relativos = errores_absolutos / np.where(np.abs(f(x_vals)) == 0, 1, np.abs(f(x_vals)))  # Se calcula el error relativo evitando división por cero
errores_cuadraticos = errores_absolutos**2  # Se calcula el error cuadrático

# Encabezado de la tabla
print(f"{'Iteración':<10}|{'x':<12}|{'Error absoluto':<18}|{'Error relativo':<18}|{'Error cuadrático'}")
print("-" * 80)

# Iterar sobre los valores calculados
for i, (x_val, error_abs, error_rel, error_cuad) in enumerate(zip(x_vals, errores_absolutos, errores_relativos, errores_cuadraticos)):  
    # Se imprime la información en formato de tabla
    print(f"{i+1:<10}|{x_val:<12.6f}|{error_abs:<18.6e}|{error_rel:<18.6e}|{error_cuad:.6e}")

# Se genera la gráfica de la función, interpolación y errores
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Subgráfica 1: Errores
ax[0].plot(x_vals, errores_absolutos, label="Error Absoluto", color='purple')
ax[0].plot(x_vals, errores_relativos, label="Error Relativo", color='orange')
ax[0].plot(x_vals, errores_cuadraticos, label="Error Cuadrático", color='brown')
ax[0].set_xlabel("x")
ax[0].set_ylabel("Errores")
ax[0].legend()
ax[0].grid(True)

# Subgráfica 2: Función y interpolación
ax[1].plot(x_vals, f(x_vals), label="f(x) = e^(-x) - x", linestyle='dashed', color='blue')  # Se grafica la función original
ax[1].plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')  # Se grafica la interpolación
ax[1].axhline(0, color='black', linewidth=0.5, linestyle='--')  # Línea horizontal en y = 0
ax[1].axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")  # Se marca la raíz encontrada
ax[1].scatter(x_points, y_points, color='black', label="Puntos de interpolación")  # Se marcan los puntos de interpolación
ax[1].set_xlabel("x")
ax[1].set_ylabel("f(x)")
ax[1].legend()
ax[1].grid(True)

# Guardar la gráfica en un archivo
plt.savefig("interpolacion_raices.png")  
plt.show()  # Mostrar la gráfica

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.4f}")

