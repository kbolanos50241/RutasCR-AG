"""
RutaTica-AG
===========
Algoritmo genético para el problema del viajero (TSP) aplicado a 20 ciudades
de Costa Rica.

Un repartidor sale de San José, visita cada ciudad una sola vez y regresa.
El algoritmo busca el orden de visita que minimice la distancia total.

Curso: Paradigmas de la programación
Carrera: Ingeniería en Sistemas - Universidad Fidélitas

Uso:
    python rutatica_ag.py

Genera en la carpeta actual:
    ruta_final.png, convergencia.png, comparacion.png
"""

import math
import random
import statistics

import matplotlib

matplotlib.use("Agg")  # permite guardar gráficos sin abrir ventanas
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. DATOS DEL PROBLEMA
# ----------------------------------------------------------------------
# Coordenadas aproximadas (latitud, longitud) en grados decimales.
CIUDADES = {
    "San José": (9.9281, -84.0907),
    "Alajuela": (10.0162, -84.2142),
    "Cartago": (9.8644, -83.9194),
    "Heredia": (9.9981, -84.1170),
    "Liberia": (10.6346, -85.4407),
    "Puntarenas": (9.9763, -84.8384),
    "Limón": (9.9907, -83.0360),
    "San Ramón": (10.0877, -84.4700),
    "Ciudad Quesada": (10.3241, -84.4287),
    "San Isidro (P. Zeledón)": (9.3739, -83.7046),
    "Golfito": (8.6394, -83.1647),
    "Nicoya": (10.1483, -85.4520),
    "Santa Cruz": (10.2610, -85.5879),
    "Cañas": (10.4300, -85.0950),
    "Turrialba": (9.9040, -83.6820),
    "Guápiles": (10.2150, -83.7870),
    "Jacó": (9.6150, -84.6290),
    "Quepos": (9.4310, -84.1620),
    "Ciudad Neily": (8.6500, -82.9500),
    "Siquirres": (10.1000, -83.5100),
}

NOMBRES = list(CIUDADES)
N = len(NOMBRES)
RADIO_TIERRA_KM = 6371.0


def haversine(coord1, coord2):
    """Distancia en línea recta (km) entre dos coordenadas (lat, lon)."""
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * RADIO_TIERRA_KM * math.asin(math.sqrt(a))


# Matriz de distancias precalculada: DIST[i][j] = km entre ciudad i y ciudad j
DIST = [[haversine(CIUDADES[a], CIUDADES[b]) for b in NOMBRES] for a in NOMBRES]

# ----------------------------------------------------------------------
# 2. REPRESENTACIÓN Y FUNCIÓN DE APTITUD
# ----------------------------------------------------------------------
# Cromosoma: lista con una permutación de los índices 0..N-1 (orden de visita).


def distancia_total(ruta):
    """Distancia del recorrido cerrado (incluye el regreso a la ciudad inicial)."""
    return sum(DIST[ruta[i]][ruta[(i + 1) % N]] for i in range(N))


def aptitud(ruta):
    """Aptitud = inverso de la distancia: a menor distancia, mayor aptitud."""
    return 1.0 / distancia_total(ruta)


# ----------------------------------------------------------------------
# 3. OPERADORES GENÉTICOS
# ----------------------------------------------------------------------
def crear_poblacion(tamano):
    """Población inicial: rutas completamente aleatorias."""
    return [random.sample(range(N), N) for _ in range(tamano)]


def seleccion_torneo(poblacion, k):
    """Selecciona k individuos al azar y devuelve el de mayor aptitud."""
    participantes = random.sample(poblacion, k)
    return max(participantes, key=aptitud)


def cruza_orden(padre1, padre2):
    """Cruza de orden (OX1): copia un segmento del padre 1 y completa
    con las ciudades faltantes en el orden en que aparecen en el padre 2."""
    a, b = sorted(random.sample(range(N), 2))
    hijo = [None] * N
    hijo[a : b + 1] = padre1[a : b + 1]
    segmento = set(padre1[a : b + 1])

    # Recorre el padre 2 empezando después del segundo punto de corte
    faltantes = [c for c in padre2[b + 1 :] + padre2[: b + 1] if c not in segmento]
    pos = (b + 1) % N
    for ciudad in faltantes:
        hijo[pos] = ciudad
        pos = (pos + 1) % N
    return hijo


def mutacion_inversion(ruta):
    """Invierte el orden de un subtramo de la ruta."""
    a, b = sorted(random.sample(range(N), 2))
    ruta[a : b + 1] = reversed(ruta[a : b + 1])


def mutacion_intercambio(ruta):
    """Intercambia dos ciudades de posición."""
    a, b = random.sample(range(N), 2)
    ruta[a], ruta[b] = ruta[b], ruta[a]


# ----------------------------------------------------------------------
# 4. ALGORITMO GENÉTICO
# ----------------------------------------------------------------------
def algoritmo_genetico(
    tam_poblacion=150,
    generaciones=300,
    prob_cruza=0.9,
    prob_mutacion=0.2,
    k_torneo=5,
    elitismo=2,
    mutacion="inversion",
    semilla=None,
):
    """Ejecuta el AG y devuelve (mejor_ruta, mejor_distancia, hist_mejor, hist_prom)."""
    if semilla is not None:
        random.seed(semilla)

    operador_mutacion = mutacion_inversion if mutacion == "inversion" else mutacion_intercambio
    poblacion = crear_poblacion(tam_poblacion)
    hist_mejor, hist_prom = [], []

    for _ in range(generaciones):
        # Evaluación y registro de estadísticas de la generación
        poblacion.sort(key=distancia_total)
        distancias = [distancia_total(r) for r in poblacion]
        hist_mejor.append(distancias[0])
        hist_prom.append(sum(distancias) / len(distancias))

        # Elitismo: las mejores rutas pasan intactas a la siguiente generación
        nueva = [r[:] for r in poblacion[:elitismo]]

        # Se completa la población con hijos de padres seleccionados por torneo
        while len(nueva) < tam_poblacion:
            p1 = seleccion_torneo(poblacion, k_torneo)
            p2 = seleccion_torneo(poblacion, k_torneo)
            hijo = cruza_orden(p1, p2) if random.random() < prob_cruza else p1[:]
            if random.random() < prob_mutacion:
                operador_mutacion(hijo)
            nueva.append(hijo)
        poblacion = nueva

    poblacion.sort(key=distancia_total)
    mejor = poblacion[0]
    return mejor, distancia_total(mejor), hist_mejor, hist_prom


# ----------------------------------------------------------------------
# 5. MÉTODOS DE COMPARACIÓN
# ----------------------------------------------------------------------
def ruta_vecino_mas_cercano(inicio=0):
    """Heurística voraz: desde la ciudad actual, ir siempre a la más cercana."""
    ruta = [inicio]
    pendientes = set(range(N)) - {inicio}
    while pendientes:
        actual = ruta[-1]
        siguiente = min(pendientes, key=lambda c: DIST[actual][c])
        ruta.append(siguiente)
        pendientes.remove(siguiente)
    return ruta


def rutas_aleatorias(cantidad=1000):
    """Distancias de `cantidad` rutas generadas al azar."""
    return [distancia_total(random.sample(range(N), N)) for _ in range(cantidad)]


# ----------------------------------------------------------------------
# 6. SALIDAS: TEXTO Y GRÁFICOS
# ----------------------------------------------------------------------
def rotar_desde(ruta, ciudad_inicio=0):
    """Reordena la ruta circular para que comience en la ciudad indicada."""
    i = ruta.index(ciudad_inicio)
    return ruta[i:] + ruta[:i]


def mostrar_ruta(ruta):
    ruta = rotar_desde(ruta)
    return " -> ".join(NOMBRES[c] for c in ruta) + " -> " + NOMBRES[ruta[0]]


def grafico_ruta(ruta, distancia, archivo="ruta_final.png"):
    ruta = rotar_desde(ruta)
    ruta_cerrada = ruta + [ruta[0]]
    lons = [CIUDADES[NOMBRES[c]][1] for c in ruta_cerrada]
    lats = [CIUDADES[NOMBRES[c]][0] for c in ruta_cerrada]

    plt.figure(figsize=(9, 7))
    plt.plot(lons, lats, "-o", color="tab:blue", linewidth=1.5, markersize=6)
    for orden, c in enumerate(ruta, start=1):
        lat, lon = CIUDADES[NOMBRES[c]]
        plt.annotate(f"{orden}. {NOMBRES[c]}", (lon, lat), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
    lat0, lon0 = CIUDADES[NOMBRES[ruta[0]]]
    plt.scatter([lon0], [lat0], color="red", s=90, zorder=5, label="Inicio / fin: San José")
    plt.title(f"Mejor ruta encontrada por el AG ({distancia:.1f} km)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


def grafico_convergencia(hist_mejor, hist_prom, archivo="convergencia.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(hist_prom, label="Distancia promedio de la población", color="tab:orange")
    plt.plot(hist_mejor, label="Mejor distancia", color="tab:blue")
    plt.title("Convergencia del algoritmo genético")
    plt.xlabel("Generación")
    plt.ylabel("Distancia total (km)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


def grafico_comparacion(etiquetas, valores, archivo="comparacion.png"):
    plt.figure(figsize=(8, 5))
    barras = plt.bar(etiquetas, valores, color=["gray", "tab:orange", "tab:blue"])
    for barra, v in zip(barras, valores):
        plt.text(barra.get_x() + barra.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")
    plt.title("Comparación de métodos (distancia total en km)")
    plt.ylabel("Distancia total (km)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# 7. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RutaTica-AG: TSP con 20 ciudades de Costa Rica")
    print("=" * 60)

    # --- Ejecución principal (semilla fija para poder reproducirla) ---
    mejor, dist, hist_mejor, hist_prom = algoritmo_genetico(semilla=42)
    print(f"\n[1] Ejecución principal (semilla 42)")
    print(f"Distancia de la mejor ruta: {dist:.2f} km")
    print(f"Ruta: {mostrar_ruta(mejor)}")
    grafico_ruta(mejor, dist)
    grafico_convergencia(hist_mejor, hist_prom)

    # --- Comparación con ruta aleatoria y vecino más cercano ---
    random.seed(1)
    aleatorias = rutas_aleatorias(1000)
    nn = distancia_total(ruta_vecino_mas_cercano())
    print(f"\n[2] Comparación con otros métodos")
    print(f"Rutas aleatorias (1000): promedio {statistics.mean(aleatorias):.2f} km, "
          f"mejor {min(aleatorias):.2f} km")
    print(f"Vecino más cercano:      {nn:.2f} km")
    print(f"Algoritmo genético:      {dist:.2f} km")
    grafico_comparacion(
        ["Aleatoria\n(promedio)", "Vecino más\ncercano", "Algoritmo\ngenético"],
        [statistics.mean(aleatorias), nn, dist],
    )

    # --- Repetición con distintas semillas (robustez) ---
    print(f"\n[3] 10 ejecuciones con semillas distintas (config. base)")
    resultados = [algoritmo_genetico(semilla=s)[1] for s in range(1, 11)]
    print("Distancias (km):", ", ".join(f"{d:.1f}" for d in resultados))
    print(f"Mejor: {min(resultados):.2f}  Promedio: {statistics.mean(resultados):.2f}  "
          f"Peor: {max(resultados):.2f}  Desv. est.: {statistics.stdev(resultados):.2f}")

    # --- Efecto de variar parámetros (10 semillas por configuración) ---
    print(f"\n[4] Efecto de los parámetros (promedio de 10 ejecuciones)")
    configuraciones = [
        ("Población 50",           dict(tam_poblacion=50)),
        ("Población 150 (base)",   dict(tam_poblacion=150)),
        ("Población 300",          dict(tam_poblacion=300)),
        ("Mutación 0.05",          dict(prob_mutacion=0.05)),
        ("Mutación 0.50",          dict(prob_mutacion=0.50)),
        ("Mutación por intercambio", dict(mutacion="intercambio")),
    ]
    print(f"{'Configuración':28s} {'Promedio km':>12s} {'Mejor km':>10s}")
    for nombre, params in configuraciones:
        ds = [algoritmo_genetico(semilla=s, **params)[1] for s in range(1, 11)]
        print(f"{nombre:28s} {statistics.mean(ds):12.2f} {min(ds):10.2f}")

    print("\nGráficos guardados: ruta_final.png, convergencia.png, comparacion.png")


if __name__ == "__main__":
    main()
