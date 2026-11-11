# RutaTica-AG

Algoritmo genético en Python para resolver el **problema del viajero (TSP)** con 20 ciudades de Costa Rica.
Un repartidor sale de San José, visita cada ciudad una sola vez y regresa; el programa busca el orden de visita que minimiza la distancia total.

Proyecto del curso **Paradigmas de la programación** (Ingeniería en Sistemas, Universidad Fidélitas) – Caso práctico #2.

## Diseño del algoritmo

| Elemento | Decisión |
|---|---|
| Cromosoma | Permutación de las 20 ciudades (orden de visita) |
| Aptitud | 1 / distancia total del recorrido cerrado |
| Selección | Torneo (k = 5) |
| Cruza | Cruza de orden OX1 (probabilidad 0.9) |
| Mutación | Inversión de un subtramo (probabilidad 0.2) |
| Elitismo | Las 2 mejores rutas pasan intactas |
| Población / generaciones | 150 / 300 |

Las distancias se calculan en línea recta con la fórmula de Haversine a partir de coordenadas aproximadas (no son distancias por carretera).

## Requisitos

- Python 3.8 o superior
- matplotlib

## Instalación y ejecución

```bash
pip install -r requirements.txt
python rutatica_ag.py
```

El programa imprime los resultados en consola y genera `ruta_final.png`, `convergencia.png` y `comparacion.png`.
Las ejecuciones usan semillas fijas, por lo que los resultados son reproducibles.

## Archivos

- `rutatica_ag.py`: código fuente completo
- `resultados.txt`: salida de la ejecución
- `ruta_final.png`, `convergencia.png`, `comparacion.png`: gráficos generados
