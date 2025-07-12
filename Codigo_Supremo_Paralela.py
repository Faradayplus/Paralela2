import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime
from collections import defaultdict


# Tamaño de cada bloque (chunk) de datos que se procesará en paralelo.
# Leer el archivo en trozos permite optimizar el uso de memoria y aprovechar el paralelismo.
CHUNK_SIZE = 500_000
CSV_PATH = "eldoria.csv"

def procesar_chunk(chunk):
    """
    Procesamos un fragmento (chunk) del dataset de Eldoria para calcular estadísticas parciales
    necesarias para su análisis en paralelo.

    Parámetros:
    ----------
    chunk : DataFrame
        Subconjunto del dataset original cargado por bloques (por ejemplo, 500.000 filas).

    Retorna:
    -------
    tuple :
        - conteo_estrato (dict): cantidad de personas por estrato social (0–9).
        - edad_stats (list of dict): edad promedio y mediana agrupadas por especie y género.
        - tramos (dict): cantidad de personas por especie, género y tramo etario (0–17, 18–35...).
        - dependencia (tuple): (numerador, denominador) para calcular índice de dependencia.
        - viajes (list): pares de código postal origen-destino con su frecuencia.
    """

    # Homogeneizar nombre de columna 'GÉNERO' → 'GENERO' para evitar errores
    if 'GÉNERO' in chunk.columns:
        chunk.rename(columns={'GÉNERO': 'GENERO'}, inplace=True)

    # Extraer el primer dígito del código postal de origen como estrato social (0–9)
    chunk["estrato"] = chunk["CP ORIGEN"].astype(str).str[0]

    # Calcular edad a partir de la fecha de nacimiento
    chunk["FECHA NACIMIENTO"] = pd.to_datetime(chunk["FECHA NACIMIENTO"], errors="coerce")
    chunk["edad"] = chunk["FECHA NACIMIENTO"].apply(
        lambda x: datetime.now().year - x.year if pd.notnull(x) else None
    )

    # Conteo de personas por estrato
    conteo_estrato = chunk["estrato"].value_counts().to_dict()

    # Calcular edad promedio y mediana por especie y género
    edad_stats = (
        chunk.groupby(["ESPECIE", "GENERO"])["edad"]
        .agg(["mean", "median"])
        .dropna()
        .reset_index()
        .to_dict(orient="records")
    )

    # Función auxiliar para clasificar edad en tramos
    def clasificar_edad(e):
        if pd.isna(e): return None
        if e < 18: return "0-17"
        elif e <= 35: return "18-35"
        elif e <= 60: return "36-60"
        else: return "61+"

    # Asignar tramo etario a cada individuo
    chunk["tramo"] = chunk["edad"].apply(clasificar_edad)

    # Conteo de personas por especie, género y tramo etario
    tramos = chunk.groupby(["ESPECIE", "GENERO", "tramo"]).size().to_dict()

    # Cálculo del índice de dependencia
    menores_15 = chunk[chunk["edad"] < 15].shape[0]
    mayores_64 = chunk[chunk["edad"] > 64].shape[0]
    edad_trabajo = chunk[(chunk["edad"] >= 15) & (chunk["edad"] <= 64)].shape[0]
    dependencia = (menores_15 + mayores_64, edad_trabajo)

    # Conteo de viajes más frecuentes entre CP ORIGEN y DESTINO
    viajes = (
        chunk.groupby(["CP ORIGEN", "CP DESTINO"])
        .size()
        .reset_index(name="conteo")
        .values
        .tolist()
    )

    # Clasificación adicional para pirámide de edades (aunque no se retorna aquí)
    chunk["grupo_edad"] = chunk["edad"].apply(clasificar_grupo_5anios)

    return (conteo_estrato, edad_stats, tramos, dependencia, viajes)


def clasificar_grupo_5anios(edad):
    """
    Clasifica una edad individual dentro de un grupo etario de 5 años.

    Esta función es utilizada para construir la pirámide de edades detallada, agrupando
    las edades en tramos uniformes como '0-4', '5-9', ..., hasta '90+'.

    Parámetros:
    -----------
    edad : float or int
        Edad numérica de la persona. Puede ser None o NaN si la fecha de nacimiento es inválida.

    Retorna:
    --------
    str or None :
        - El grupo etario en formato 'X-Y' (por ejemplo, '10-14').
        - '90+' si la edad es igual o superior a 90 años.
        - None si la edad es NaN o negativa.
    """
    if pd.isna(edad) or edad < 0:
        return None
    elif edad >= 90:
        return "90+"
    else:
        base = int(edad // 5) * 5
        return f"{base}-{base+4}"



def combinar_resultados(resultados):
    """
    Combina los resultados parciales procesados en paralelo (por chunks) y consolida
    las métricas globales requeridas por el análisis censal de Eldoria.

    Esta función recibe una lista de tuplas, donde cada tupla contiene los cálculos
    realizados sobre un fragmento del dataset, y los une en estructuras finales
    agregadas.

    Parámetros:
    -----------
    resultados : list of tuple
        Lista de resultados devueltos por cada worker del Pool. Cada tupla contiene:
            - estrato (dict): conteo por estrato social
            - edades (list of dict): estadísticas de edad (mean y median) por especie y género
            - tramos (dict): conteo por especie, género y tramo etario
            - dependencia (tuple): (numerador, denominador) del índice de dependencia
            - viajes (list of tuples): lista de (CP_ORIGEN, CP_DESTINO, cantidad)

    Retorna:
    --------
    dict :
        Diccionario con los siguientes datos agregados:
            - "conteo_estrato" : dict con cantidad total de personas por estrato social
            - "porcentaje_estrato" : dict con porcentaje de la población por estrato
            - "edad_estadisticas" : lista combinada de todas las estadísticas parciales de edad
            - "tramos" : dict consolidado de tramos etarios por especie y género
            - "indice_dependencia" : float con el valor global del índice de dependencia
            - "top_viajes" : lista de los 10.000 pares de poblados con más viajes (ordenados)
    """
    total_estrato = defaultdict(int)
    edad_registros = []
    tramos_total = defaultdict(int)
    dep_numerador = 0
    dep_denominador = 0
    viajes_total = defaultdict(int)

    # Acumulación de los resultados parciales
    for res in resultados:
        estrato, edades, tramos, dependencia, viajes = res

        for k, v in estrato.items():
            total_estrato[k] += v
        edad_registros.extend(edades)
        for k, v in tramos.items():
            tramos_total[k] += v
        dep_numerador += dependencia[0]
        dep_denominador += dependencia[1]
        for ori, dst, count in viajes:
            viajes_total[(ori, dst)] += count

    # Cálculo del total y porcentaje por estrato
    total_poblacion = sum(total_estrato.values())
    porcentaje_estrato = {k: round(v * 100 / total_poblacion, 2) for k, v in total_estrato.items()}

    # Selección de los 10.000 viajes más frecuentes
    top_10k = sorted(viajes_total.items(), key=lambda x: x[1], reverse=True)[:10000]

    return {
        "conteo_estrato": dict(total_estrato),
        "porcentaje_estrato": porcentaje_estrato,
        "edad_estadisticas": edad_registros,
        "tramos": dict(tramos_total),
        "indice_dependencia": round(dep_numerador / dep_denominador, 3) if dep_denominador > 0 else None,
        "top_viajes": top_10k,
    }


if __name__ == "__main__":
    print("Procesando datos por chunks con paralelismo...\n")
    pool = Pool(processes=cpu_count())
    reader = pd.read_csv(CSV_PATH, sep=";", quotechar='"', chunksize=CHUNK_SIZE, encoding="utf-8")
    resultados = pool.map(procesar_chunk, reader)
    pool.close()
    pool.join()

    final = combinar_resultados(resultados)

    print("\n=== RESULTADOS ===\n")

    print("1. ¿Cuántas personas pertenecen a cada estrato social?")
    for k, v in sorted(final["conteo_estrato"].items()):
        print(f"   - Estrato {k}: {v} personas")

    print("\n2. ¿Qué porcentaje de la población pertenece a cada estrato social?")
    for k, v in sorted(final["porcentaje_estrato"].items()):
        print(f"   - Estrato {k}: {v}%")

    print("\n3. ¿Cuál es la edad promedio según cada especie y género?")
    for row in final["edad_estadisticas"][:10]:
        print(f"   - {row['ESPECIE']} / {row['GENERO']}: Promedio = {round(row['mean'], 2)}")

    print("\n4. ¿Cuál es la edad mediana según cada especie y género?")
    for row in final["edad_estadisticas"][:10]:
        print(f"   - {row['ESPECIE']} / {row['GENERO']}: Mediana = {round(row['median'], 2)}")

    print("\n5. ¿Qué proporción de la población tiene menos de 18 años, entre 18–35, 36–60, más de 60 según especie y género?")
    for k, v in list(final["tramos"].items())[:10]:
        especie, genero, tramo = k
        print(f"   - {especie} / {genero} / {tramo}: {v} personas")

    print("\n6. ¿Cuál es la pirámide de edades de la población según especie, género? ---")
    

    print("\n7. ¿Cuál es el índice de dependencia?")
    print(f"   - Índice de dependencia: {final['indice_dependencia']}")

    print("\n8. ¿Cuáles son los 5 poblados con más viajes (CP ORIGEN -> CP DESTINO)?")
    for ((ori, dst), count) in final["top_viajes"][:5]:
        print(f"   - {ori} -> {dst}: {count} viajes")




    ###                                                                                      ###
    ##  Guardar resultados en CSV "Opcional, Lo hicimos para facilitar el análisis posterior" ##
    ###                                                                                      ###                                     

    pd.DataFrame(list(final["conteo_estrato"].items()), columns=["Estrato", "Cantidad"]) \
        .to_csv("conteo_estrato.csv", index=False)

    pd.DataFrame(list(final["porcentaje_estrato"].items()), columns=["Estrato", "Porcentaje"]) \
        .to_csv("porcentaje_estrato.csv", index=False)

    pd.DataFrame(final["edad_estadisticas"]).to_csv("edad_estadisticas.csv", index=False)

    tramos_df = pd.DataFrame([
        {"ESPECIE": k[0], "GENERO": k[1], "TRAMO": k[2], "CANTIDAD": v}
        for k, v in final["tramos"].items()
    ])
    tramos_df.to_csv("tramos_edad.csv", index=False)

    pd.DataFrame([{
        "numerador (menores 15 + mayores 64)": final["dep_numerador"],
        "denominador (15 a 64)": final["dep_denominador"],
        "indice_dependencia": final["indice_dependencia"]
    }]).to_csv("indice_dependencia.csv", index=False)

    pd.DataFrame([
        {"CP_ORIGEN": ori, "CP_DESTINO": dst, "CANTIDAD": count}
        for (ori, dst), count in final["top_viajes"]
    ]).to_csv("top_viajes.csv", index=False)

    print("\nTodos los resultados fueron guardados en archivos CSV.")