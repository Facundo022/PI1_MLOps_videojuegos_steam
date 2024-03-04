import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity



app = FastAPI()

# Importaion de datos
df_PlayTimeGenre = pd.read_parquet("data/play_time_genres.parquet")
df_UserForGenre_parte_1 = pd.read_parquet("data/user_for_genre_part_1.parquet")
df_UserForGenre_parte_2 = pd.read_parquet("data/user_for_genre_part_2.parquet")
df_UsersRecommend = pd.read_parquet("data/UsersRecommend.parquet")
df_UsersWorstDeveloper = pd.read_parquet("data/UsersWorstDeveloper.parquet")
df_Sentiment_Analysis = pd.read_parquet("data/sentiment_analysis.parquet")


# Primera funcion: PlaytimeGenre

@app.get("/PlayTimeGenre")
def PlayTimeGenre( genero : str ):
    """
    Funcion que devuelve el año con mas horas jugadas para dicho género.
    """
    generos = df_PlayTimeGenre[df_PlayTimeGenre["genres"] == genero] 
    if generos.empty: 
        return f"No se encontraron datos para el género {genero}"
    año_max = generos.loc[generos["playtime_forever"].idxmax()]
    result = {
        'Genero': genero,
        'Año con Más Horas Jugadas': int(año_max["year"]),
        'Total de Horas Jugadas': año_max["playtime_forever"]
    }

    return result

# Segunda funcion: UserForGenre parte 1

@app.get("/UserForGenre_parte1")
def UserForGenre(genero: str):
    """
    Función que devuelve el usuario que acumula más horas jugadas para el género dado 
    y el total de horas jugadas para ese género.
    """
    # Filtrar el DataFrame por el género dado
    generos2 = df_UserForGenre_parte_1[df_UserForGenre_parte_1["genres"] == genero]
    
    # Obtener el usuario con más horas jugadas para ese género
    user_max = generos2.loc[generos2["playtime_forever"].idxmax()]["user_id"]
    
    # Calcular el total de horas jugadas para ese género
    horas_total = generos2["playtime_forever"].sum()

    # Crear el diccionario de resultados
    result = {
        "Genero": genero,
        "Usuario con Más Horas Jugadas": user_max,
        "Total de Horas Jugadas": horas_total
    }
    
    return result

# Segunda funcion: UserForGenre parte 2

@app.get("/UserForGenre_parte2")
def UserForGenre2(genero: str):
    """
    Función que devuelve un DataFrame filtrado por el género dado y las horas jugadas por año para ese género.
    """
    # Filtrar el DataFrame por el género dado
    generos3 = df_UserForGenre_parte_2[df_UserForGenre_parte_2["genres"] == genero]
   
    # Calcular las horas jugadas por año para ese género
    horas_x_año3 = generos3["playtime_forever"].sum()
    
    # Crear el DataFrame con los datos filtrados y las horas jugadas por año
    df_result = pd.DataFrame({
        "Genero": [genero],
        "Total de Horas Jugadas hasta ahora": [horas_x_año3]
    })

    return df_result


# Tercera funcion: UsersRecommend

@app.get("/UsersRecommend")
def UsersRecommend( año : int ):
    """
    Funcion que devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
    """
    df_año= df_UsersRecommend[df_UsersRecommend["anio"]== año]
    if type(año) != int:
        return {"Debes colocar el año en entero, Ejemplo:2012"}
    if año < df_UsersRecommend["anio"].min() or año > df_UsersRecommend["anio"].max():
        return {"Año no encontrado"}
    df_ordenado_recomendacion = df_año.sort_values(by="num_reviews_positivas", ascending=False)
    top_3_juegos = df_ordenado_recomendacion.head(3)[["app_name","num_reviews_positivas"]]
    result3 ={
        "Año": año,
        "Top 3 Juegos Más Recomendados": top_3_juegos.to_dict(orient='records')
    }
    return result3

# Cuarta funcion: UsersWorstDeveloper

@app.get("/UsersWorstDeveloper")
def UsersWorstDeveloper( año : int ):
    """
    Funcion que devuelve el top 3 de desarrolladoras con juegos MENOS 
    recomendados por usuarios para el año dado.
    """
    df_año2 = df_UsersWorstDeveloper[df_UsersWorstDeveloper["anio"]== año]
    if type(año) != int:
        return {"Debes colocar el año en entero, Ejemplo:2012"}
    if año < df_UsersRecommend["anio"].min() or año > df_UsersRecommend["anio"].max():
        return {"Año no encontrado "}
    df_ordenado_recomendacion2 = df_año2.sort_values(by="num_reviews_negativas", ascending=False)
    top_3_developers = df_ordenado_recomendacion2[["developer","num_reviews_negativas"]]
    result4 = {
        'Año': año,
        'Top 3 Desarrolladoras Menos Recomendadas': top_3_developers.to_dict(orient="records")
    }
    return result4


# Quinta funcion : sentiment_analysis

@app.get("/SentimentAnalysis")
def sentiment_analysis( desarrollador : str ):
    """
    Funcion que devuelve un diccionario con el nombre de la desarrolladora como llave y una lista 
    con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con 
    un análisis de sentimiento como valor.
    """
    if type(desarrollador) != str:
        return "Debes colocar un developer de tipo str, EJ:'07th Expansion'"
    if len(desarrollador) == 0:
        return "Debes colocar un developer en tipo String"
    
    # Filtrar el DataFrame por el desarrollador dado
    df_desarrollador = df_Sentiment_Analysis[df_Sentiment_Analysis["developer"] == desarrollador]
    
    # Sumar el número total de reseñas negativas, neutrales y positivas
    reseñas_negativas = df_desarrollador["cant_sentiment_negative"].sum()
    reseñas_neutrales = df_desarrollador["cant_sentiment_neutral"].sum()
    reseñas_positivas = df_desarrollador["cant_sentiment_positive"].sum()

    # Crear el diccionario de resultados
    resultados = {
        "El desarrollador: ": desarrollador,
        "tiene de reseñas Negativas": reseñas_negativas,
        "tiene de reseñas Neutrales": reseñas_neutrales,
        "tiene de reseñas Positivas": reseñas_positivas
    }
    
    return resultados

# Sexta funcion: Sistema de recomendacion de juegos

modelo_railway = pd.read_parquet("data/modelo_railway.parquet")

@app.get("/recomendacion_juego/{id}", name= "RECOMENDACION_JUEGO")
async def recomendacion_juego(id: int):
    
    """La siguiente funcion genera una lista de 5 juegos similares a un juego dado (id)
    Parametros:
    El id del juego para el que se desean encontrar juegos similares. Ej: 10
    Retorna:
    Un diccionario con 5 juegos similares 
    """
    game = modelo_railway[modelo_railway['id'] == id]

    if game.empty:
        return("El juego '{id}' no posee registros.")
    
    # Obtiene el índice del juego dado
    idx = game.index[0]

    # Toma una muestra aleatoria del DataFrame df_games
    sample_size = 2000  # Define el tamaño de la muestra (ajusta según sea necesario)
    df_sample = modelo_railway.sample(n=sample_size, random_state=42)  # Ajusta la semilla aleatoria según sea necesario

    # Calcula la similitud de contenido solo para el juego dado y la muestra
    sim_scores = cosine_similarity([modelo_railway.iloc[idx, 3:]], df_sample.iloc[:, 3:])

    # Obtiene las puntuaciones de similitud del juego dado con otros juegos
    sim_scores = sim_scores[0]

    # Ordena los juegos por similitud en orden descendente
    similar_games = [(i, sim_scores[i]) for i in range(len(sim_scores)) if i != idx]
    similar_games = sorted(similar_games, key=lambda x: x[1], reverse=True)

    # Obtiene los 5 juegos más similares
    similar_game_indices = [i[0] for i in similar_games[:5]]

    # Lista de juegos similares (solo nombres)
    similar_game_names = df_sample['app_name'].iloc[similar_game_indices].tolist()

    return {"similar_games": similar_game_names}