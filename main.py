import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_data(pokemon_file='pokemon.csv', ratings_file='pokemon_ratings.csv'):
    """Carga y preprocesa los datos de Pokémon y las calificaciones."""
    # Cargar los datos de Pokémon
    pokemon_df = pd.read_csv(pokemon_file)
    
    # El pokemon.csv tiene pokemon repetidos, supongo que es porque tienen formas diferentes o regionales
    # Agrupar por 'name' y tomar la primera entrada de cada grupo para eliminar duplicados
    pokemon_df = pokemon_df.groupby('name').first().reset_index()
    
    # Cargar los datos de calificaciones
    ratings_df = pd.read_csv(ratings_file)
    
    # Unir los datos de Pokémon con las calificaciones
    merged_df = pd.merge(ratings_df, pokemon_df, on='pokemonId')
    
    # Crear una tabla dinámica con usuarios como filas y Pokémon como columnas
    ratings_pivot = merged_df.pivot_table(index='userId', columns='pokemonId', values='rating').fillna(0)
    
    return pokemon_df, ratings_pivot

def fit_knn_model(ratings_pivot):
    """Ajusta el modelo KNN en la tabla dinámica de calificaciones."""
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(ratings_pivot.values)
    return knn

def get_favorite_pokemon(user_id, ratings_pivot, pokemon_df, n=5):
    """Obtiene los Pokémon favoritos para un usuario dado."""
    if user_id not in ratings_pivot.index:
        raise ValueError(f"User ID {user_id} not found in ratings data.")
    
    # Obtener las calificaciones del usuario y ordenarlas de mayor a menor
    user_ratings = ratings_pivot.loc[user_id].sort_values(ascending=False)
    
    # Obtener los IDs de los Pokémon favoritos
    favorite_pokemon_ids = user_ratings.head(n).index
    
    # Filtrar los datos de Pokémon para obtener solo los favoritos
    favorite_pokemon = pokemon_df[pokemon_df['pokemonId'].isin(favorite_pokemon_ids)]
    
    # Seleccionar las columnas relevantes
    favorite_pokemon = favorite_pokemon[['pokemonId', 'name', 'Type1', 'Type2']]
    
    # Añadir la calificación del usuario para cada Pokémon favorito
    favorite_pokemon['rating'] = favorite_pokemon['pokemonId'].apply(lambda x: user_ratings[x])
    
    return favorite_pokemon

def recommend_pokemon(user_id, ratings_pivot, pokemon_df, knn, n_neighbors=5, n_recommendations=5):
    """Recomienda Pokémon para un usuario dado basado en las calificaciones de usuarios similares."""
    if user_id not in ratings_pivot.index:
        raise ValueError(f"User ID {user_id} not found in ratings data.")
    
    # Obtener el índice del usuario en la tabla dinámica
    user_index = ratings_pivot.index.get_loc(user_id)
    
    # Encontrar los vecinos más cercanos del usuario
    distances, indices = knn.kneighbors(ratings_pivot.iloc[user_index, :].values.reshape(1, -1), n_neighbors=n_neighbors)
    
    # Obtener los usuarios similares
    similar_users = indices.flatten()
    
    # Obtener las calificaciones de los usuarios similares
    similar_users_ratings = ratings_pivot.iloc[similar_users]
    
    # Calcular la calificación media ponderada para cada Pokémon
    weights = 1 - distances.flatten()  # Convertir distancias a puntuaciones de similitud
    weighted_ratings = similar_users_ratings.T.dot(weights) / weights.sum()
    
    # Filtrar los Pokémon ya calificados por el usuario
    user_rated_pokemon = ratings_pivot.loc[user_id][ratings_pivot.loc[user_id] > 0].index
    weighted_ratings = weighted_ratings.drop(user_rated_pokemon)
    
    # Obtener las N mejores recomendaciones
    recommended_pokemon_ids = weighted_ratings.sort_values(ascending=False).head(n_recommendations).index
    recommended_pokemon = pokemon_df[pokemon_df['pokemonId'].isin(recommended_pokemon_ids)].copy()
    
    # Esto queda aquí porque la calidad de los datos es muy importanten para el modelo KNN
    # Añadir "justificación" (la media ponderada) de porque se recomeinda para cada recomendación
    recommended_pokemon.loc[:, 'justification'] = recommended_pokemon['pokemonId'].apply(lambda x: f"Weighted rating: {weighted_ratings[x]:.2f}")
    
    return recommended_pokemon[['pokemonId', 'name', 'Type1', 'Type2', 'justification']]

# Cargar datos
pokemon_df, ratings_pivot = load_data()

# Ajustar el modelo KNN
knn = fit_knn_model(ratings_pivot)

# Ejemplo de uso
user_id = 231
print("Pokémon favoritos:")
print(get_favorite_pokemon(user_id, ratings_pivot, pokemon_df))

print("\nPokémon recomendados:")
print(recommend_pokemon(user_id, ratings_pivot, pokemon_df, knn))

# Nota: El método de recomendación no es del todo infalible. La justificación de las recomendaciones se basa en la "calificación ponderada" (weighted rating),
# que se calcula utilizando las calificaciones de usuarios similares. Sin embargo, esto puede no siempre reflejar con precisión las preferencias individuales
# debido a la variabilidad en los patrones de calificación y la cantidad de datos disponibles. Por ejemplo, para el usuario con ID 231 se obtienen buenas 
# recomendaciones, pero para el usuario con ID 20, las recomendaciones pueden no ser tan precisas. 
# Esto se puede ver reflejado en los tipos de los diferentes pokémon.
# El usuario 231 tiene preferencia por pokémon de tipo "Water", "Steel" y "Dragon", y las recomendaciones incluyen pokémon de esos tipos.
# Por otro lado, el usuario 20 tiene preferencia por pokémon de tipo variado y las recomendaciones no incluyen pokémon de esos tipos.