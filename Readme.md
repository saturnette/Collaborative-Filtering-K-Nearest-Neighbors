# Collaborative Filtering con K-Nearest Neighbors (KNN)

El filtrado colaborativo es una técnica utilizada en sistemas de recomendación que se basa en la colaboración entre usuarios para hacer recomendaciones. Una de las metodologías más comunes para implementar el filtrado colaborativo es mediante el uso del algoritmo K-Nearest Neighbors (KNN).

## Matrices Utilizadas

En el filtrado colaborativo, se utilizan principalmente dos matrices:

1. **Matriz de Usuarios-Ítems (R)**: Esta matriz contiene las calificaciones que los usuarios han dado a los ítems (productos, películas, etc.). Cada fila representa un usuario y cada columna representa un ítem.

$$
R = \begin{pmatrix}
r_{11} & r_{12} & \cdots & r_{1n} \\
r_{21} & r_{22} & \cdots & r_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
r_{m1} & r_{m2} & \cdots & r_{mn}
\end{pmatrix}
$$

2. **Matriz de Similitud (S)**: Esta matriz contiene las similitudes entre los usuarios o ítems. En el caso de KNN, se puede calcular la similitud entre usuarios o ítems utilizando métricas como la similitud del coseno.

$$
S = \begin{pmatrix}
s_{11} & s_{12} & \cdots & s_{1m} \\
s_{21} & s_{22} & \cdots & s_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
s_{m1} & s_{m2} & \cdots & s_{mm}
\end{pmatrix}
$$

## Fórmulas Utilizadas

### Similitud del Coseno

Para calcular la similitud entre dos usuarios \( u \) y \( v \), se puede utilizar la similitud del coseno:

$$
\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui} \cdot r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2} \cdot \sqrt{\sum_{i \in I_{uv}} r_{vi}^2}}
$$

donde <i>I<sub>uv</sub></i> es el conjunto de ítems que tanto el usuario <i>u</i> como el usuario <i>v</i> han calificado.

### Predicción de Calificaciones

Para predecir la calificación que un usuario \( u \) daría a un ítem \( i \), se puede utilizar la siguiente fórmula:

$$
\hat{r}_{ui} = \frac{\sum_{v \in N_k(u)} \text{sim}(u, v) \cdot r_{vi}}{\sum_{v \in N_k(u)} |\text{sim}(u, v)|}
$$

donde \(N_k(u)\) es el conjunto de los \( k \) vecinos más cercanos del usuario \( u \).

El uso del algoritmo KNN en el filtrado colaborativo permite hacer recomendaciones personalizadas basadas en las similitudes entre usuarios o ítems, mejorando así la experiencia del usuario en el sistema de recomendación.

# Ejemplo de Cálculo de Similitud y Normas con KNN en Filtrado Colaborativo

Supongamos que tenemos una matriz de calificaciones de usuarios a productos como la siguiente. En esta matriz, las columnas representan productos y las filas representan usuarios:



$$
R = \begin{pmatrix}
4 & 0 & 0 & 5 & 1 & 0 & 0 & 3 \\
3 & 3 & 1 & 0 & 0 & 0 & 0 & 2 \\
0 & 0 & 0 & 2 & 4 & 5 & 0 & 1 \\
5 & 1 & 0 & 0 & 0 & 0 & 4 & 0 \\
0 & 0 & 5 & 4 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

**NOTA**: Las celdas con "0" significa que el usuario no ha evaluado ese item.
Podemos representar las calificaciones de los usuarios como vectores:

$$
A = \begin{pmatrix}
4 \\
0 \\
0 \\
5 \\
1 \\
0 \\
0 \\
3
\end{pmatrix}, \quad
B = \begin{pmatrix}
3 \\
3 \\
1 \\
0 \\
0 \\
0 \\
0 \\
2
\end{pmatrix}, \quad
C = \begin{pmatrix}
0 \\
0 \\
0 \\
2 \\
4 \\
5 \\
0 \\
1
\end{pmatrix}, \quad
D = \begin{pmatrix}
5 \\
1 \\
0 \\
0 \\
0 \\
0 \\
4 \\
0
\end{pmatrix}, \quad
E = \begin{pmatrix}
0 \\
0 \\
5 \\
4 \\
0 \\
0 \\
0 \\
0
\end{pmatrix}
$$

### Producto Punto

El producto punto entre los vectores \( A \) y \( B \) se calcula como:

$$
A \cdot B = 4 \cdot 3 + 0 \cdot 3 + 0 \cdot 1 + 5 \cdot 0 + 1 \cdot 0 + 0 \cdot 0 + 0 \cdot 0 + 3 \cdot 2 = 12 + 6 = 18
$$

El producto punto entre los vectores \( A \) y \( C \) se calcula como:

$$
A \cdot C = 4 \cdot 0 + 0 \cdot 0 + 0 \cdot 0 + 5 \cdot 2 + 1 \cdot 4 + 0 \cdot 5 + 0 \cdot 0 + 3 \cdot 1 = 10 + 4 + 3 = 17
$$

El producto punto entre los vectores \( A \) y \( D \) se calcula como:

$$
A \cdot D = 4 \cdot 5 + 0 \cdot 1 + 0 \cdot 0 + 5 \cdot 0 + 1 \cdot 0 + 0 \cdot 0 + 0 \cdot 4 + 3 \cdot 0 = 20
$$

El producto punto entre los vectores \( A \) y \( E \) se calcula como:

$$
A \cdot E = 4 \cdot 0 + 0 \cdot 0 + 0 \cdot 5 + 5 \cdot 4 + 1 \cdot 0 + 0 \cdot 0 + 0 \cdot 0 + 3 \cdot 0 = 20
$$

### Normas

La norma del vector \( A \) se calcula como:

$$
\|A\| = \sqrt{4^2 + 0^2 + 0^2 + 5^2 + 1^2 + 0^2 + 0^2 + 3^2} = \sqrt{16 + 25 + 1 + 9} = \sqrt{51}
$$

La norma del vector \( B \) se calcula como:

$$
\|B\| = \sqrt{3^2 + 3^2 + 1^2 + 0^2 + 0^2 + 0^2 + 0^2 + 2^2} = \sqrt{9 + 9 + 1 + 4} = \sqrt{23}
$$

La norma del vector \( C \) se calcula como:

$$
\|C\| = \sqrt{0^2 + 0^2 + 0^2 + 2^2 + 4^2 + 5^2 + 0^2 + 1^2} = \sqrt{4 + 16 + 25 + 1} = \sqrt{46}
$$

La norma del vector \( D \) se calcula como:

$$
\|D\| = \sqrt{5^2 + 1^2 + 0^2 + 0^2 + 0^2 + 0^2 + 4^2 + 0^2} = \sqrt{25 + 1 + 16} = \sqrt{42}
$$

La norma del vector \( E \) se calcula como:

$$
\|E\| = \sqrt{0^2 + 0^2 + 5^2 + 4^2 + 0^2 + 0^2 + 0^2 + 0^2} = \sqrt{25 + 16} = \sqrt{41}
$$

### Similitud del Coseno

Finalmente, la similitud del coseno entre los vectores \( A \) y \( B \) se calcula como:

$$
\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{18}{\sqrt{51} \cdot \sqrt{23}} \approx 0.548
$$

Y la similitud del coseno entre los vectores \( A \) y \( C \) se calcula como:

$$
\text{sim}(A, C) = \frac{A \cdot C}{\|A\| \|C\|} = \frac{17}{\sqrt{51} \cdot \sqrt{46}} \approx 0.353
$$

La similitud del coseno entre los vectores \( A \) y \( D \) se calcula como:

$$
\text{sim}(A, D) = \frac{A \cdot D}{\|A\| \|D\|} = \frac{20}{\sqrt{51} \cdot \sqrt{42}} \approx 0.434
$$

La similitud del coseno entre los vectores \( A \) y \( E \) se calcula como:

$$
\text{sim}(A, E) = \frac{A \cdot E}{\|A\| \|E\|} = \frac{20}{\sqrt{51} \cdot \sqrt{41}} \approx 0.438
$$

### Análisis de Similitud

En este ejemplo, la similitud del coseno entre los usuarios \( A \) y \( B \) es aproximadamente 0.548, mientras que la similitud entre los usuarios \( A \) y \( C \) es aproximadamente 0.353. Esto indica que el usuario \( A \) es más similar al usuario \( B \) que al usuario \( C \). La similitud entre \( A \) y \( D \) es aproximadamente 0.434, y la similitud entre \( A \) y \( E \) es aproximadamente 0.438.

### Predicción de Calificaciones

Una vez que hemos calculado las similitudes, podemos usar estas similitudes para predecir las calificaciones que un usuario podría dar a productos que aún no ha calificado. Por ejemplo, si queremos predecir la calificación que el usuario \( A \) daría a un producto que no ha calificado, podemos usar las calificaciones de los usuarios más similares a \( A \).

### Predicción Basada en Vecinos Más Cercanos

Supongamos que queremos predecir la calificación que el usuario \( A \) daría al producto 2. Podemos usar las calificaciones de los usuarios \( B \), \( C \), \( D \), y \( E \) (los vecinos más cercanos) y sus similitudes con \( A \).

La fórmula para predecir la calificación \( \hat{r}_{A,2} \) es:

$$
\hat{r}_{A,2} = \frac{\sum_{i \in N(A)} \text{sim}(A, i) \cdot r_{i,2}}{\sum_{i \in N(A)} \text{sim}(A, i)}
$$

Donde \( N(A) \) es el conjunto de vecinos más cercanos de \( A \).

En este caso, los vecinos más cercanos son \( B \), \( C \), \( D \), y \( E \):

$$
\hat{r}_{A,2} = \frac{0.548 \cdot 3 + 0.353 \cdot 0 + 0.434 \cdot 1 + 0.438 \cdot 0}{0.548 + 0.353 + 0.434 + 0.438} = \frac{1.644 + 0 + 0.434 + 0}{1.773} \approx 1.17
$$

### Recomendación de Productos

Finalmente, podemos recomendar productos al usuario \( A \) basándonos en las calificaciones predichas. En este caso, podríamos recomendar el producto 2 al usuario \( A \) con una calificación predicha de aproximadamente 1.17.

Este proceso de predicción y recomendación se puede repetir para todos los productos que el usuario \( A \) no ha calificado, utilizando las similitudes calculadas y las calificaciones de los vecinos más cercanos.

Para aplicar un umbral y hacer que solo algunos usuarios tengan más similitud que otros, podemos establecer un umbral de similitud, por ejemplo, 0.4. Solo consideraremos vecinos cuya similitud con el usuario \( A \) sea mayor o igual a este umbral.

### Similitud del Coseno con Umbral

Recalculamos las similitudes y aplicamos el umbral:

- Similitud <i>sim(A, B) ≈ 0.548</i> (mayor que 0.4, se considera)
- Similitud <i>sim(A, C) ≈ 0.353</i> (menor que 0.4, no se considera)
- Similitud <i>sim(A, D) ≈ 0.434</i> (mayor que 0.4, se considera)
- Similitud <i>sim(A, E) ≈ 0.438</i> (mayor que 0.4, se considera)

### Predicción de Calificaciones con Umbral

Usamos solo los vecinos \( B \), \( D \), y \( E \) para predecir la calificación que el usuario \( A \) daría al producto 2:

$$
\hat{r}_{A,2} = \frac{\sum_{i \in N(A)} \text{sim}(A, i) \cdot r_{i,2}}{\sum_{i \in N(A)} \text{sim}(A, i)}
$$

Donde \( N(A) \) es el conjunto de vecinos más cercanos de \( A \) que cumplen con el umbral.

En este caso, los vecinos más cercanos son \( B \), \( D \), y \( E \):

$$
\hat{r}_{A,2} = \frac{0.548 \cdot 3 + 0.434 \cdot 1 + 0.438 \cdot 0}{0.548 + 0.434 + 0.438} = \frac{1.644 + 0.434 + 0}{1.42} \approx 1.46
$$

### Recomendación de Productos

Finalmente, podemos recomendar productos al usuario \( A \) basándonos en las calificaciones predichas. En este caso, podríamos recomendar el producto 2 al usuario \( A \) con una calificación predicha de aproximadamente 1.46.

Este proceso de predicción y recomendación se puede repetir para todos los productos que el usuario \( A \) no ha calificado, utilizando las similitudes calculadas y las calificaciones de los vecinos más cercanos que cumplen con el umbral.

**NOTA**: En este proyecto se recomiendan pokemon, basados en usuarios similares. 