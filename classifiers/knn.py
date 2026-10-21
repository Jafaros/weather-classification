from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# Vytvoří a vrátí instanci klasifikátoru k-Nearest Neighbors (kNN) s daným počtem sousedů.
def create_knn(n_neighbors: int = 5) -> Pipeline:
    # Pipeline učí měřítko pouze na trénovací části každého foldu.
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="brute"),
    )
