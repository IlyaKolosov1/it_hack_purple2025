import pandas as pd
import numpy as np 
from annoy import AnnoyIndex
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

class DbVectorizer:
    def __init__(self, data_path: str, num_trees : int = 10):
        self.data_path = data_path
        self.num_trees = num_trees
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = StandardScaler()
        self.index = None  
        self.df = None
        self.num_features = self.num_features = [
    'energy_100g', 'fat_100g', 'saturated-fat_100g', 'unsaturated-fat_100g',
    'monounsaturated-fat_100g', 'polyunsaturated-fat_100g', 'trans-fat_100g',
    'cholesterol_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g', 'sodium_100g', 'omega-3-fat_100g',
    'omega-6-fat_100g', 'omega-9-fat_100g', 'alpha-linolenic-acid_100g',
    'linoleic-acid_100g', 'oleic-acid_100g', 'stearic-acid_100g',
    'vitamin-a_100g', 'beta-carotene_100g', 'vitamin-d_100g', 'vitamin-e_100g',
    'vitamin-k_100g', 'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g',
    'vitamin-b6_100g', 'vitamin-b9_100g', 'vitamin-b12_100g', 'iron_100g',
    'magnesium_100g', 'calcium_100g', 'phosphorus_100g', 'zinc_100g',
    'potassium_100g', 'selenium_100g', 'iodine_100g', 'caffeine_100g',
    'taurine_100g', 'carnitine_100g', 'choline_100g', 'nutrition-score-fr_100g',
    'nutrition-score-uk_100g', 'carbon-footprint_100g', 'fruits-vegetables-nuts_100g'
]
    def load_data(self):
        self.df = pd.read_csv(self.data_path, sep='\t')
        self.df.fillna('', inplace=True)

    def vectorize_data(self):
        """Преобразует текстовые и числовые данные в векторы."""
        product_vectors = np.array([self.model.encode(name) for name in self.df['product_name']])
        num_vectors = self.scaler.fit_transform(self.df[self.num_features].replace('', 0).astype(float))
        return np.hstack([product_vectors, num_vectors])
    def build_index(self):
        """Создаёт индекс Annoy."""
        vectors = self.vectorize_data()
        d = vectors.shape[1]
        self.index = AnnoyIndex(d, 'angular')
        for i, vec in enumerate(vectors):
            self.index.add_item(i, vec)
        self.index.build(self.num_trees)
        self.index.save("vector_index.ann")
        return "vector_index.ann"
    
    def get_similar_items(self, query: str, n: int = 5):
        """Ищет похожие продукты по запросу."""
        query_vector = self.model.encode(query)
        query_vector = np.hstack([query_vector, np.zeros(len(self.num_features))])
        indices = self.index.get_nns_by_vector(query_vector, n)
        return self.df.iloc[indices]
    
if __name__ == "__main__":
    vectorizer = DbVectorizer("C:\\Users\\Kolos\\recomender_project\\tbank\\data\\en.openbeautyfacts.org.products.csv")
    vectorizer.load_data()
    index_path = vectorizer.build_index()
    print(f"Annoy индекс сохранён в: {index_path}")