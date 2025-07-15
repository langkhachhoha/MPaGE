import numpy as np
class GetData():
    def __init__(self,n_instance,n_cities):
        self.n_instance = n_instance
        self.n_cities = n_cities

    def generate_instances(self):
        np.random.seed(2025)
        instance_data = []
        for _ in range(self.n_instance):
            coordinates_1 = np.random.rand(self.n_cities, 2)
            coordinates_2 = np.random.rand(self.n_cities, 2)
            coordinates_3 = np.random.rand(self.n_cities, 2)
            coordinates = np.concatenate((coordinates_1, coordinates_2, coordinates_3), axis=1)
            distances_1 = np.linalg.norm(coordinates_1[:, np.newaxis] - coordinates_1, axis=2)
            distances_2 = np.linalg.norm(coordinates_2[:, np.newaxis] - coordinates_2, axis=2)
            distances_3 = np.linalg.norm(coordinates_3[:, np.newaxis] - coordinates_3, axis=2)
            instance_data.append((coordinates,distances_1, distances_2, distances_3))
        return instance_data