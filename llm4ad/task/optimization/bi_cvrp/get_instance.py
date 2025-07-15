import numpy as np

class GetData():
    def __init__(self, n_instance: int, n_customers: int):
        self.n_instance = n_instance
        self.n_customers = n_customers

    def generate_instances(self):
        np.random.seed(2025)
        instance_data = []

        for _ in range(self.n_instance):
            # Depot + customers
            coords = np.random.rand(self.n_customers + 1, 2)  # (x, y) positions in [0, 1]^2
            demands = np.random.randint(1, 10, size=self.n_customers+1)
            demands[0]  = 0  # Depot has no demand

            # calculate distance matrix
            distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)

            # Set vehicle capacity based on number of customers
            if 20 <= self.n_customers < 40:
                capacity = 30
            elif 40 <= self.n_customers < 70:
                capacity = 40
            elif 70 <= self.n_customers <= 100:
                capacity = 50
            else:
                raise ValueError("Number of customers must be between 20 and 100.")

            instance_data.append((coords, demands, distance_matrix))

        return instance_data, capacity


if __name__ == '__main__':
    getData = GetData(3, 50)
    instance_data, capacity = getData.generate_instances()
    print("Coordinates (first instance):\n", instance_data[0][0].shape)
    print("Normalized Demands (first instance):\n", instance_data[0][1].shape)
    print("Distance Matrix (first instance):\n", instance_data[0][2])
    print("Vehicle Capacity:", capacity)
