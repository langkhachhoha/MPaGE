import numpy as np
class GetData():
    def __init__(self, n_instance: int, n_items: int):
        self.n_instance = n_instance
        self.n_items = n_items

    def generate_instances(self):
        np.random.seed(2025)
        instance_data = []
        for _ in range(self.n_instance):
            weights = np.random.rand(self.n_items)
            values_obj1 = np.random.rand(self.n_items)
            values_obj2 = np.random.rand(self.n_items)
            if 50 <= self.n_items < 100:
                capacity = 12.5
            elif 100 <= self.n_items <= 200:
                capacity = 25
            else:
                raise ValueError("Number of items must be between 50 and 200.")

            instance_data.append((weights, values_obj1, values_obj2))
        return instance_data, capacity
    

if __name__ == '__main__':
    getData = GetData(5, 200)
    instance_data, capacity = getData.generate_instances()
    print(instance_data[0])
