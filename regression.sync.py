# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %matplotlib inline


# %%
class BaseFunc:
    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def derivative(self, data):
        pass

    def parameters(self):
        return self._parameters

    def update_parameters(self, new_parameters):
        self._parameters = new_parameters


# %%
class LossFunc:
    @abstractmethod
    def loss_value(self, prediction, ground_trouth):
        pass

    @abstractmethod
    def loss_derivative(self, prediction, ground_trouth):
        pass


# %%
class Model:
    def __init__(self, base_func: BaseFunc, loss_func: LossFunc, learning_rate) -> None:
        self.base_func = base_func
        self.loss_func = loss_func
        self.learning_rate = learning_rate

    def train(self, epochs_num, data, labels):
        for _ in range(epochs_num):
            prediction = self.base_func.predict(data)

            gradient = self.base_func.derivative(
                data
            ).T @ self.loss_func.loss_derivative(prediction, labels)

            new_parameters = self.base_func.parameters() - self.learning_rate * gradient

            # print(new_parameters)

            self.base_func.update_parameters(new_parameters)

    def test(self, data, labels):
        prediction = base_func.predict(data)
        return loss_func.loss_value(prediction, labels)


# %% [markdown]
# # Concrete implementations


# %%
class OrdinaryBaseFunc(BaseFunc):
    def __init__(self, degree, number_of_inputs) -> None:
        self._parameters = np.sqrt(2 / number_of_inputs) * np.random.randn(degree)

    def predict(self, data):
        # temp = data @ self._parameters
        # print(temp.shape)
        return data @ self._parameters

    def derivative(self, data):
        return data


# %%
class MSE(LossFunc):
    def loss_value(self, prediction, ground_trouth):
        return (
            (1 / (2 * self.number_of_inputs))
            * (ground_trouth - prediction).T
            @ (ground_trouth - prediction)
        )

    def loss_derivative(self, prediction, ground_trouth):
        return 1 / self.number_of_inputs * (prediction - ground_trouth)

    def __init__(self, number_of_inputs) -> None:
        self.number_of_inputs = number_of_inputs


# %% [markdown]
# # Load and display data

# %%
df = pd.read_csv("dane.data", delimiter=r"\s+", header=None, decimal=",")


# %%
def divide_data(data, train_ratio):
    # return train_test_split(data, train_size=train_ratio,  random_state=42)
    return train_test_split(data, train_size=train_ratio, random_state=8)
    # return train_test_split(data, train_size=train_ratio, stratify=housing["median_house_value"], random_state=8)


train_set, test_set = divide_data(df, 0.75)
# %%
train_labels = train_set.pop(train_set.columns[7])
test_labels = test_set.pop(test_set.columns[7])
# train_set.insert(0, "Ones", 1)
train_set.head()

# %%
train_labels.head()


# %%
def preprocess_data(data_matrix):
    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0)
    std[std == 0] = 1

    standarized = (data_matrix - mean) / std
    return np.c_[np.ones(standarized.shape[0]), standarized]


# %%
train_matrix = train_set.to_numpy()
train_labels_matrix = train_labels.to_numpy()

train_matrix = preprocess_data(train_matrix)

print(train_matrix[:6])
print(train_matrix.shape)
print(train_labels_matrix.shape)


# %% [markdown]
# # Ordinary model

# %%
base_func = OrdinaryBaseFunc(train_matrix.shape[1], train_matrix.shape[0])
loss_func = MSE(train_matrix.shape[0])

model = Model(base_func, loss_func, 0.01)

# %%
model.train(10000, train_matrix, train_labels_matrix)

print(model.test(train_matrix, train_labels_matrix))

# %%
