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
from scipy.sparse import data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %matplotlib inline


# %%
class BaseFunc:
    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def derivative(self, data) -> np.matrix:
        pass

    def parameters(self):
        return self._parameters

    def update_parameters(self, new_parameters):
        # print("New par shape: ", new_parameters.shape)
        self._parameters = new_parameters


# %%
class LossFunc:
    @abstractmethod
    def loss_value(self, prediction, ground_trouth):
        pass

    @abstractmethod
    def loss_derivative(self, prediction, ground_trouth, parameters=None):
        pass


# %%
class Model:
    def __init__(
        self,
        base_func: BaseFunc,
        loss_func: LossFunc,
        learning_rate,
        regularization_derivative=lambda x: np.zeros(x.shape),
    ) -> None:
        self.base_func = base_func
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.regularization_derivative = regularization_derivative

    def train(self, epochs_num, data, labels, loss_at_epoch=None):
        for i in range(epochs_num):
            prediction = self.base_func.predict(data)
            # print(prediction.shape)

            reg_derivative = self.regularization_derivative(self.base_func.parameters())

            assert (
                reg_derivative.shape == self.base_func.parameters().shape
            ), f"""This should not happen: reg_derivative.shape: {reg_derivative.shape} != 
                parameters_shape: {self.base_func.parameters().shape}"""

            # print(reg_derivative.shape)
            gradient = (
                self.base_func.derivative(data).T
                @ self.loss_func.loss_derivative(prediction, labels)
                + reg_derivative
            )
            # print(
            #     f"bsd: {self.base_func.derivative(data).T.shape}, lsd: {self.loss_func.loss_derivative(prediction, labels).shape}, reg_der: {reg_derivative.shape}"
            # )
            # print("Gradient shape: ", gradient.shape)

            new_parameters = self.base_func.parameters() - self.learning_rate * gradient

            self.base_func.update_parameters(new_parameters)

            if loss_at_epoch is not None:
                loss_at_epoch.append(self.test(data, labels))

            # if (i % 10) == 0:
            #     self.learning_rate *= 0.99

    def test(self, data, labels):
        # print(data.shape)
        prediction = self.base_func.predict(data)
        answer = self.loss_func.loss_value(prediction, labels)
        # print(np.array(answer).shape)
        assert np.array(answer).shape == (), f"Found shape: {np.array(answer).shape}"

        return answer


# %% [markdown]
# # Concrete implementations


# %%
class OrdinaryBaseFunc(BaseFunc):
    def __init__(self, degree, number_of_inputs) -> None:
        # print(degree)
        self._parameters = np.random.randn(degree)

    def predict(self, data):
        # temp = data @ self._parameters
        # print(self._parameters.shape)

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

    def loss_derivative(self, prediction, ground_trouth, parameters=None):
        return 1 / self.number_of_inputs * (prediction - ground_trouth)

    def __init__(self, number_of_inputs) -> None:
        self.number_of_inputs = number_of_inputs


# %%
lambda_coefficient = 0.02


def L2_regularization(parameters):
    temp_parameters = np.array(parameters)
    temp_parameters[0] = 0
    return lambda_coefficient * 2 * temp_parameters.T


def L1_regularization(parameters):
    temp_parameters = np.array(parameters)
    temp_parameters[0] = 0
    return lambda_coefficient * np.sign(temp_parameters)


# %% [markdown]
# # Load and display data

# %%
df = pd.read_csv("dane.data", delimiter=r"\s+", header=None, decimal=",")


# %%
def divide_data(data, test_ratio):
    # return train_test_split(data, train_size=train_ratio,  random_state=42)
    return train_test_split(data, test_size=test_ratio, random_state=6)
    # return train_test_split(data, train_size=train_ratio, stratify=housing["median_house_value"], random_state=8)


train_set, test_set = divide_data(df, test_ratio=0.2)
train_set, validation_set = divide_data(train_set, test_ratio=0.2 / 0.6)
# %%
train_labels = train_set.pop(train_set.columns[7])
validation_labels = validation_set.pop(validation_set.columns[7])
test_labels = test_set.pop(test_set.columns[7])
# train_set.insert(0, "Ones", 1)
train_set.head()

# %%
train_labels.head()

# %% [markdown]
# # Analyze Data

# %%
df.info()

# %%
df.describe()

# %%
df_copy = df.copy()
cols_num = len(df_copy.T)
print(cols_num)
for exponent in range(2, 6):
    for i in range(cols_num - 1):
        df_copy[f"feature {i}, exponent {exponent}"] = df_copy[i] ** exponent

for exponent in range(1, 3):
    for left in range(cols_num - 1):
        for right in range(left + 1, cols_num - 1):
            df_copy[f"f_{left}^{exponent} * f_{right}"] = (
                df_copy[left] ** exponent
            ) * df_copy[right]
            if exponent > 1:
                df_copy[f"f_{left} * f_{right}^{exponent}"] = (
                    df_copy[left] * df_copy[right] ** exponent
                )

for first in range(cols_num - 1):
    for second in range(first + 1, cols_num - 1):
        for third in range(second + 1, cols_num - 1):
            df_copy[f"f_{first}*f_{second}*f_{third}"] = (
                df_copy[first] * df_copy[second] * df_copy[third]
            )

cols_num = len(df_copy.T)
print(cols_num)

corr_matrix = df_copy.corr()
# corr_matrix = corr_matrix[abs(corr_matrix[1]) >= 0.05]
pd.set_option("display.max_rows", 200)
corr_matrix[7].sort_values(ascending=False)
# %%
pd.reset_option("display.max_rows")
df.hist(bins=50, figsize=(20, 15))
plt.show()


# %%
def preprocess_data(data_matrix):
    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0)
    std[std == 0] = 1

    standarized = (data_matrix - mean) / std
    return np.c_[np.ones(standarized.shape[0]), standarized]


# %%
def to_matrices(data, labels, preprocess_func=lambda x: x):
    return preprocess_func(data.to_numpy()), labels.to_numpy()


train_matrix, train_labels_matrix = to_matrices(
    train_set, train_labels, preprocess_data
)


print(train_matrix[:6])
print(train_matrix.shape)
print(train_labels_matrix.shape)

# %%

validation_matrix, validation_labels_matrix = to_matrices(
    validation_set, validation_labels, preprocess_data
)
test_matrix, test_labels_matrix = to_matrices(test_set, test_labels, preprocess_data)

print(test_matrix.shape, test_labels_matrix.shape)

# test_set

# %% [markdown]
# # Plot results


# %%
def train_and_test(
    model,
    train_matrix,
    train_labels_matrix,
    test_matrix,
    test_labels_matrix,
    epochs_num=700,
):
    loss_at_epochs = [model.test(train_matrix, train_labels_matrix)]
    model.train(epochs_num, train_matrix, train_labels_matrix, loss_at_epochs)
    print(model.test(train_matrix, train_labels_matrix))
    print(model.test(test_matrix, test_labels_matrix))

    plt.plot(range(len(loss_at_epochs)), loss_at_epochs, marker="o", linewidth=0.0025)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Loss at epochs")
    plt.grid(True)
    plt.show()


# %% [markdown]
# # Ordinary model

# %%
base_func = OrdinaryBaseFunc(train_matrix.shape[1], train_matrix.shape[0])
loss_func = MSE(train_matrix.shape[0])

model = Model(base_func, loss_func, 0.01)

# %%
train_and_test(
    model, train_matrix, train_labels_matrix, test_matrix, test_labels_matrix
)
# %% [markdown]
# # Ordinary with L2 regularization

# %%
base_func_reg_L2 = OrdinaryBaseFunc(train_matrix.shape[1], train_matrix.shape[0])
loss_func_reg_L2 = MSE(train_matrix.shape[0])


model = Model(base_func_reg_L2, loss_func_reg_L2, 0.01, L2_regularization)

# %%
train_and_test(
    model, train_matrix, train_labels_matrix, test_matrix, test_labels_matrix
)

# %% [markdown]
# # Ordinary with L1 regularization

# %%
lambda_coefficient = 10
base_func_reg_L1 = OrdinaryBaseFunc(train_matrix.shape[1], train_matrix.shape[0])
loss_func_reg_L1 = MSE(train_matrix.shape[0])


model = Model(base_func_reg_L1, loss_func_reg_L1, 0.01, L1_regularization)

# %%
train_and_test(
    model, train_matrix, train_labels_matrix, test_matrix, test_labels_matrix
)

# %% [markdown]
# # First Hypothesis


# %%
def test_hypothesis(
    preprocess_data_func,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    epochs_num=700,
):
    train_matrix_hyp = preprocess_data_func(train_matrix)
    validation_matrix_hyp = preprocess_data_func(validation_matrix)
    test_matrix_hyp = preprocess_data_func(test_matrix)
    base_func_hyp = OrdinaryBaseFunc(
        train_matrix_hyp.shape[1], train_matrix_hyp.shape[0]
    )
    loss_func_hyp = MSE(train_matrix_hyp.shape[0])
    model = Model(base_func_hyp, loss_func_hyp, 0.01)

    train_and_test(
        model,
        train_matrix_hyp,
        train_labels_matrix,
        validation_matrix_hyp,
        validation_labels_matrix,
        epochs_num,
    )


# %%
def preprocess_data_matrix_1(data_matrix):
    answer = data_matrix[:, [3]]
    return answer


test_hypothesis(
    preprocess_data_matrix_1,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
)

# %% [markdown]
# # Second Hypothesis


# %%
def preprocess_data_matrix_2(data_matrix):
    answer = data_matrix[:, [3, 4]]
    return answer


test_hypothesis(
    preprocess_data_matrix_2,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
)


# %% [markdown]
# # Third Hypothesis


# %%
def preprocess_data_matrix_3(data_matrix):
    answer = np.column_stack(
        (data_matrix[:, 3], data_matrix[:, 4], data_matrix[:, 3] * data_matrix[:, 4])
    )
    # print(answer.shape)
    return answer


test_hypothesis(
    preprocess_data_matrix_3,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
)
