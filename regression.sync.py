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
# %autosave 0

# %%
from abc import abstractmethod
import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Any
import types
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import time

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
    def loss_value(self, prediction, ground_truth) -> Any:
        pass

    @abstractmethod
    def loss_derivative(self, prediction, ground_truth, parameters=None) -> Any:
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

        assert (
            self.regularization_derivative(self.base_func.parameters())
            is not types.FunctionType
        ), "This should not be a function"

    def _get_gradient(self, data, labels):
        prediction = self.base_func.predict(data)

        reg_derivative = self.regularization_derivative(self.base_func.parameters())

        assert (
            reg_derivative.shape == self.base_func.parameters().shape
        ), f"""This should not happen: reg_derivative.shape: {reg_derivative.shape} != 
            parameters_shape: {self.base_func.parameters().shape}"""

        gradient = (
            self.base_func.derivative(data).T
            @ self.loss_func.loss_derivative(prediction, labels)
            + reg_derivative
        )
        return gradient

    def train(self, epochs_num, data, labels, loss_at_epoch=None):
        for _ in range(epochs_num):
            gradient = self._get_gradient(data, labels)
            new_parameters = self.base_func.parameters() - self.learning_rate * gradient

            self.base_func.update_parameters(new_parameters)

            if loss_at_epoch is not None:
                loss_at_epoch.append(self.test(data, labels))

    def test(self, data, labels):
        # print(data.shape)
        prediction = self.base_func.predict(data)
        answer = self.loss_func.loss_value(prediction, labels)
        # print(np.array(answer).shape)
        assert np.array(answer).shape == (), f"Found shape: {np.array(answer).shape}"

        return answer

    def analytical_mse(self, data, labels):
        old_parameters = self.base_func.parameters()

        pseudo_inv = np.linalg.pinv(data.T @ data)
        analytical_parameters = pseudo_inv @ data.T @ labels
        self.base_func.update_parameters(analytical_parameters)

        answer = self.test(data, labels)
        self.base_func.update_parameters(old_parameters)
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
    def loss_value(self, prediction, ground_truth):
        return (
            # (1 / (2 * self.number_of_inputs))
            # * (ground_truth - prediction).T
            # @ (ground_truth - prediction)
            (1 / (len(prediction)))
            * (ground_truth - prediction).T
            @ (ground_truth - prediction)
        )

    def loss_derivative(self, prediction, ground_truth, parameters=None):
        # return 1 / self.number_of_inputs * (prediction - ground_truth)
        return 1 / len(prediction) * (prediction - ground_truth)

    def __init__(self, number_of_inputs) -> None:
        self.number_of_inputs = number_of_inputs


# %%
class HUBER(LossFunc):
    def loss_value(self, prediction, ground_truth):
        error = ground_truth - prediction
        abs_error = np.abs(error)

        quadratic = 0.5 * np.square(error)
        linear = self.delta * (abs_error - 0.5 * self.delta)

        loss = np.where(abs_error <= self.delta, quadratic, linear)
        return np.mean(loss)

    def loss_derivative(self, prediction, ground_truth, parameters=None):
        error = prediction - ground_truth
        abs_error = np.abs(error)

        gradient = np.where(abs_error <= self.delta, error, self.delta * np.sign(error))
        return gradient / self.number_of_inputs

    def __init__(self, number_of_inputs, delta=1.0):
        self.number_of_inputs = number_of_inputs
        self.delta = delta


# %%
def L2_regularization(lambda_coefficient=0.02):
    def L2_reg_func(parameters):
        temp_parameters = np.array(parameters)
        temp_parameters[0] = 0
        return lambda_coefficient * 2 * temp_parameters.T

    return L2_reg_func


def L1_regularization(lambda_coefficient=0.02):
    def L1_reg_func(parameters):
        temp_parameters = np.array(parameters)
        temp_parameters[0] = 0
        return lambda_coefficient * np.sign(temp_parameters)

    return L1_reg_func


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
cols_num = len(df_copy.columns)

new_columns = {}

for exponent in range(2, 6):
    for i in range(cols_num - 1):
        new_columns[f"feature_{i}_exp_{exponent}"] = df_copy.iloc[:, i] ** exponent

# for left in range(cols_num - 1):
#     for right in range(left + 1, cols_num - 1):
#         new_columns[f"f_{left} + f_{right}"] = df_copy.iloc[:, left] + df_copy.iloc[:, right]

for exponent in range(1, 3):
    for left in range(cols_num - 1):
        for right in range(left + 1, cols_num - 1):
            new_columns[f"f_{left}^{exponent} * f_{right}"] = (
                df_copy.iloc[:, left] ** exponent * df_copy.iloc[:, right]
            )
            if exponent > 1:
                new_columns[f"f_{left} * f_{right}^{exponent}"] = (
                    df_copy.iloc[:, left] * df_copy.iloc[:, right] ** exponent
                )

for first in range(cols_num - 1):
    for second in range(first + 1, cols_num - 1):
        for third in range(second + 1, cols_num - 1):
            new_columns[f"f_{first} * f_{second} * f_{third}"] = (
                df_copy.iloc[:, first]
                * df_copy.iloc[:, second]
                * df_copy.iloc[:, third]
            )


base_col = df.iloc[:, 3]
mean_3 = np.mean(base_col.to_numpy())
powers = [1 / 10, 1 / 20, 1 / 100]
powers += range(1, 100)
temp = pd.DataFrame(
    {
        f"exp -([3]-mean[3])^2/{s}^2": np.exp(-((base_col - mean_3) ** 2) / (s**2))
        for s in powers
    }
)

df_copy = pd.concat([df_copy, pd.DataFrame(new_columns), temp], axis=1)
cols_num = len(df_copy.T)
print(cols_num)

corr_matrix = df_copy.corr()
# # corr_matrix = corr_matrix[abs(corr_matrix[1]) >= 0.05]
pd.set_option("display.max_rows", 200)

correlations = corr_matrix[7]
correlations = correlations[abs(correlations) > 0.25]
correlations.sort_values(ascending=False)

# %%
pd.reset_option("display.max_rows")
df.hist(bins=50, figsize=(20, 15))
plt.show()


# %%
def standarize_matrix(data_matrix):
    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0)
    std[std == 0] = 1
    standarized = (data_matrix - mean) / std
    return standarized


def standarize_matrix_and_add_ones(data_matrix):
    standarized = standarize_matrix(data_matrix)
    return np.c_[np.ones(standarized.shape[0]), standarized]


# %%
def to_matrices(data, labels, preprocess_func=lambda x: x):
    return preprocess_func(data.to_numpy()), labels.to_numpy()


train_matrix, train_labels_matrix = to_matrices(
    train_set, train_labels, standarize_matrix_and_add_ones
)


print(train_matrix[:6])
print(train_matrix.shape)
print(train_labels_matrix.shape)

# %%

validation_matrix, validation_labels_matrix = to_matrices(
    validation_set, validation_labels, standarize_matrix_and_add_ones
)
test_matrix, test_labels_matrix = to_matrices(
    test_set, test_labels, standarize_matrix_and_add_ones
)

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
    skip_this_many_in_plot=0,
):
    loss_at_epochs = [model.test(train_matrix, train_labels_matrix)]
    model.train(epochs_num, train_matrix, train_labels_matrix, loss_at_epochs)
    print(model.test(train_matrix, train_labels_matrix))
    print(model.test(test_matrix, test_labels_matrix))

    print("Analytical score: ", model.analytical_mse(test_matrix, test_labels_matrix))

    loss_at_epochs = loss_at_epochs[skip_this_many_in_plot:]
    plt.plot(range(len(loss_at_epochs)), loss_at_epochs, marker="o", linewidth=0.0025)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Loss at epochs")
    plt.grid(True)
    plt.show()


# %% [markdown]
# # Ordinary model

# %%
learning_rate = 0.005

base_func = OrdinaryBaseFunc(train_matrix.shape[1], train_matrix.shape[0])
loss_func = MSE(train_matrix.shape[0])

model = Model(base_func, loss_func, learning_rate)

# %%
train_and_test(
    model,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
)
# %% [markdown]
# # Ordinary with L2 regularization

# %%
base_func_reg_L2 = OrdinaryBaseFunc(train_matrix.shape[1], train_matrix.shape[0])
loss_func_reg_L2 = MSE(train_matrix.shape[0])


model = Model(base_func_reg_L2, loss_func_reg_L2, learning_rate, L2_regularization())

# %%
train_and_test(
    model,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
)

# %% [markdown]
# # Ordinary with L1 regularization

# %%
lambda_coefficient = 10
base_func_reg_L1 = OrdinaryBaseFunc(train_matrix.shape[1], train_matrix.shape[0])
loss_func_reg_L1 = MSE(train_matrix.shape[0])


model = Model(base_func_reg_L1, loss_func_reg_L1, learning_rate, L1_regularization())

# %%
train_and_test(
    model,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
)

# %% [markdown]
# # First Hypothesis


# %%
def test_hypothesis(
    preprocess_matrix_func,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    epochs_num=700,
    regularization_derivative=None,
):
    train_matrix_hyp = preprocess_matrix_func(train_matrix)
    validation_matrix_hyp = preprocess_matrix_func(validation_matrix)
    test_matrix_hyp = preprocess_matrix_func(test_matrix)
    base_func_hyp = OrdinaryBaseFunc(
        train_matrix_hyp.shape[1], train_matrix_hyp.shape[0]
    )
    loss_func_hyp = MSE(train_matrix_hyp.shape[0])
    model = None
    if regularization_derivative is None:
        model = Model(base_func_hyp, loss_func_hyp, learning_rate)
    else:
        model = Model(
            base_func_hyp, loss_func_hyp, learning_rate, regularization_derivative
        )

    train_and_test(
        model,
        train_matrix_hyp,
        train_labels_matrix,
        validation_matrix_hyp,
        validation_labels_matrix,
        epochs_num,
    )

    return model


# %%
def preprocess_data_matrix_1(data_matrix):
    answer = data_matrix[:, [0, 4]]
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
    answer = data_matrix[:, [0, 4, 5]]
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
        (
            data_matrix[:, 0],
            data_matrix[:, 3 + 1],
            data_matrix[:, 4 + 1],
            data_matrix[:, 3 + 1] * data_matrix[:, 4 + 1],
        )
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


# %%
def preprocess_data_matrix_4(data_matrix):
    answer = np.column_stack(
        (
            data_matrix[:, 0],
            data_matrix[:, 1],
            data_matrix[:, 3 + 1],
            data_matrix[:, 4 + 1],
            data_matrix[:, 3 + 1] * data_matrix[:, 4 + 1],
        )
    )
    # print(answer.shape)
    return answer


model = test_hypothesis(
    preprocess_data_matrix_4,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
)

print(model.test(preprocess_data_matrix_4(test_matrix), test_labels_matrix))


# %%
def preprocess_data_matrix_5(data_matrix):
    answer = np.column_stack(
        (data_matrix[:, 0], data_matrix[:, 1], data_matrix[:, 4], data_matrix[:, 5])
    )
    # print(answer.shape)
    return answer


model = test_hypothesis(
    preprocess_data_matrix_5,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
)


# print(model.test(preprocess_data_matrix_4(test_matrix), test_labels_matrix))
# %%
def preprocess_data_matrix_6(data_matrix):
    answer = np.column_stack(
        (
            data_matrix[:, 0],
            data_matrix[:, 1],
            data_matrix[:, 5],
            data_matrix[:, 4] * data_matrix[:, 5],
        )
    )
    # print(answer.shape)
    return answer


model = test_hypothesis(
    preprocess_data_matrix_6,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
)


# %%
def preprocess_data_matrix_7(data_matrix):
    answer = np.column_stack(
        (data_matrix[:, 0], data_matrix[:, 1], data_matrix[:, 4] * data_matrix[:, 5])
    )
    # print(answer.shape)
    return answer


model = test_hypothesis(
    preprocess_data_matrix_7,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
)


# %%
def preprocess_data_matrix_8(data_matrix):
    answer = np.column_stack((data_matrix[:, 0], data_matrix[:, 4] * data_matrix[:, 5]))
    # print(answer.shape)
    return answer


model = test_hypothesis(
    preprocess_data_matrix_8,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
)

# %% [markdown]
# # Best hyphothesis with regularization

# %%
model = test_hypothesis(
    preprocess_data_matrix_4,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
    L1_regularization(),
)

print(model.test(preprocess_data_matrix_4(test_matrix), test_labels_matrix), end="\n\n")

model = test_hypothesis(
    preprocess_data_matrix_4,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
    L2_regularization(),
)

print(model.test(preprocess_data_matrix_4(test_matrix), test_labels_matrix))


# %% [markdown]
# # Adding parameters to best hyphothesis
# %%
threshold = 1e-8


def preprocess_data_matrix_9(data_matrix):
    answer = np.column_stack(
        (
            data_matrix[:, 0],
            data_matrix[:, 1],
            # data_matrix[:, 1] ** 2,
            # data_matrix[:, 1] ** 3,
            # data_matrix[:, 2],
            # data_matrix[:, 3],
            # data_matrix[:, 6],
            # data_matrix[:, 7],
            data_matrix[:, 4],
            data_matrix[:, 4] ** 2,
            data_matrix[:, 4] ** 3,
            data_matrix[:, 4] ** 4,
            data_matrix[:, 4] ** 5,
            data_matrix[:, 4] ** 6,
            # data_matrix[:, 4] ** 7,
            np.where(data_matrix[:, 4] ** 7 < threshold, 0, data_matrix[:, 4] ** 7),
            # np.where(data_matrix[:, 4] ** 8 < threshold, 0, data_matrix[:, 4] ** 8),
            data_matrix[:, 5],
            # data_matrix[:, 5] ** 2,
            # data_matrix[:, 5] ** 3,
            data_matrix[:, 4] * data_matrix[:, 5],
            # data_matrix[:, 1] * data_matrix[:, 5]
            # data_matrix[:, 4] * data_matrix[:, 1]
            # data_matrix[:, 4] * data_matrix[:, 5] * data_matrix[:, 0]
        )
    )
    # print(answer.shape)
    return answer


model = test_hypothesis(
    preprocess_data_matrix_9,
    train_matrix,
    train_labels_matrix,
    validation_matrix,
    validation_labels_matrix,
    test_matrix,
    test_labels_matrix,
    1000,
    # L1_regularization(),
)


# %% [markdown]
# # Korekta błędu myślowego
# %%
def new_test_hypothesis(
    preprocess_data_func,
    train_data,
    train_labels,
    validation_data,
    validation_labels,
    epochs_num=700,
    regularization_derivative=None,
    learning_rate=0.01,
):
    train_matrix_hyp = preprocess_data_func(train_data)
    validation_matrix_hyp = preprocess_data_func(validation_data)
    base_func_hyp = OrdinaryBaseFunc(
        train_matrix_hyp.shape[1], train_matrix_hyp.shape[0]
    )
    loss_func_hyp = MSE(train_matrix_hyp.shape[0])
    model = None
    if regularization_derivative is None:
        model = Model(base_func_hyp, loss_func_hyp, learning_rate)
    else:
        model = Model(
            base_func_hyp, loss_func_hyp, learning_rate, regularization_derivative
        )

    train_labels_matrix = train_labels.to_numpy()
    validation_labels = validation_labels.to_numpy()
    train_and_test(
        model,
        train_matrix_hyp,
        train_labels_matrix,
        validation_matrix_hyp,
        validation_labels_matrix,
        epochs_num,
    )

    return model


# %%
# def preprocess_data_best_model(data_df: pd.DataFrame):
def preprocess_data_best_model(data_df):
    data_copy = data_df.copy()
    columns = pd.DataFrame()
    columns[0] = data_copy[0]
    columns[3] = data_copy[3]
    columns[4] = data_copy[4]
    columns["[3]*[4]"] = data_copy.iloc[:, 3] * data_copy.iloc[:, 4]
    powers = range(2, 200)

    # base_col = data_copy.iloc[:, 0]
    # temp = pd.DataFrame({f"[0]^{i}": base_col**i for i in powers})
    # columns = pd.concat([columns, temp], axis=1)

    base_col = data_copy.iloc[:, 3]
    temp = pd.DataFrame({f"[3]^{i}": base_col**i for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    # base_col = data_copy.iloc[:, 4]
    # temp = pd.DataFrame({f"[4]^{i}": base_col**i for i in powers})
    # columns = pd.concat([columns, temp], axis=1)

    assert len(columns.T) > 200, "Columns were not added"
    return standarize_matrix_and_add_ones(columns.to_numpy())


model = new_test_hypothesis(
    preprocess_data_best_model,
    train_set,
    train_labels,
    validation_set,
    validation_labels,
)


test_matrix = preprocess_data_best_model(test_set)
print(model.test(test_matrix, test_labels.to_numpy()))

test_error = model.base_func.predict(test_matrix) - test_labels
print("Test error std: ", np.std(test_error))


# %%
def preprocess_data_exponent3(data_df):
    data_copy = data_df.copy()
    columns = pd.DataFrame()
    columns[0] = data_copy[0]

    columns[1] = data_copy[1]
    columns[2] = data_copy[2]

    columns[3] = data_copy[3]
    columns[4] = data_copy[4]

    # columns[5] = data_copy[5]
    # columns[6] = data_copy[6]

    prod_of34 = data_copy.iloc[:, 3] * data_copy.iloc[:, 4]
    base_col = data_copy.iloc[:, 3]
    # prod_of34 = np.array(data_copy.iloc[:, 3] * data_copy.iloc[:, 4]).astype(np.float64)
    # prod_of34[prod_of34 == 0] = 1e-9
    powers = [1, 2]
    # powers = [1]
    temp = pd.DataFrame({f"([3]*[4])**{i}": np.pow(prod_of34, i) for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    powers = [
        0.5,
        0.75,
        # 1,
        # 1.25,
        1.5,
    ]
    # powers = []
    temp = pd.DataFrame(
        {
            f"|[3]*[4]|**{i}": np.sign(prod_of34) * np.pow(np.abs(prod_of34) * 1.0, i)
            for i in powers
        }
    )
    columns = pd.concat([columns, temp], axis=1)

    powers = range(2, 4)
    powers = []
    temp = pd.DataFrame(
        {f"([3])**{i}*[4]": np.pow(base_col, i) * data_copy.iloc[:, 4] for i in powers}
    )
    columns = pd.concat([columns, temp], axis=1)

    powers = range(2, 5)
    powers = []
    temp = pd.DataFrame({f"([3])**{i}": np.pow(base_col, i) for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    powers = [0.5, 1.5]
    powers = []
    temp = pd.DataFrame({f"|[3]|**{i}": np.pow(np.abs(base_col), i) for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    # powers = [0.5, 0.25, 0.05]
    powers = []
    powers += range(1, 10)
    powers += range(20, 24)
    # powers += range(60, 68)
    mean_3 = np.mean(base_col.to_numpy())
    temp = pd.DataFrame(
        {
            f"exp -([3]-mean[3])^2/{s}^2": np.exp(-((base_col - mean_3) ** 2) / (s**2))
            for s in powers
        }
    )
    columns = pd.concat([columns, temp], axis=1)
    # print(mean_3)

    # powers = []
    # powers += range(1, 100)
    # mean_of_prod34 = np.mean(prod_of34.to_numpy())
    # temp = pd.DataFrame(
    #     {
    #         f"exp -([3]-mean[3])^2/{s}^2": np.exp(
    #             -((prod_of34 - mean_of_prod34) ** 2) / (s**2)
    #         )
    #         for s in powers
    #     }
    # )
    # columns = pd.concat([columns, temp], axis=1)

    # powers = []
    # powers += range(1, 100)
    # mean_of_4 = np.mean(data_copy.iloc[:, 4])
    # temp = pd.DataFrame(
    #     {
    #         f"exp -([3]-mean[3])^2/{s}^2": np.exp(
    #             -((data_copy.iloc[:, 4] - mean_of_4) ** 2) / (s**2)
    #         )
    #         for s in powers
    #     }
    # )
    # columns = pd.concat([columns, temp], axis=1)

    # print(mean_3)
    return standarize_matrix_and_add_ones(columns.to_numpy())


model = new_test_hypothesis(
    preprocess_data_exponent3,
    train_set,
    train_labels,
    validation_set,
    validation_labels,
    700,
    # L1_regularization(),
)

test_matrix = preprocess_data_exponent3(test_set)
# print(test_matrix.shape, model.base_func.parameters().shape)
print(model.test(test_matrix, test_labels.to_numpy()))


# %% [markdown]
# # Experiments
# %%
def ultimate_test_hypothesis(
    preprocess_data_func,
    train_data,
    train_labels,
    validation_data,
    validation_labels,
    model,
    epochs_num=700,
):
    train_matrix_hyp = preprocess_data_func(train_data)
    validation_matrix_hyp = preprocess_data_func(validation_data)

    train_labels_matrix = train_labels.to_numpy()
    validation_labels = validation_labels.to_numpy()
    train_and_test(
        model,
        train_matrix_hyp,
        train_labels_matrix,
        validation_matrix_hyp,
        validation_labels_matrix,
        epochs_num,
    )

    return model


# %% [markdown]
# # Huber loss
# %%
# delta = 1e20
# epochs_num = 700
delta = 400.0
epochs_num = 1500
train_matrix_hyp = preprocess_data_exponent3(train_set)
validation_matrix_hyp = preprocess_data_exponent3(validation_set)
test_matrix_hyp = preprocess_data_exponent3(test_set)

base_func_hyp = OrdinaryBaseFunc(train_matrix_hyp.shape[1], train_matrix_hyp.shape[0])
loss_func_hyp = HUBER(train_matrix_hyp.shape[0], delta)
reg_parameter = 0.8
model = Model(
    base_func_hyp, loss_func_hyp, learning_rate, L1_regularization(reg_parameter)
)
# model = Model(base_func_hyp, loss_func_hyp, learning_rate)
# reg_parameter = 0.00001
# model = Model(base_func_hyp, loss_func_hyp, learning_rate, L2_regularization(reg_parameter))
train_and_test(
    model,
    train_matrix_hyp,
    train_labels_matrix,
    validation_matrix_hyp,
    validation_labels_matrix,
    epochs_num,
)


# print("Huber loss: ", model.test(test_matrix, test_labels.to_numpy()))
model.loss_func = MSE(train_matrix_hyp.shape[0])
print(
    "MSE validation: ",
    model.test(validation_matrix_hyp, validation_labels.to_numpy()),
    end="\n\n",
)

print(np.sort(model.base_func.parameters()), end="\n\n")
# print("MSE: ", model.test(test_matrix_hyp, test_labels.to_numpy()))

train_error = model.base_func.predict(train_matrix_hyp) - train_labels
print("Train error std: ", np.std(train_error))
validation_error = model.base_func.predict(validation_matrix_hyp) - validation_labels

# plt.boxplot(
#     [train_error, validation_error], tick_labels=["Train errors", "Validation errors"]
# )
# plt.title("Error boxplot")
# plt.show()

print("Validation errors:\n", validation_error.sort_values(ascending=False), "\n")
error_df = pd.DataFrame(
    {
        "Error": pd.concat([train_error, validation_error], ignore_index=True),
        "Set": ["Train"] * len(train_error) + ["Validation"] * len(validation_error),
    }
)

# Plot with seaborn
sns.boxplot(x="Set", y="Error", data=error_df)
plt.title("Error boxplot")
# plt.grid(True)
plt.show()


# %% [markdown]
# # Purely polynomial
# %%
def preprocess_data_polynomial(data_df):
    data_copy = data_df.copy()
    columns = pd.DataFrame()
    columns[0] = data_copy[0]

    # columns[1] = data_copy[1]
    # columns[2] = data_copy[2]

    columns[3] = data_copy[3]
    columns[4] = data_copy[4]

    # columns[5] = data_copy[5]
    # columns[6] = data_copy[6]

    prod_of34 = data_copy.iloc[:, 3] * data_copy.iloc[:, 4]
    base_col = data_copy.iloc[:, 3]
    feature_0 = data_copy.iloc[:, 0]
    feature_4 = data_copy.iloc[:, 4]

    # columns["2*3*4"] = base_col * feature_4 * feature_0
    # columns["2*3*6"] = base_col * feature_4 * data_copy.iloc[:, 6]
    # prod_of34 = np.array(data_copy.iloc[:, 3] * data_copy.iloc[:, 4]).astype(np.float64)
    # prod_of34[prod_of34 == 0] = 1e-9
    powers = [1, 2]
    # powers = [1]
    temp = pd.DataFrame({f"([3]*[4])**{i}": np.pow(prod_of34, i) for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    # temp = pd.DataFrame(
    #     {f"[3]*[{i}]": base_col * data_copy.iloc[:, i] for i in range(7) if i != 3}
    # )
    # columns = pd.concat([columns, temp], axis=1)
    # temp = pd.DataFrame(
    #     {
    #         f"|[3]|*[{i}]": np.abs(base_col) * data_copy.iloc[:, i]
    #         for i in range(7)
    #         if i != 3
    #     }
    # )
    # columns = pd.concat([columns, temp], axis=1)

    # temp = pd.DataFrame(
    #     {
    #         f"[3]*|[{i}]|": (base_col) * np.abs(data_copy.iloc[:, i])
    #         for i in range(7)
    #         if i != 3
    #     }
    # )
    # columns = pd.concat([columns, temp], axis=1)

    powers = [
        # 0.5,
        # 0.75,
        1,
        # 1.25,
        # 1.5,
        # 1.8,
        2,
    ]
    # powers += range(2, 10)
    # powers = []
    temp = pd.DataFrame(
        {
            f"|[3]*[4]|**{i}": np.sign(prod_of34) * np.pow(np.abs(prod_of34) * 1.0, i)
            for i in powers
        }
    )
    columns = pd.concat([columns, temp], axis=1)

    powers = range(2, 20)
    powers = []
    temp = pd.DataFrame(
        {f"([3])**{i}*[4]": np.pow(base_col, i) * data_copy.iloc[:, 4] for i in powers}
    )
    columns = pd.concat([columns, temp], axis=1)

    powers = []
    powers += range(2, 70)
    # powers += range(20, 30)
    # powers += range(50, 60)
    # powers += range(90, 100)
    # powers += range(145, 150)
    # powers = []
    temp = pd.DataFrame({f"([3])**{i}": np.pow(base_col, i) for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    powers = [0.1, 0.33, 0.5, 1.5]
    # powers += range(2, 100)
    powers = []
    temp = pd.DataFrame({f"|[3]|**{i}": np.pow(np.abs(base_col), i) for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    return standarize_matrix_and_add_ones(columns.to_numpy())


model = new_test_hypothesis(
    preprocess_data_polynomial,
    train_set,
    train_labels,
    validation_set,
    validation_labels,
    700,
    # L1_regularization(),
)

test_matrix = preprocess_data_polynomial(test_set)
# print(test_matrix.shape, model.base_func.parameters().shape)
print(model.test(test_matrix, test_labels.to_numpy()))


# %% [markdown]
# # All polynomial features
# %%

poly_degree = 7
treeshold = 20
k_value = 31


def preprocess_all_polynomial_features(data_df):
    global poly_degree
    # data_copy = data_df.copy()
    columns = pd.DataFrame()
    columns[0] = data_df[0]

    # columns[1] = data_df[1]
    columns[2] = data_df[2]

    columns[3] = data_df[3]
    columns[4] = data_df[4]

    # columns[5] = data_df[5]
    # columns[6] = data_df[6]

    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    # columns_matrix = poly.fit_transform(columns.to_numpy())
    columns_matrix = poly.fit_transform(columns)

    base_col = data_df[3]
    powers = []
    # powers += range(2, 100)
    # powers += range(20, 30)
    # powers += range(50, 60)
    # powers += range(90, 100)
    # powers += range(145, 150)
    # powers = []
    temp = pd.DataFrame({f"([3])**{i}": np.pow(base_col, i) for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    columns_matrix = standarize_matrix_and_add_ones(columns_matrix)
    return columns_matrix


reg_par = 1
# reg_par = 0.3

model = new_test_hypothesis(
    preprocess_all_polynomial_features,
    train_set,
    train_labels,
    validation_set,
    validation_labels,
    # 50000,
    700,
    L1_regularization(reg_par),
)

validation_matrix = preprocess_all_polynomial_features(validation_set)
print(
    "Analytical validation MSE: ",
    model.analytical_mse(validation_matrix, validation_labels.to_numpy()),
)

test_matrix = preprocess_all_polynomial_features(test_set)
# print(test_matrix.shape, model.base_func.parameters().shape)
print(model.test(test_matrix, test_labels.to_numpy()))

# %%
trained_parameters = model.base_func.parameters()
print(np.sort(trained_parameters), trained_parameters.shape)
sns.histplot(model.base_func.parameters(), bins=50, kde=True)
plt.title("uuu Sigma Rozkład wartości cech (małe wartości)")
plt.xlabel("Wartość")
plt.ylabel("Liczba")
plt.yscale("log")  # przy logarytmicznej osi łatwiej zauważyć małe wartości
plt.show()


# %% [markdown]
# # Reducing features count
# %%
def preprocess_all_polynomial_features_cut_loosely_correlated(data_df):
    global trained_parameters
    # data_copy = data_df.copy()
    columns = pd.DataFrame()
    columns[0] = data_df[0]

    # columns[1] = data_df[1]
    columns[2] = data_df[2]

    columns[3] = data_df[3]
    columns[4] = data_df[4]

    # columns[5] = data_df[5]
    # columns[6] = data_df[6]

    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    # columns_matrix = poly.fit_transform(columns.to_numpy())
    columns_matrix = poly.fit_transform(columns)

    columns_matrix = standarize_matrix_and_add_ones(columns_matrix)

    indices_to_remove = []
    for i in range(len(trained_parameters)):
        if np.abs(trained_parameters[i]) <= treeshold:
            indices_to_remove += [i]

    print(columns_matrix.shape)
    columns_matrix = np.delete(columns_matrix, indices_to_remove, axis=1)
    print(columns_matrix.shape)

    return columns_matrix


model = new_test_hypothesis(
    preprocess_all_polynomial_features_cut_loosely_correlated,
    train_set,
    train_labels,
    validation_set,
    validation_labels,
    700,
    L1_regularization(reg_par),
)

test_matrix = preprocess_all_polynomial_features_cut_loosely_correlated(test_set)
# print(test_matrix.shape, model.base_func.parameters().shape)
print(model.test(test_matrix, test_labels.to_numpy()))


# %%
def cut_and_leave_top_k(data_df):
    global trained_parameters, k_value
    # data_copy = data_df.copy()
    columns = pd.DataFrame()
    columns[0] = data_df[0]

    # columns[1] = data_df[1]
    columns[2] = data_df[2]

    columns[3] = data_df[3]
    columns[4] = data_df[4]

    # columns[5] = data_df[5]
    # columns[6] = data_df[6]

    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    # columns_matrix = poly.fit_transform(columns.to_numpy())
    columns_matrix = poly.fit_transform(columns)

    columns_matrix = standarize_matrix_and_add_ones(columns_matrix)

    sorted_by_abs = np.sort(np.abs(trained_parameters))
    temp_threeshold = sorted_by_abs[-k_value]
    indices_to_remove = []
    for i in range(len(trained_parameters)):
        if np.abs(trained_parameters[i]) < temp_threeshold:
            indices_to_remove += [i]

    print(columns_matrix.shape)
    columns_matrix = np.delete(columns_matrix, indices_to_remove, axis=1)
    print(columns_matrix.shape)

    # Calculate the correlation matrix
    corr = pd.DataFrame(columns_matrix).corr()

    show_plot = False
    # Plot the heatmap
    if show_plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            cbar=True,
            square=True,
        )
        plt.title("Correlation Matrix")
        plt.show()

    # sorted = np.sort(corr.to_numpy().flatten())
    # cut_small = np.ma.masked_less(sorted, 0.9)
    # print(cut_small)

    return columns_matrix


model = new_test_hypothesis(
    cut_and_leave_top_k,
    train_set,
    train_labels,
    validation_set,
    validation_labels,
    700,
    None,
    # regularization_derivative=L1_regularization(reg_par),
    learning_rate=0.01,
)

validation_matrix = cut_and_leave_top_k(validation_set)
print(
    "Analytical validation MSE: ",
    model.analytical_mse(validation_matrix, validation_labels.to_numpy()),
)

test_matrix = cut_and_leave_top_k(test_set)
# print(test_matrix.shape, model.base_func.parameters().shape)
print(model.test(test_matrix, test_labels.to_numpy()))

print(
    "Analytical test MSE: ",
    model.analytical_mse(test_matrix, test_labels.to_numpy()),
)


class BatchModel(Model):
    def train(self, epochs_num, data, labels, loss_at_epoch=None):
        # data_copy = data.copy()
        # labels_copy = labels.copy()

        # data_with_labels = np.hstack((data_copy, labels_copy.reshape(-1, 1)))
        # print(
        #     f"Data: {data_copy.shape}",
        #     f"Labels: {labels.reshape(-1, 1).shape}",
        #     f"Labeled data: {data_with_labels.shape}",
        # )

        random_seed = 42
        np.random.seed(random_seed)

        rows_count = len(labels)
        indices = np.arange(rows_count)

        for i in range(epochs_num):
            np.random.shuffle(indices)
            all_batches_indices = np.array_split(indices, rows_count // self.batch_size)

            # time.sleep(1)

            for batch_indices in all_batches_indices:
                batch = data[batch_indices]
                batch_labels = labels[batch_indices]
                # print(len(batch))

                gradient = self._get_gradient(batch, batch_labels)
                new_parameters = (
                    self.base_func.parameters() - self.learning_rate * gradient
                )

                self.base_func.update_parameters(new_parameters)

            if loss_at_epoch is not None:
                loss_at_epoch.append(self.test(data, labels))

            # if (i % 10) == 0:
            #     self.learning_rate *= 0.99

    def __init__(
        self,
        base_func: BaseFunc,
        loss_func: LossFunc,
        learning_rate,
        regularization_derivative=lambda x: np.zeros(x.shape),
        batch_size=32,
    ):
        super().__init__(
            base_func,
            loss_func,
            learning_rate,
            regularization_derivative,
        )

        self.batch_size = batch_size


# %% [markdown]
# # Train with batches
# %%
preprocess_func = cut_and_leave_top_k
preprocess_func = preprocess_all_polynomial_features

train_matrix_hyp = preprocess_func(train_set)
validation_matrix_hyp = preprocess_func(validation_set)
test_matrix_hyp = preprocess_func(test_set)

base_func_hyp = OrdinaryBaseFunc(train_matrix_hyp.shape[1], train_matrix_hyp.shape[0])
loss_func_hyp = MSE(train_matrix_hyp.shape[0])
# delta = 400.0
# loss_func_hyp = HUBER(train_matrix_hyp.shape[0], 100)

# reg_parameter = 0.8
# model = BatchModel(base_func_hyp, loss_func_hyp, learning_rate, L1_regularization(reg_parameter))

model = BatchModel(base_func_hyp, loss_func_hyp, learning_rate, batch_size=32)
# model = Model(base_func_hyp, loss_func_hyp, 0.01)
# reg_parameter = 0.00001
# model = Model(base_func_hyp, loss_func_hyp, 0.01, L2_regularization(reg_parameter))

epochs_num = 10000

train_and_test(
    model,
    train_matrix_hyp,
    train_labels_matrix,
    validation_matrix_hyp,
    validation_labels_matrix,
    epochs_num,
    skip_this_many_in_plot=500,
)

model.loss_func = MSE(train_matrix_hyp.shape[0])
print(
    "MSE validation: ",
    model.test(validation_matrix_hyp, validation_labels.to_numpy()),
    end="\n\n",
)
