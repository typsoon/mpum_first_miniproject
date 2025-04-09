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
# %matplotlib inline

# %%
from abc import abstractmethod
import math
import numpy as np
import pandas as pd
import copy
import math
from pandas.core.indexes.multi import get_adjustment
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Any
import types
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from collections import defaultdict

# from tqdm import tqdm
from tqdm.notebook import tqdm

from typing import Tuple
from typing import cast


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

    @abstractmethod
    def reset(self):
        pass


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

    def get_analytical_pars(self, data, labels):
        pseudo_inv = np.linalg.pinv(data.T @ data)
        return pseudo_inv @ data.T @ labels

    def L2_ref_analytical_mse(self, data, labels, lambda_val=0.02):
        old_parameters = self.base_func.parameters()

        dt_T_times_dt = data.T @ data
        pseudo_inv = np.linalg.pinv(
            dt_T_times_dt + lambda_val * np.eye(dt_T_times_dt.shape[0])
        )
        analytical_parameters = pseudo_inv @ data.T @ labels
        self.base_func.update_parameters(analytical_parameters)

        answer = self.test(data, labels)
        self.base_func.update_parameters(old_parameters)
        return answer

    def reset(self):
        self.base_func.reset()


# %% [markdown]
# # Concrete implementations


# %%
class OrdinaryBaseFunc(BaseFunc):
    def __init__(self, degree, number_of_inputs) -> None:
        # print(degree)
        self.degree = degree
        self.reset()

    def predict(self, data):
        # temp = data @ self._parameters
        # print(self._parameters.shape)

        return data @ self._parameters

    def derivative(self, data):
        return data

    def reset(self):
        self._parameters = np.random.randn(self.degree)


# %%
class MSE(LossFunc):
    def loss_value(self, prediction, ground_truth):
        return np.mean((prediction - ground_truth) ** 2)
        return (
            # (1 / (2 * self.number_of_inputs))
            # * (ground_truth - prediction).T
            # @ (ground_truth - prediction)
            # (1 / (len(prediction)))
            # * (ground_truth - prediction).T
            # @ (ground_truth - prediction)
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


def elastic_net_regularization(lambda_coefficient_1=0.02, lambda_coefficient_2=0.5):
    def elastic_net_reg_func(parameters):
        temp_parameters = np.array(parameters)
        temp_parameters[0] = 0
        return (
            lambda_coefficient_1 * np.sign(temp_parameters)
            + lambda_coefficient_2 * 2 * temp_parameters.T
        )

    return elastic_net_reg_func


# %% [markdown]
# # Load and display data

# %%
df = pd.read_csv("dane.data", delimiter=r"\s+", header=None, decimal=",")


# %%
def divide_data(data, test_ratio) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # return train_test_split(data, train_size=train_ratio,  random_state=42)
    ans = train_test_split(data, test_size=test_ratio, random_state=6)
    return cast(Tuple[pd.DataFrame, pd.DataFrame], ans)
    # return train_test_split(data, train_size=train_ratio, stratify=housing["median_house_value"], random_state=8)


train_set, test_set = divide_data(df, test_ratio=0.2)
train_set, validation_set = divide_data(train_set, test_ratio=0.2 / 0.6)


# %%
train_labels = train_set.pop(train_set.columns[7])  # pyright: ignore
validation_labels = validation_set.pop(validation_set.columns[7])  # pyright: ignore
test_labels = test_set.pop(test_set.columns[7])  # pyright: ignore
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
correlations.sort_values(ascending=False)  # pyright: ignore

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

learning_rate = 0.01
some_models = []


def train_and_test(
    model,
    train_matrix,
    train_labels_matrix,
    test_matrix,
    test_labels_matrix,
    epochs_num=700,
    skip_this_many_in_plot=0,
    lambda_val=0.02,
):
    loss_at_epochs = [model.test(train_matrix, train_labels_matrix)]
    model.train(epochs_num, train_matrix, train_labels_matrix, loss_at_epochs)
    print(model.test(train_matrix, train_labels_matrix))
    print(model.test(test_matrix, test_labels_matrix))

    print("Analytical score: ", model.analytical_mse(test_matrix, test_labels_matrix))
    print(
        "Analytical L2 regularization score: ",
        model.L2_ref_analytical_mse(test_matrix, test_labels_matrix, lambda_val),
    )

    loss_at_epochs = loss_at_epochs[skip_this_many_in_plot:]
    plt.plot(range(len(loss_at_epochs)), loss_at_epochs, marker="o", linewidth=0.0025)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Loss at epochs")
    plt.grid(True)
    plt.show()


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


# %% [markdown]
# # All polynomial features
# %%

poly_degree = 7
poly_degree = 8
# poly_degree = 9
treeshold = 20
k_value = 31
k_value = 80


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

# poly_degree = 7
all_polynomial_no_batch = copy.deepcopy(model)
some_models.append(
    (
        "All polynomial no batch",
        all_polynomial_no_batch,
        preprocess_all_polynomial_features,
    )
)

# %% [markdown]
# # Saving array
# %%
temp_train_set = df.copy()
temp_train_labels = temp_train_set.pop(temp_train_set.columns[7])  # pyright: ignore

train_matrix_hyp = preprocess_all_polynomial_features(temp_train_set)
trained_parameters = model.get_analytical_pars(
    train_matrix_hyp,
    temp_train_labels,
)

np.set_printoptions(threshold=int(1e9))
with open("parameters", "w") as f:
    f.write(np.array2string(trained_parameters))

error = (train_matrix_hyp @ trained_parameters) - temp_train_labels
SSE = np.sum(error**2)
RMSE = np.sqrt(SSE / len(temp_train_labels))
print(RMSE)

# %%
trained_parameters = model.base_func.parameters()
print(np.sort(trained_parameters), trained_parameters.shape)
sns.histplot(model.base_func.parameters(), bins=50, kde=True)
plt.title("Rozkład wartości cech (małe wartości)")
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

    # print(columns_matrix.shape)
    columns_matrix = np.delete(columns_matrix, indices_to_remove, axis=1)
    # print(columns_matrix.shape)

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

no_batches_cut_by_treeshold = copy.deepcopy(model)
some_models.append(
    (
        "No batches cut by treeshold",
        no_batches_cut_by_treeshold,
        preprocess_all_polynomial_features_cut_loosely_correlated,
    )
)


# %%


def get_cut_and_leave_top_k(trained_parameters, k_value):
    def cut_and_leave_top_k(data_df):
        nonlocal trained_parameters, k_value
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

        # print(columns_matrix.shape)
        columns_matrix = np.delete(columns_matrix, indices_to_remove, axis=1)
        # print(columns_matrix.shape)

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

    return cut_and_leave_top_k


# %%

temp_cut_and_leave = get_cut_and_leave_top_k(trained_parameters.copy(), k_value)

model = new_test_hypothesis(
    temp_cut_and_leave,
    train_set,
    train_labels,
    validation_set,
    validation_labels,
    700,
    None,
    # regularization_derivative=L1_regularization(reg_par),
    learning_rate=0.01,
)

validation_matrix = temp_cut_and_leave(validation_set)
print(
    "Analytical validation MSE: ",
    model.analytical_mse(validation_matrix, validation_labels.to_numpy()),
)

test_matrix = temp_cut_and_leave(test_set)
# print(test_matrix.shape, model.base_func.parameters().shape)
print(model.test(test_matrix, test_labels.to_numpy()))

print(
    "Analytical test MSE: ",
    model.analytical_mse(test_matrix, test_labels.to_numpy()),
)

cut_and_leave_top_k_no_batches = copy.deepcopy(model)
some_models.append(
    (
        "Cut and leave top k no batches",
        cut_and_leave_top_k_no_batches,
        copy.deepcopy(temp_cut_and_leave),
    )
)


# %%
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

        iter_range = tqdm(range(epochs_num), "Epoch: ")
        batches_count = rows_count // self.batch_size
        # iter_range = range(epochs_num)
        for i in iter_range:
            np.random.shuffle(indices)

            # time.sleep(1)

            # all_batches_indices = np.array_split(indices, batches_count)
            # for batch_indices in all_batches_indices:
            if batches_count == 0:
                continue

            for batch_indices in np.array_split(indices, batches_count):
                # for batch_indices in range(0, rows_count, 32):
                # batch_indices = indices[batch_indices : batch_indices + 32]
                batch = data[batch_indices]
                batch_labels = labels[batch_indices]
                # print(len(batch))

                gradient = self._get_gradient(batch, batch_labels)
                new_parameters = (
                    self.base_func.parameters() - self.learning_rate * gradient
                )

                self.base_func.update_parameters(new_parameters)

            if loss_at_epoch is not None:
                current_loss = self.test(data, labels)
                loss_at_epoch.append(current_loss)
                if i % 1000 == 0:
                    iter_range.set_postfix(loss=f"{current_loss:.2f}")

            if (i > 8e5) and ((i % 10) == 0):
                self.learning_rate *= 0.9999

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
# epochs_num = 200000
epochs_num = 2000

reg_parameter = 0.0005
train_and_test(
    model,
    train_matrix_hyp,
    train_labels_matrix,
    validation_matrix_hyp,
    validation_labels_matrix,
    epochs_num,
    skip_this_many_in_plot=20000,
    lambda_val=reg_parameter,
)

model.loss_func = MSE(train_matrix_hyp.shape[0])
print(
    "MSE validation: ",
    model.test(validation_matrix_hyp, validation_labels.to_numpy()),
    end="\n\n",
)

all_polynomial_batch_model = copy.deepcopy(model)
some_models.append(
    (
        "All polynomial batch model",
        all_polynomial_batch_model,
        preprocess_func,
    )
)
# %% [markdown]
# # Elastic net regularization
# %%

preprocess_func = preprocess_all_polynomial_features

train_matrix_hyp = preprocess_func(train_set)
validation_matrix_hyp = preprocess_func(validation_set)
test_matrix_hyp = preprocess_func(test_set)

base_func_hyp = OrdinaryBaseFunc(train_matrix_hyp.shape[1], train_matrix_hyp.shape[0])
loss_func_hyp = MSE(train_matrix_hyp.shape[0])
# delta = 400.0
# loss_func_hyp = HUBER(train_matrix_hyp.shape[0], 100)


reg_parameter_1 = 0.0001
reg_parameter_2 = 0.00000001
model = Model(
    base_func_hyp,
    loss_func_hyp,
    0.01,
    elastic_net_regularization(
        reg_parameter_1,
        reg_parameter_2,
    ),
)

epochs_num = 10000
# epochs_num = 200000
epochs_num = 2000

reg_parameter = 0.0005
train_and_test(
    model,
    train_matrix_hyp,
    train_labels_matrix,
    validation_matrix_hyp,
    validation_labels_matrix,
    epochs_num,
    skip_this_many_in_plot=20000,
    lambda_val=reg_parameter,
)

model.loss_func = MSE(train_matrix_hyp.shape[0])
print(
    "MSE validation: ",
    model.test(validation_matrix_hyp, validation_labels.to_numpy()),
    end="\n\n",
)


# %% [markdown]
# ## Another try
# %%
k_value = 120
preprocess_func = get_cut_and_leave_top_k(trained_parameters.copy(), k_value)

train_matrix_hyp = preprocess_func(train_set)
validation_matrix_hyp = preprocess_func(validation_set)
test_matrix_hyp = preprocess_func(test_set)

base_func_hyp = OrdinaryBaseFunc(train_matrix_hyp.shape[1], train_matrix_hyp.shape[0])
loss_func_hyp = MSE(train_matrix_hyp.shape[0])
# delta = 400.0
# loss_func_hyp = HUBER(train_matrix_hyp.shape[0], 100)

# reg_parameter = 0.8
# model = BatchModel(base_func_hyp, loss_func_hyp, learning_rate, L1_regularization(reg_parameter))

# model = BatchModel(base_func_hyp, loss_func_hyp, learning_rate,  batch_size=64)
# model = Model(base_func_hyp, loss_func_hyp, 0.01)

reg_parameter_1 = 0.00001
reg_parameter_2 = 0.0005

model = BatchModel(
    base_func_hyp,
    loss_func_hyp,
    0.01,
    elastic_net_regularization(reg_parameter_1, reg_parameter_2),
    batch_size=64,
)

epochs_num = 60000
# epochs_num = 120000
epochs_num = 200000
# epochs_num = 400000
epochs_num = 800000
#
#
epochs_num = 2000
# epochs_num = 40000
#
train_and_test(
    model,
    train_matrix_hyp,
    train_labels_matrix,
    validation_matrix_hyp,
    validation_labels_matrix,
    epochs_num,
    skip_this_many_in_plot=20000,
    lambda_val=reg_parameter,
)

model.loss_func = MSE(train_matrix_hyp.shape[0])
print(
    "MSE validation: ",
    model.test(validation_matrix_hyp, validation_labels.to_numpy()),
    end="\n\n",
)

k_top_batch_model = copy.deepcopy(model)
some_models.append(
    (
        "k top batch model",
        k_top_batch_model,
        preprocess_func,
    )
)

# %% [markdown]
# # Elastic net regularization second try
# %%
preprocess_func = get_cut_and_leave_top_k(trained_parameters.copy(), k_value)

train_matrix_hyp = preprocess_func(train_set)
validation_matrix_hyp = preprocess_func(validation_set)
test_matrix_hyp = preprocess_func(test_set)

base_func_hyp = OrdinaryBaseFunc(train_matrix_hyp.shape[1], train_matrix_hyp.shape[0])
loss_func_hyp = MSE(train_matrix_hyp.shape[0])
# delta = 400.0
# loss_func_hyp = HUBER(train_matrix_hyp.shape[0], 100)

# reg_parameter = 0.8
# model = BatchModel(base_func_hyp, loss_func_hyp, learning_rate, L1_regularization(reg_parameter))

# model = BatchModel(base_func_hyp, loss_func_hyp, learning_rate,  batch_size=64)
# model = Model(base_func_hyp, loss_func_hyp, 0.01)

reg_parameter_1 = 0.00001
reg_parameter_2 = 0.0005

model = BatchModel(
    base_func_hyp,
    loss_func_hyp,
    0.01,
    elastic_net_regularization(reg_parameter_1, reg_parameter_2),
    batch_size=64,
)

epochs_num = 60000
# epochs_num = 120000
epochs_num = 200000
# epochs_num = 400000
epochs_num = 800000
#
#
epochs_num = 2000
# epochs_num = 40000
#
train_and_test(
    model,
    train_matrix_hyp,
    train_labels_matrix,
    validation_matrix_hyp,
    validation_labels_matrix,
    epochs_num,
    skip_this_many_in_plot=20000,
    lambda_val=reg_parameter,
)

model.loss_func = MSE(train_matrix_hyp.shape[0])
print(
    "MSE validation: ",
    model.test(validation_matrix_hyp, validation_labels.to_numpy()),
    end="\n\n",
)


# %% [markdown]
# # Trying out NAdam
# %%
class Nadam:
    def __init__(self, shape, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def step(self, params, grads):
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        nesterov_term = self.beta1 * m_hat + ((1 - self.beta1) * grads) / (
            1 - self.beta1**self.t
        )

        update = self.lr * nesterov_term / (np.sqrt(v_hat) + self.epsilon)
        return params - update


# %%
class BatchModelWithNadam(Model):
    def train(self, epochs_num, data, labels, loss_at_epoch=None):
        random_seed = 42
        np.random.seed(random_seed)

        rows_count = len(labels)
        indices = np.arange(rows_count)

        iter_range = tqdm(range(epochs_num), "Epoch: ")
        batches_count = rows_count // self.batch_size
        # iter_range = range(epochs_num)
        epoch_idx = 0
        for i in iter_range:
            np.random.shuffle(indices)

            for batch_indices in np.array_split(indices, batches_count):
                batch = data[batch_indices]
                batch_labels = labels[batch_indices]

                gradient = self._get_gradient(batch, batch_labels)

                new_parameters = None
                if epoch_idx <= 4e5:
                    new_parameters = (
                        self.base_func.parameters() - self.learning_rate * gradient
                    )
                else:
                    new_parameters = self.nadam.step(
                        self.base_func.parameters(), gradient
                    )

                self.base_func.update_parameters(new_parameters)

            if loss_at_epoch is not None:
                current_loss = self.test(data, labels)
                loss_at_epoch.append(current_loss)
                if i % 1000 == 0:
                    iter_range.set_postfix(loss=f"{current_loss:.2f}")

            # if (i > 8e5) and ((i % 10) == 0):
            #     self.learning_rate *= 0.9999
            epoch_idx += 1

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
        self.nadam = Nadam(self.base_func.parameters().shape, learning_rate)


# %%
k_value = 120
preprocess_func = get_cut_and_leave_top_k(trained_parameters.copy(), k_value)

train_matrix_hyp = preprocess_func(train_set)
validation_matrix_hyp = preprocess_func(validation_set)
test_matrix_hyp = preprocess_func(test_set)

base_func_hyp = OrdinaryBaseFunc(train_matrix_hyp.shape[1], train_matrix_hyp.shape[0])
loss_func_hyp = MSE(train_matrix_hyp.shape[0])

reg_parameter = 0.00001
model = BatchModelWithNadam(base_func_hyp, loss_func_hyp, learning_rate, batch_size=64)
# model = Model(base_func_hyp, loss_func_hyp, 0.01)
# reg_parameter = 0.00001
model = Model(base_func_hyp, loss_func_hyp, 0.01, L2_regularization(reg_parameter))

epochs_num = 60000
# epochs_num = 120000
# epochs_num = 200000
# epochs_num = 400000
epochs_num = 800000

epochs_num = 40000

# %%
train_and_test(
    model,
    train_matrix_hyp,
    train_labels_matrix,
    validation_matrix_hyp,
    validation_labels_matrix,
    epochs_num,
    skip_this_many_in_plot=20000,
    lambda_val=reg_parameter,
)

model.loss_func = MSE(train_matrix_hyp.shape[0])
print(
    "MSE validation: ",
    model.test(validation_matrix_hyp, validation_labels.to_numpy()),
    end="\n\n",
)

batch_nadam = copy.deepcopy(model)
some_models.append(
    (
        "Batch NAdam",
        batch_nadam,
        preprocess_func,
    )
)


# %%
# for _ in tqdm(range(200)):
#     print(_)
fractions = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
# fractions = [0.125]

train_data = (train_set, train_labels)
test_data = (test_set, test_labels)


def make_plot(models_data, epochs_num, fractions, sample_count, train_data, test_data):
    train_set, train_labels = train_data
    test_set, test_labels = test_data

    input_size = len(train_set)
    indices = np.arange(len(train_set))

    plot_data = []

    # for model_name, model, preprocess_func in models_data:
    for model_name, model, preprocess_func in tqdm(models_data, "Model: "):
        # print("ASSUUASAISUAISUAIAUSI ", model_name)
        temp_train_matrix = preprocess_func(train_set.copy())
        temp_train_labels = train_labels.copy().to_numpy()

        # print(temp_train_matrix.shape[0], temp_train_labels.shape[0])
        assert temp_train_matrix.shape[0] == temp_train_labels.shape[0], (
            f"This should be equal {temp_train_matrix.shape[1]} {temp_train_labels.shape[0]}"
        )
        # pyright: ignore

        # model: Model
        for fraction in fractions:
            sample_size = math.ceil(input_size * fraction)
            if sample_size == 0:
                continue

            test_scores = []
            for _ in range(sample_count):
                model.reset()
                # print(indices)
                # sample_indices = np.random.choice(indices, int(input_size * fraction))
                # print(sample_size
                sample_indices = np.random.choice(indices, sample_size)
                # print(sample_indices)

                # print(len(temp_train_matrix), len(temp_train_labels))

                sample_train_matrix = temp_train_matrix[sample_indices]
                sample_train_labels = temp_train_labels[sample_indices]

                # print(
                #     sample_train_matrix.shape,
                #     sample_train_labels.shape,
                #     model.base_func.parameters().shape,
                # )
                #
                assert (
                    sample_train_matrix.shape[1]
                    == model.base_func.parameters().shape[0]
                ), (
                    f"This sample sizes should be equal {sample_train_matrix.shape[1]} {model.base_func.parameters().shape[0]}"
                )

                model.train(epochs_num, sample_train_matrix, sample_train_labels)
                test_score = model.test(preprocess_func(test_set), test_labels)
                test_score = np.clip(test_score, 0, 1e5)

                test_scores += [test_score]

            plot_data.append((model_name, fraction, np.mean(test_scores)))

    return plot_data


# %% [markdown]
# # Make plot
# %%
# print(some_models)

plot_data = make_plot(
    some_models,
    2000,
    fractions,
    10,
    train_data,
    test_data,
)

# %%
grouped = defaultdict(lambda: {"x": [], "y": []})

# Grupowanie po modelu
for model_name, fraction, test_score in plot_data:
    grouped[model_name]["x"].append(fraction)
    # test_score = np.clip(test_score, 0, 1e5)
    test_score = np.clip(test_score, 0, 1.3e4)
    grouped[model_name]["y"].append(test_score)

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
for model_name, values in grouped.items():
    plt.plot(values["x"], values["y"], label=model_name, marker="o")

plt.xlabel("Fraction")
plt.ylabel("Test Score")
plt.title("Model Performance vs Fraction")
plt.xlim(0, 1.01)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
