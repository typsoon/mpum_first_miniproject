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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Any

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
    def loss_value(self, prediction, ground_truth):
        return (
            (1 / (2 * self.number_of_inputs))
            * (ground_truth - prediction).T
            @ (ground_truth - prediction)
            # (1 / (2 * len(prediction)))
            # * (ground_truth - prediction).T
            # @ (ground_truth - prediction)
        )

    def loss_derivative(self, prediction, ground_truth, parameters=None):
        return 1 / self.number_of_inputs * (prediction - ground_truth)
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
def standarize_matrix_and_add_ones(data_matrix):
    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0)
    std[std == 0] = 1

    standarized = (data_matrix - mean) / std
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


model = Model(base_func_reg_L2, loss_func_reg_L2, 0.01, L2_regularization)

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


model = Model(base_func_reg_L1, loss_func_reg_L1, 0.01, L1_regularization)

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
        model = Model(base_func_hyp, loss_func_hyp, 0.01)
    else:
        model = Model(base_func_hyp, loss_func_hyp, 0.01, regularization_derivative)

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
    L1_regularization,
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
    L2_regularization,
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
    # L1_regularization,
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
):
    train_matrix_hyp = preprocess_data_func(train_data)
    validation_matrix_hyp = preprocess_data_func(validation_data)
    base_func_hyp = OrdinaryBaseFunc(
        train_matrix_hyp.shape[1], train_matrix_hyp.shape[0]
    )
    loss_func_hyp = MSE(train_matrix_hyp.shape[0])
    model = None
    if regularization_derivative is None:
        model = Model(base_func_hyp, loss_func_hyp, 0.01)
    else:
        model = Model(base_func_hyp, loss_func_hyp, 0.01, regularization_derivative)

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


# %%
def preprocess_data_exponent3(data_df):
    data_copy = data_df.copy()
    columns = pd.DataFrame()
    columns[0] = data_copy[0]
    columns[3] = data_copy[3]
    columns[4] = data_copy[4]

    prod_of34 = data_copy.iloc[:, 3] * data_copy.iloc[:, 4]
    # prod_of34 = np.array(data_copy.iloc[:, 3] * data_copy.iloc[:, 4]).astype(np.float64)
    # prod_of34[prod_of34 == 0] = 1e-9
    powers = [1, 2]
    temp = pd.DataFrame({f"([3]*[4])**{i}": np.pow(prod_of34, i) for i in powers})
    columns = pd.concat([columns, temp], axis=1)

    base_col = data_copy.iloc[:, 3]

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
    # L1_regularization,
)

test_matrix = preprocess_data_exponent3(test_set)
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
delta = 75.0
delta = 1e20
epochs_num = 700
# epochs_num = 3000
train_matrix_hyp = preprocess_data_exponent3(train_set)
validation_matrix_hyp = preprocess_data_exponent3(validation_set)
test_matrix_hyp = preprocess_data_exponent3(test_set)

base_func_hyp = OrdinaryBaseFunc(train_matrix_hyp.shape[1], train_matrix_hyp.shape[0])
loss_func_hyp = HUBER(train_matrix_hyp.shape[0], delta)
model = Model(base_func_hyp, loss_func_hyp, 0.01)
# model = Model(base_func_hyp, loss_func_hyp, 0.01, L1_regularization)
# model = Model(base_func_hyp, loss_func_hyp, 0.01, L2_regularization)
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
    "MSE validation: ", model.test(validation_matrix_hyp, validation_labels.to_numpy())
)
# print("MSE: ", model.test(test_matrix_hyp, test_labels.to_numpy()))
