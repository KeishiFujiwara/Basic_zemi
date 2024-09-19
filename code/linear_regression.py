import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


class RegressionModel:
    def __init__(self, data_file, target_column):
        self.df = pd.read_csv(data_file)
        self.df['const'] = np.ones(len(self.df))  # 定数項を追加
        cols = ['const'] + self.df.columns[:-1].tolist()
        self.df = self.df[cols]
        self.X = self.df.iloc[:, :-1].values
        self.Y = self.df.iloc[:, target_column].values

    def compute_inverse_beta(self):
        # 逆行列を使った回帰係数の計算
        self.beta_inverse = np.linalg.inv(self.X.T @ self.X) @ (self.X.T @ self.Y)
        return self.beta_inverse

    def manual_regression_metrics(self, Y_hat):
        # R^2 と MAE の手動計算
        RSS = np.sum((self.Y - Y_hat) ** 2)
        TSS = np.sum((self.Y - np.mean(self.Y)) ** 2)
        r2 = 1 - (RSS / TSS)
        mae = np.mean(np.abs(self.Y - Y_hat))
        return r2, mae

    def scikit_learn_regression(self):
        # Scikit-learnによる回帰モデルのトレーニングと評価
        x_train, x_test, y_train, y_test = train_test_split(
            self.df.iloc[:, 1:3], self.df.iloc[:, 3], test_size=0.2, random_state=1
        )
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        pred_lr = lr.predict(x_test)
        r2 = r2_score(y_test, pred_lr)
        mae = mean_absolute_error(y_test, pred_lr)
        return lr.coef_, lr.intercept_, r2, mae

    def gradient_descent(self, learning_rate=0.01, iterations=1000):
        # 勾配降下法による回帰
        beta = np.zeros(self.X.shape[1])
        beta[0] = -1
        m = len(self.df)

        for i in range(iterations):
            y_pred = self.X @ beta
            dbeta = (1 / m) * (self.X.T @ (y_pred - self.Y))
            beta -= learning_rate * dbeta

            if i % 100 == 0:
                cost = self.compute_cost(beta)
                #print(f"Iteration {i}: Cost = {cost}")

        self.beta_gradient = beta
        return beta

    def compute_cost(self, beta):
        # コスト関数の計算
        m = len(self.df)
        total_cost = np.sum((self.X @ beta - self.Y) ** 2) / (2 * m)
        return total_cost

    def print_results(self, method, coef, intercept, r2, mae):
        print(f"{method} Results:")
        print(f"Coef = {coef}")
        print(f"Intercept = {intercept}")
        print(f"R^2: {r2:.5f}")
        print(f"MAE: {mae:.3f}")
        print('-' * 30)

    def run(self):
        # 逆行列による結果
        beta_inverse = self.compute_inverse_beta()
        Y_hat_inverse = self.X @ beta_inverse
        r2_manual, mae_manual = self.manual_regression_metrics(Y_hat_inverse)
        self.print_results("Inverse Matrix", beta_inverse[1:], beta_inverse[0], r2_manual, mae_manual)

        # Scikit-learnによる結果
        coef, intercept, r2_sklearn, mae_sklearn = self.scikit_learn_regression()
        self.print_results("Scikit-learn", coef, intercept, r2_sklearn, mae_sklearn)

        # 勾配降下法による結果
        beta_gradient = self.gradient_descent()
        Y_hat_gradient = self.X @ beta_gradient
        r2_gradient, mae_gradient = self.manual_regression_metrics(Y_hat_gradient)
        self.print_results("Gradient Descent", beta_gradient[1:], beta_gradient[0], r2_gradient, mae_gradient)


# 使用例
if __name__ == "__main__":
    model = RegressionModel('/Users/kc/Desktop/Basic_zemi/data/sampledata.csv', target_column=3)
    model.run()
