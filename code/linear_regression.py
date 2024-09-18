import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# データの読み込み
df = pd.read_csv('/Users/kc/Desktop/Basic_zemi/data/sampledata.csv')

# 列に定数項（切片用の列）を追加
df['const'] = np.ones(len(df))
cols = ['const'] + df.columns[:-1].tolist()
df = df[cols]

# 特徴量 (X) と目的変数 (Y) の抽出
X = df.iloc[:, :-1].values
Y = df.iloc[:, 3].values

# 逆行列を用いた回帰係数 β の計算
beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
Y_hat = X @ beta

# R^2 と MAE の計算
RSS = np.sum((Y - Y_hat) ** 2)
TSS = np.sum((Y - np.mean(Y)) ** 2)
r2_manual = 1 - (RSS / TSS)
mae_manual = np.mean(np.abs(Y - Y_hat))

# 結果出力
print("Coef = ", beta[1:3])
print("Intercept =", beta[0])
print(f"R^2 (manual): {r2_manual:.5f}")
print(f"MAE (manual): {mae_manual:.3f}")

# Scikit-learnによる回帰モデルと比較
x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:3], df.iloc[:, 3], test_size=0.2, random_state=1
)

lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)

# R^2 と MAE の計算（scikit-learn使用）
r2_sklearn = r2_score(y_test, pred_lr)
mae_sklearn = mean_absolute_error(y_test, pred_lr)

# 結果出力
print(f"R^2 (scikit-learn): {r2_sklearn:.5f}")
print(f"MAE (scikit-learn): {mae_sklearn:.3f}")
print("Coef = ", lr.coef_)
print("Intercept =", lr.intercept_)

# 勾配降下法による β の最適化
beta = np.zeros(X.shape[1])  # パラメータの初期化
beta = np.array([-1,0,0])
learning_rate = 0.01  # 学習率
iterations = 1000  # 繰り返し回数
m = len(df)  # サンプル数

# コスト関数（損失関数）
def compute_cost(X, Y, beta):
    total_cost = np.sum((X @ beta - Y) ** 2) / (2 * m)
    return total_cost

# 勾配降下法
for i in range(iterations):
    y_pred = X @ beta
    dbeta = (1 / m) * (X.T @ (y_pred - Y))  # 勾配の計算
    beta = beta - learning_rate * dbeta  # パラメータの更新
    
    # 100回ごとにコストを表示
    if i % 100 == 0: 
        cost = compute_cost(X, Y, beta)
        # print(f"Iteration {i}: Cost = {cost}, beta = {beta}")

# 最終結果
#print(f"Final beta: {beta}")

# 逆行列を用いた回帰係数 β の計算
Y_hat = X @ beta

# R^2 と MAE の計算
RSS = np.sum((Y - Y_hat) ** 2)
TSS = np.sum((Y - np.mean(Y)) ** 2)
r2_manual = 1 - (RSS / TSS)
mae_manual = np.mean(np.abs(Y - Y_hat))

# 結果出力
print("Coef = ", beta[1:3])
print("Intercept =", beta[0])
print(f"R^2 (gradient)): {r2_manual:.5f}")
print(f"MAE (gradient): {mae_manual:.3f}")
