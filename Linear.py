import numpy as np
class LinearRegression:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    

# def mse(coef, x, y):
#     return np.mean((np.dot(x, coef) - y)**2)/2
 
 
# def gradients(coef, x, y):
#     return np.mean(x.transpose()*(np.dot(x, coef) - y), axis=1)
 
 
# def multilinear_regression(coef, x, y, lr, b1=0.9, b2=0.999, epsilon=1e-8):
#     prev_error = 0
#     m_coef = np.zeros(coef.shape)
#     v_coef = np.zeros(coef.shape)
#     moment_m_coef = np.zeros(coef.shape)
#     moment_v_coef = np.zeros(coef.shape)
#     t = 0
 
#     while True:
#         error = mse(coef, x, y)
#         if abs(error - prev_error) <= epsilon:
#             break
#         prev_error = error
#         grad = gradients(coef, x, y)
#         t += 1
#         m_coef = b1 * m_coef + (1-b1)*grad
#         v_coef = b2 * v_coef + (1-b2)*grad**2
#         moment_m_coef = m_coef / (1-b1**t)
#         moment_v_coef = v_coef / (1-b2**t)
 
#         delta = ((lr / moment_v_coef**0.5 + 1e-8) *
#                  (b1 * moment_m_coef + (1-b1)*grad/(1-b1**t)))
 
#         coef = np.subtract(coef, delta)
#     return coef
 
 
# coef = np.array([0, 0, 0])
# c = multilinear_regression(coef, x, y, 1e-1)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
 
# ax.scatter(x[:, 1], x[:, 2], y, label='y',
#            s=5, color="dodgerblue")
 
# ax.scatter(x[:, 1], x[:, 2], c[0] + c[1]*x[:, 1] + c[2]*x[:, 2],
#            label='regression', s=5, color="orange")
 
# ax.view_init(45, 0)
# ax.legend()
# plt.show()