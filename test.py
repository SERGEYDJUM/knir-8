from cho import CHO
import numpy as np


if __name__ == "__main__":
    X = np.random.uniform(0, 1, (200, 64, 64))
    y = np.concat((np.ones(100, dtype=np.bool), np.zeros(100, dtype=np.bool)))

    model = CHO(_debug_mode=True)
    model.train(X, y)
    model.test(X[10:30])

    X = np.random.uniform(0, 1, (200, 64, 64))
    y = np.concat((np.ones(100, dtype=np.bool), np.zeros(100, dtype=np.bool)))

    print(model.measure(X, y))
