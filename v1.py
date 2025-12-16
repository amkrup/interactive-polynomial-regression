import matplotlib.pyplot as plt
import numpy as np

xs = np.array([])
ys = np.array([])
drawing = False

def reset_canvas(event):
    global drawing, xs, ys
    drawing = False
    xs = np.array([])
    ys = np.array([])
    ax.cla()
    ax.set_title("draw with mouse. right-click to clear the canvas.")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.grid(True)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    plt.draw()
    print("canvas reset.")


def on_click(event):
    global drawing, xs, ys
    if event.xdata is None or event.ydata is None:
        return
    if event.button == 3:
        reset_canvas(event)
        return

    drawing = True
    xs = np.append(xs, event.xdata)
    ys = np.append(ys, event.ydata)


def on_motion(event):
    global drawing, xs, ys
    if not drawing or event.xdata is None or event.ydata is None: return

    xs = np.append(xs, event.xdata)
    ys = np.append(ys, event.ydata)
    plt.plot(xs[-2:], ys[-2:], linewidth=2)
    plt.draw()


def ridge_lstsq(X, y, k):
    n_features = X.shape[1]

    X_aug = np.vstack([X, np.sqrt(k) * np.identity(n_features)])
    y_aug = np.vstack([y.reshape(-1, 1), np.zeros((n_features, 1))])

    theta = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]
    return theta.flatten()


def cross_validate_n_k(xs, ys, n_values, k_values, n_folds=5):
    m = len(ys)
    fold_size = m // n_folds

    best_score = float('inf')
    best_n = n_values[0]
    best_k = k_values[0]

    for n_val in n_values:
        feature_matrix = build_feature_matrix(xs, n_val)

        for k_val in k_values:
            fold_errors = []

            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_folds - 1 else m

                val_indices = list(range(val_start, val_end))
                train_indices = [i for i in range(m) if i not in val_indices]

                X_train = feature_matrix[train_indices]
                y_train = ys[train_indices]
                X_val = feature_matrix[val_indices]
                y_val = ys[val_indices]

                theta = ridge_lstsq(X_train, y_train, k_val)

                val_error = compute_mse(X_val, y_val, theta)
                fold_errors.append(val_error)

            avg_error = np.mean(fold_errors)

            if avg_error < best_score:
                best_score = avg_error
                best_n = n_val
                best_k = k_val

    return best_n, best_k

def compute_mse(X, y, theta):
    prediction = X @ theta
    return np.mean((prediction - y) ** 2)


def build_feature_matrix(xs, n):
    m = len(xs)

    feature_matrix = np.ones((m, 1))
    for i in range(1, n + 1):
        feature_matrix = np.column_stack([feature_matrix, xs ** i])

    return feature_matrix


def on_release(event):
    global drawing, xs, ys
    drawing = False

    if event.button == 3:
        return

    if event.xdata is not None and event.ydata is not None:
        xs = np.append(xs, event.xdata)
        ys = np.append(ys, event.ydata)

    print("stroke done")
    print("collected ", len(xs), " points")

    m = len(xs)

    max_degree = min(9, m - 2)
    n_values = list(range(2, max_degree + 1))
    k_values = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    n, k = cross_validate_n_k(xs, ys, n_values, k_values, n_folds=5)


    feature_matrix = np.ones((m, 1))
    for i in range(1, n + 1):
        feature_matrix = np.column_stack([feature_matrix, xs ** i])


    theta_best = ridge_lstsq(feature_matrix, ys, k)

    X_test = np.linspace(xs.min(), xs.max(), 200).reshape(-1, 1)

    X_test_poly = np.ones((X_test.shape[0], 1))
    for i in range(1, n + 1):
        X_test_poly = np.column_stack([X_test_poly, X_test ** i])

    y_test_pred = X_test_poly @ theta_best

    plt.plot(X_test, y_test_pred, c='r', label="Model", linewidth=2)
    plt.legend()
    plt.draw()

    for i in range(len(theta_best)):
        if abs(theta_best[i]) < 1e-3:
            theta_best[i] = 0.0

    print("\nHypothesis function:")
    print(f"h(x) = {theta_best[0]:.4f}", end="")
    for i in range(1, len(theta_best)):
        if theta_best[i] == 0.0: continue
        sign = "+" if theta_best[i] >= 0 else "-"
        coeff = abs(theta_best[i])
        if i == 1:
            print(f" {sign} {coeff:.4f}x", end="")
        else:
            print(f" {sign} {coeff:.4f}x^{i}", end="")
    print()


fig, ax = plt.subplots()
ax.set_title("draw with mouse. right-click to clear the canvas.")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

cid_press = fig.canvas.mpl_connect("button_press_event", on_click)
cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

plt.grid(True)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

plt.show()