[//]: # (# Interactive Polynomial Regression)

[//]: # ()
[//]: # (This project is an interactive tool for polynomial regression that allows)

[//]: # (users to draw data points with the mouse and automatically fits a )

[//]: # (polynomial curve using least squares and ridge &#40;L2&#41; regularization with )

[//]: # (k-fold cross-validation.)

[//]: # ()
[//]: # (## Project Structure)

[//]: # ()
[//]: # (- `v1.py` - Main script containing the interactive regression system)

[//]: # (- `README.md` - Project overview and development notes)

[//]: # (- `docs/math.md` - Mathematical background and derivations)

[//]: # ()
[//]: # ()
[//]: # (## Features)

[//]: # (- Mouse-based data collection)

[//]: # (- Polynomial regression)

[//]: # (- Ridge &#40;L2&#41; regularization)

[//]: # (- Least squares solver &#40;```np.linalg.lstsq```&#41;)

[//]: # (- k-fold cross-validation)

[//]: # (- Real-time curve visualization)

[//]: # ()
[//]: # (## How to Run)

[//]: # (```bash)

[//]: # (pip install numpy matplotlib)

[//]: # (python v1.py)

[//]: # (```)

[//]: # ()
[//]: # (## Development Journey)

[//]: # ()
[//]: # (This project was built incrementally to explore the bias-variance tradeoff in )

[//]: # (polynomial regression.)

[//]: # ()
[//]: # (The motivation came from a curiosity sparked by using interactive )

[//]: # (graphing tools like Desmos. Specifically, whether a mathematical equation)

[//]: # (could be recovered from a hand-drawn curve.)

[//]: # ()
[//]: # ()
[//]: # (### Interactive System Design)

[//]: # ()
[//]: # (To support easy experimentation, the model is embedded into an interactive)

[//]: # (drawing interface built using Matplotlib event handlers. Mouse input is captured)

[//]: # (in real time to collect $&#40;x, y&#41;$ points, which are then used to fit and visualize)

[//]: # (the regression model immediately after each drawing.)

[//]: # ()
[//]: # ()
[//]: # (**Version 1**)

[//]: # (- Implemented basic polynomial regression using the least squares method)

[//]: # (- Initially used `np.linalg.inv&#40;&#41;`, which produced a valid hypothesis equation)

[//]: # (  but failed to consistently generate the curve)

[//]: # (- Suspected this was probably due to the nearly singular nature of the involved )

[//]: # (matrices)

[//]: # (- Tried to use the `np.linalg.solve&#40;&#41;`, but observed similar issues)

[//]: # ()
[//]: # (**Version 2**)

[//]: # (- Switched to `np.linalg.pinv&#40;&#41;`, which can deal with singular or )

[//]: # (near-singular matrices)

[//]: # (- Observed severe overfitting for higher-degree polynomials )

[//]: # (- Using `np.linalg.lstsq&#40;&#41;` helped a little but still faced issues with )

[//]: # (overfitting)

[//]: # ()
[//]: # (**Version 3**)

[//]: # (- Introduced ridge &#40;L2&#41; regularization)

[//]: # (- Reduced overfitting but struggled with selection of hyperparameters)

[//]: # ()
[//]: # (**Version 4**)

[//]: # (- Added k-fold cross-validation to select optimal polynomial degree and λ values)

[//]: # (- Achieved stable generalization across different drawings)

[//]: # ()
[//]: # ()
[//]: # (## Numerical Considerations)

[//]: # ()
[//]: # (Direct matrix inversion was avoided in the final implementation due to numerical)

[//]: # (instability when dealing with ill-conditioned matrices, especially when using )

[//]: # (higher-degree polynomial features. Instead, least squares solvers and )

[//]: # (regularization techniques were used to ensure stable solutions.)

[//]: # ()
[//]: # (For a detailed mathematical explanation of the methods used, see)

[//]: # ([docs/math.md]&#40;docs/math.md&#41;.)

[//]: # ()
[//]: # ()
[//]: # (## Limitations)

[//]: # ()
[//]: # (- The model is designed for smooth functions and may struggle with sharp discontinuities)

[//]: # (- The model cannot sketch closed or self-intersecting curves)

[//]: # (- The interactive system is intended for experimentation rather than)

[//]: # (  large-scale datasets)

[//]: # ()
[//]: # ()
[//]: # (Each stage introduced new challenges, which shaped the architectural decisions)

[//]: # (in the final implementation.)

[//]: # ()
[//]: # (## Future Improvements)

[//]: # ()
[//]: # (- Extend the model to a fully parametric approach, expressing both)

[//]: # (  $x$ and $y$ as functions of a common parameter $t$, enabling the)

[//]: # (  representation of closed curves)

[//]: # (- Extend the regression beyond polynomial bases by adding)

[//]: # (  exponential, logarithmic, or other nonlinear basis functions to better)

[//]: # (  capture functional behaviors)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (Feedback and suggestions are welcome via email.)

# Interactive Polynomial Regression

This project is an interactive tool for polynomial regression that allows
users to draw data points with the mouse and automatically fits a 
polynomial curve using least squares and ridge (L2) regularization with 
k-fold cross-validation.

## Project Structure

- `v1.py` - Main script containing the interactive regression system
- `README.md` - Project overview and development notes
- `docs/math.md` - Mathematical background and derivations


## Features
- Mouse-based data collection
- Polynomial regression
- Ridge (L2) regularization
- Least squares solver (`np.linalg.lstsq`)
- k-fold cross-validation
- Real-time curve visualization

## How to Run
```bash
pip install numpy matplotlib
python v1.py
```

## Development Journey

This project was built incrementally to explore the bias-variance tradeoff in 
polynomial regression.

The motivation came from a curiosity sparked by using interactive 
graphing tools like Desmos. Specifically, whether a mathematical equation
could be recovered from a hand-drawn curve.


### Interactive System Design

To support easy experimentation, the model is embedded into an interactive
drawing interface built using Matplotlib event handlers. Mouse input is captured
in real time to collect $(x, y)$ points, which are then used to fit and visualize
the regression model immediately after each drawing.


**Version 1**
- Implemented basic polynomial regression using the least squares method
- Initially used `np.linalg.inv()`, which produced a valid hypothesis equation
  but failed to consistently generate the curve
- Suspected this was probably due to the nearly singular nature of the involved 
matrices
- Tried to use the `np.linalg.solve()`, but observed similar issues

**Version 2**
- Switched to `np.linalg.pinv()`, which can deal with singular or 
near-singular matrices
- Observed severe overfitting for higher-degree polynomials 
- Using `np.linalg.lstsq()` helped a little but still faced issues with 
overfitting

**Version 3**
- Introduced ridge (L2) regularization
- Reduced overfitting but struggled with selection of hyperparameters

**Version 4**
- Added k-fold cross-validation to select optimal polynomial degree and λ values
- Achieved stable generalization across different drawings


## Numerical Considerations

Direct matrix inversion was avoided in the final implementation due to numerical
instability when dealing with ill-conditioned matrices, especially when using 
higher-degree polynomial features. Instead, least squares solvers and 
regularization techniques were used to ensure stable solutions.

For a detailed mathematical explanation of the methods used, see
[docs/math.md](docs/math.md).


## Limitations

- The model is designed for smooth functions and may struggle with sharp discontinuities
- The model cannot sketch closed or self-intersecting curves
- The interactive system is intended for experimentation rather than
  large-scale datasets


Each stage introduced new challenges, which shaped the architectural decisions
in the final implementation.

## Future Improvements

- Extend the model to a fully parametric approach, expressing both
  $x$ and $y$ as functions of a common parameter $t$, enabling the
  representation of closed curves
- Extend the regression beyond polynomial bases by adding
  exponential, logarithmic, or other nonlinear basis functions to better
  capture functional behaviors



Feedback and suggestions are welcome via email.