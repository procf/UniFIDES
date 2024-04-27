try:
    from .solver import PINNSolver  # When run as a package
    from .models import PINN_NeuralNet  # When run as a package
except ImportError:
    from solver import PINNSolver  # When run directly
    from models import PINN_NeuralNet  # When run directly

import tensorflow as tf
import numpy as np

# Constants
pi = np.pi
Type = 'Fredholm'
N_r = 32

# Setup computation
xmin, xmax = -pi/2., pi/2.
tmin, tmax = -pi/2, pi/2
alpha = 1.
h = (xmax - xmin)/(N_r - 1)

# Generate data
t_dummy = tf.linspace(tmin, tmax, N_r)
t_dummy = tf.reshape(t_dummy, (-1,1))
t_dummy = tf.convert_to_tensor(t_dummy, dtype=tf.float32)

x_r_f = np.linspace(xmin, xmax, N_r)
X_r = np.reshape(x_r_f, (-1,1))
X_r = tf.convert_to_tensor(X_r, dtype=tf.float32)

# Initialize model and solver
model = PINN_NeuralNet()
solver = PINNSolver(model, X_r, t_dummy, Type, alpha, h)

# Configure and start training
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000], [1e-2, 5e-3, 1e-3])
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve_with_TFoptimizer(optim, N=int(3e4+1))
