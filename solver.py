import tensorflow as tf
try:
    from .models import PINN_NeuralNet  # When run as a package
except ImportError:
    from models import PINN_NeuralNet  # When run directly
from time import time
import numpy as np

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
tf.random.set_seed(42)

def gamma_tf(x):
    return tf.exp(tf.math.lgamma(x))

class PINNSolver():
    def __init__(self, model, X_r, t_dummy, Type, alpha, h):
        self.model = model
        self.X_r = X_r
        self.t_dummy = t_dummy
        self.Type = Type
        self.alpha = alpha
        self.h = h
        self.hist = []
        self.iter = 0
        self.last_n_losses = []

    def update_last_n_losses(self, loss):
        self.last_n_losses.append(loss)
        if len(self.last_n_losses) > 20:
            self.last_n_losses.pop(0)

    def ES(self):
        if len(self.last_n_losses) < 20:
            return 100  # a large number
        current_loss = self.last_n_losses[-1]
        max_relative_error = 100.*max([abs(current_loss - loss) / current_loss for loss in self.last_n_losses[:-1]])
        return max_relative_error

    def get_r(self):
        with tf.GradientTape() as tape:
            tape.watch(self.X_r)
            u = self.model(self.X_r)            
        u_x = tape.gradient(u, self.X_r)
        del tape
        
        if self.Type == 'Volterra':  # Calculate Volterra-type integrals
            uf = []
            for index in range(len(self.X_r)):
                XD = self.X_r[:index+1]
                X = self.X_r[index:index+1]
                u = self.model(XD)
                integrand = u
                uf_pw = self.RL(self.alpha, integrand, index)
                uf.append(uf_pw)
            uf = tf.reshape(tf.concat(uf, axis=0), (-1,1))
            res = u - (tf.sqrt(np.pi)*(1. + self.X_r)**-1.5 - 0.02*self.X_r**3/(1. + self.X_r) + 0.01*self.X_r**2.5*uf)
            loss_r = tf.reduce_mean(tf.square(res))
            loss = loss_r # No BC loss is needed for this problem
        else:  # Calculate Fredholm-type integrals
            u_dummy = self.model(self.t_dummy)
            uf = [self.RL(self.alpha, self.t_dummy*u_dummy**2, index) for index in range(len(self.X_r))][0]
            res = u_x - (tf.cos(self.X_r) - self.X_r + 0.25*self.X_r*uf)
            loss_r = tf.reduce_mean(tf.square(res))
            loss_bc = tf.reduce_mean(tf.square(self.model(-np.pi/2.*tf.ones(tf.shape(self.X_r))) - (0.)))
            loss = loss_r + loss_bc
        return loss

    def RL(self, alpha, f, data_point):
        # Implements the Riemann-Liouville fractional calculus logic
        summation = 0.
        for k in range(data_point+1):
            if k==0 and data_point != 0:
                sigma = (1 + alpha)*data_point**alpha - data_point**(1. + alpha) + (data_point - 1.)**(1. + alpha)
            elif k==data_point:
                sigma = 1.
            elif 0 < k < data_point:
                sigma = (data_point - k + 1.)**(1. + alpha) - \
                        2.*(data_point - k)**(1. + alpha) + (data_point - k - 1.)**(1. + alpha)
            summation += sigma * f[k]
        fractional = self.h**alpha * 1./gamma_tf(2. + alpha) * summation
        return fractional

    def loss_fn(self):
        loss_eq = self.get_r()
        loss = loss_eq
        return loss

    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn()            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape       
        return loss, g

    def solve_with_TFoptimizer(self, optimizer, N=1001):
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad()
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        for i in range(N):          
            loss = train_step()
            self.current_loss = loss.numpy()
            self.max_relative_error = self.ES()
            self.callback(self.max_relative_error)
            self.update_last_n_losses(self.current_loss)

            if self.max_relative_error < 1e-3: # in %
                tf.print('Early stopping... \nIt {:05,d}: Loss = {:10.4e}, Max. rel. error = {} %'.format(self.iter,
                                                             self.current_loss,
                                                            np.round(self.max_relative_error, 3)))
                break

    def callback(self, xr=None):
        if self.iter % 2000 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter, self.current_loss))
        self.hist.append(self.current_loss)
        self.iter += 1

    def plot_loss_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.hist)), self.hist, 'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax
