from scipy.fftpack import fft, fft2, ifft, ifft2, dct, idct, dst, idst, fftshift, fftfreq
from numpy import linspace,sqrt, zeros, array, pi, sin, cos, exp, arange, matmul, abs, conj, tile, real, convolve, ones, meshgrid
import matplotlib.pyplot as plt
import difflib
import scipy.sparse as sp
import numpy as np

plt.rcParams['figure.figsize'] = 15, 6

Nx = 2**7
Lx= 6 * pi
dx = Lx/Nx

Ny = 2**7
Ly= 6 * pi
dy = Ly/Ny

x = linspace(-0.5*Lx, 0.5*Lx, Nx)
y = linspace(-0.5*Ly, 0.5*Ly, Ny)

X, Y = meshgrid(x,y)

kx = fftfreq(Nx, dx)
ky = fftfreq(Ny, dy)

KX, KY = meshgrid(kx, ky)

def laplacian_using_fft(some_2d_signal):
    Fhat = fft2(some_2d_signal)
    F_xx = ifft2( (1j * KX) * (1j * KX) *  Fhat)
    F_yy = ifft2( (1j * KY) * (1j * KY) *  Fhat)

    return real(F_xx + F_yy)






# TEST CASE 2: RADIAL SINUSOID
#
def func(x,y):
    return sin(sqrt(x**2+y**2))

def exact_partial_derivative_wrt_x(x,y):
    return x*cos(sqrt(x**2 + y**2))/(sqrt(x**2+y**2))

def exact_partial_derivative_wrt_y(x,y):
    return y*cos(sqrt(x**2 + y**2))/(sqrt(x**2+y**2))

def exact_laplacian(x,y):
    return -sin(sqrt(x**2+y**2))

F = func(X,Y)

F_x_exact = exact_partial_derivative_wrt_x(X,Y)
F_y_exact = exact_partial_derivative_wrt_y(X,Y)

# First partial derivatives
Fhat = fft2(F)
F_x = real(ifft2( (1j * KX) *  Fhat))
F_y = real(ifft2( (1j * KY) *  Fhat))

# Laplacians
laplacian_exact = exact_laplacian(X,Y)
laplacian_calculated = laplacian_using_fft(F)


# Rest is plotting
xleft = -Lx/2
xright = Lx/2
ytop = Ly/2
ybottom = -Ly/2
color_levels = 128

fig = plt.figure()
ax0 = plt.subplot2grid((2,5), (0,0) , colspan=2, rowspan=2) # leftmost, function itself

ax1 = plt.subplot2grid((2,5), (0,2)) # top left, ∂/∂x exact
ax2 = plt.subplot2grid((2,5), (1,2)) # bottom left, ∂/∂x using FFT

ax3 = plt.subplot2grid((2,5), (0,3)) # top mid, ∂/∂y exact
ax4 = plt.subplot2grid((2,5), (1,3)) # bottom mid, ∂/∂y using FFT

ax5 = plt.subplot2grid((2,5), (0,4)) # top right, laplacian exact
ax6 = plt.subplot2grid((2,5), (1,4)) # bottom right, laplacian using FFT

# Funcion itself
ax0.contourf(X, Y, F, color_levels)
ax0.set_title(r'$f(x,y) = \sin \left( \sqrt{x^2 + y^2} \right)$')
ax0.set_xlim(xleft,xright)
ax0.set_ylim(ybottom,ytop)

# top left, ∂/∂x exact
ax1.contourf(X, Y, F_x_exact, color_levels)
ax1.set_title(r'$\partial f/ \partial x$ (Exact)')
ax1.set_xlim(xleft,xright)
ax1.set_ylim(ybottom,ytop)

# bottom left, ∂/∂x using FFT
ax2.contourf(X, Y, F_x, color_levels)
ax2.set_title(r'$\partial f/ \partial x$ (Using FFT)')
ax2.set_xlim(xleft,xright)
ax2.set_ylim(ybottom,ytop)

# top mid, ∂/∂y exact
ax3.contourf(X, Y, F_y_exact, color_levels)
ax3.set_title(r'$\partial f/ \partial y$ (Exact)')
ax3.set_xlim(xleft,xright)
ax3.set_ylim(ybottom,ytop)

# bottom mid, ∂/∂y fft
ax4.contourf(X, Y, F_y, color_levels)
ax4.set_title(r'$\partial f/ \partial y$ (Using FFT)')
ax4.set_xlim(xleft,xright)
ax4.set_ylim(ybottom,ytop)

# top mid, ∂/∂y exact
ax5.contourf(X, Y, laplacian_exact, color_levels)
ax5.set_title(r'$\nabla^2 y$ (Exact)')
ax5.set_xlim(xleft,xright)
ax5.set_ylim(ybottom,ytop)

# bottom mid, ∂/∂y fft
ax6.contourf(X, Y, laplacian_calculated, color_levels)
ax6.set_title(r'$\nabla^2$ (Using FFT)')
ax6.set_xlim(xleft,xright)
ax6.set_ylim(ybottom,ytop)

plt.tight_layout()

plt.show()

