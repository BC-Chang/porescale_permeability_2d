import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter_ns


def initialize_weights():

  # Define lattice velocity vectors
  ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.double)
  ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.double)


  # Define weights
  w_i = np.array([4./9.,
                  1./9.,
                  1./9.,
                  1./9.,
                  1./9.,
                  1./36.,
                  1./36.,
                  1./36.,
                  1./36.], dtype=np.double)

  return ex, ey, w_i

def macroscopic(f, fx, fy, ex, ey, nx, ny):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Initialize outputs to 0s
  u_x = torch.zeros((nx, ny), dtype=torch.float64).to(device)
  u_y = torch.zeros((nx, ny), dtype=torch.float64).to(device)
  rho = torch.zeros((nx, ny), dtype=torch.float64).to(device)

  # Calculate macroscopic properties from moments. Only calculate where there are fluid nodes
  # Density
  rho[fx, fy] += torch.sum(f[:, fx, fy], axis=0)

  # Velocity
  u_x[fx, fy] += torch.sum(ex[:, None] * f[:, fx, fy], axis=0)
  u_y[fx, fy] += torch.sum(ey[:, None] * f[:, fx, fy], axis=0)
  u_x[fx, fy] = u_x[fx, fy] / rho[fx, fy]
  u_y[fx, fy] = u_y[fx, fy] / rho[fx, fy]

  return rho, u_x, u_y

def equilibrium(rho, ux, uy, ex, ey, w_i, nx, ny, fx, fy, tau, g):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Extract only fluid nodes and add forcing to velocity
  ux = ux[fx, fy] + tau*g
  uy = uy[fx, fy]

  feq = torch.zeros((9, nx, ny), dtype=torch.float64).to(device)

  for i in range(9):
    # Compute 2nd term in parenthesis of feq equation above
    uc = ex[i]*ux + ey[i]*uy
    feq[i, fx, fy] = rho[fx, fy] * w_i[i] * (1 + 3*uc + (9./2.)*uc**2 - (3./2.)*(ux**2 + uy**2))

  return feq

def collision(f, feq, tau, sx, sy, fx, fy):
    # Standard Bounceback for Solid Nodes
    # Left-Right
    f[1, sx, sy], f[3, sx, sy] = f[3, sx, sy], f[1, sx, sy]

    # Up-Down
    f[2, sx, sy], f[4, sx, sy] = f[4, sx, sy], f[2, sx, sy]

    # Top Right - Bottom Left
    f[5, sx, sy], f[7, sx, sy] = f[7, sx, sy], f[5, sx, sy]

    # Top Left - Bottom Right
    f[6, sx, sy], f[8, sx, sy]  = f[8, sx, sy], f[6, sx, sy]

    # Regular collision in fluid nodes
    f[:, fx, fy] -= (f[:, fx, fy] - feq[:, fx, fy]) / tau

    return f

def run_lbm(data, F=0.00001):

    # Initialization
    tau = 1.0  # Relaxation time
    g = F  # Gravity or other force
    density = 1.
    tf = 10001  # Maximum number of iteration steps
    precision = 1.E-5  # Convergence criterion
    vold = 1000
    eps = 1E-6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.tensor(data).to(device)
    check_convergence = 30  # Check convergence every [check_convergence] time steps

    # Define lattice velocity vectors
    ex, ey, w_i = initialize_weights()
    ex = torch.tensor(ex).to(device)
    ey = torch.tensor(ey).to(device)
    w_i = torch.tensor(w_i).to(device)

    # Indices of fluid nodes
    fluid_id = torch.argwhere(data == 0).to(device)
    fx = fluid_id[:, 0]
    fy = fluid_id[:, 1]

    # Indices of solid nodes
    solid_id = torch.argwhere(data == 1).to(device)
    sx = solid_id[:, 0]
    sy = solid_id[:, 1]

    # Solid nodes are labeled 1, fluid nodes are labeled 0
    is_solid_node = data

    nx, ny = data.shape

    # Initialize distribution functions
    f = w_i * density
    # Broadcast to 3D array with each slice corresponding to a direction's weights
    f = torch.tile(f[:, None, None], (nx, ny)).type(torch.float64).to(device)

    # Allocate memory to equilibrium functions
    feq = torch.empty_like(f, dtype=torch.float64).to(device)

    # Each point has x-component ex, and y-component ey
    u_x = torch.empty((nx, ny), dtype=torch.float64).to(device)
    u_y = torch.empty((nx, ny), dtype=torch.float64).to(device)

    # Node density
    rho = torch.zeros((nx, ny), dtype=torch.float64)

    # # Begin time loop
    tic = perf_counter_ns()
    for ts in tqdm(range(tf)):
        # print(f"{ts = }")  # Print timestep

        # Compute macroscopic density, rho and velocity.
        rho, u_x, u_y = macroscopic(f, fx, fy, ex, ey, nx, ny)

        # Add forcing to velocity and compute equilibrium function
        feq = equilibrium(rho, u_x, u_y, ex, ey, w_i, nx, ny, fx, fy, tau, g)

        # Collision Step
        f = collision(f, feq, tau, sx, sy, fx, fy)

        # Streaming Step
        f[1] = torch.roll(f[1], 1, dims=1)
        f[2] = torch.roll(f[2], 1, dims=0)
        f[3] = torch.roll(f[3], -1, dims=1)
        f[4] = torch.roll(f[4], -1, dims=0)

        f[5] = torch.roll(f[5], (1, 1), dims=(0,1))
        f[6] = torch.roll(f[6], (-1, 1), dims=(1,0))
        f[7] = torch.roll(f[7], (-1, -1), dims=(0,1))
        f[8] = torch.roll(f[8], (1, -1), dims=(1,0))

        # Calculate velocity
        u = torch.sqrt(u_x**2 + u_y**2)

        # Check convergence every check_convergence time step
        if ts % check_convergence == 0:

            vnew = torch.mean(u)
            error = torch.abs(vold - vnew) / (vold+eps)
            vold = vnew

            if error < precision:
                print(f'Simulation has converged in {ts} time steps')
                break

        if ts == tf:
            print('Reached maximum iterations')

    toc = perf_counter_ns()
    print(f"Elapsed Time: {(toc - tic)*1E-9}s")
    u_x[is_solid_node] = torch.nan
    u_y[is_solid_node] = torch.nan
    u[is_solid_node] = torch.nan
    u_x = u_x.to('cpu')
    u_y = u_y.to('cpu')
    u = u.to('cpu')
    

    return u_x, u_y, u
