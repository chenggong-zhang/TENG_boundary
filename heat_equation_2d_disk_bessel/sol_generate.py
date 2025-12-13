import jax
import numpy as np
import scipy as sp
import os
import jax.numpy as jnp




alpha = 0.1
T_final = 2.0               # Final time
M = 2000                     # Number of time steps
dt = T_final / M            # Time step size
points_per_dim = 512

bessel_zeros = [sp.special.jn_zeros(i, 5) for i in range(5)]

rs = np.linspace(0, 20, 1000000)
jns = [sp.special.jn(n, rs) for n in range(5)]


# jax_jns = [[jnp.interp(rs, sp.special.jn(0, rs))]]

def diskharmonic(r, theta, m, n, t):
    # return sp.special.jn(m, bessel_zeros[m][n-1] * r) * np.cos(m * theta)
    return jnp.interp(bessel_zeros[m][n - 1] * r, rs, jns[m]) * jnp.cos(m * theta) * np.exp(-alpha * bessel_zeros[m][n - 1]**2 * t)
    # return sp.special.jn(m, bessel_zeros[m][n-1] * r) * jnp.cos(m * theta)


def u_target_func(xy, t):
    xx = xy[..., 0]
    yy = xy[..., 1]
    r = jnp.sqrt(xx ** 2 + yy ** 2)
    theta = jnp.arctan2(yy, xx)
    return (diskharmonic(r, theta, 0, 1, t) - \
            diskharmonic(r, theta, 0, 2, t) / 4 + \
            diskharmonic(r, theta, 0, 3, t) / 16 - \
            diskharmonic(r, theta, 0, 4, t) / 64 + \
            diskharmonic(r, theta, 1, 1, t) - \
            diskharmonic(r, theta, 1, 2, t) / 2 + \
            diskharmonic(r, theta, 1, 3, t) / 4 - \
            diskharmonic(r, theta, 1, 4, t) / 8 + \
            diskharmonic(r, theta, 2, 1, t) + \
            diskharmonic(r, theta, 3, 1, t) + \
            diskharmonic(r, theta, 4, 1, t)) / 4



grid = jnp.linspace(-1, 1, points_per_dim, endpoint=True)
grid2d = jnp.stack(jnp.meshgrid(grid, grid, indexing='ij'), axis=-1).reshape(-1, 2)
distances = jnp.linalg.norm(grid2d, axis=-1)
inside_disk_mask = distances <= 1.0
grid2d_inside_disk = grid2d[inside_disk_mask]
grid2d_inside_disk = grid2d_inside_disk.reshape(1, -1, 2)

#print(grid2d_inside_disk)


## This is to test whether the grid points alligned with self.xs by printing out the x-coordinates
def u_target_artificial(xy, t):
    xx = xy[..., 0]
    return xx


output_dir = "heat_solution_files"
os.makedirs(output_dir, exist_ok=True)



# Time evolution loop
for time_step in range(M):
    t = time_step * dt
    u_t = u_target_func(grid2d_inside_disk, t)  # Shape: (1, num_points_inside,)
    # u_t = u_target_artificial(grid2d_inside_disk, t)
    filename = os.path.join(output_dir, f"T_{t:.3f}".rstrip("0").rstrip(".") + ".npy")
    np.save(filename, u_t)
    print(f"Saved {filename}")
