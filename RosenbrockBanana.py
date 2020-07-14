import pypesto
import pypesto.visualize as visualize
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pypesto.optimize as optimize


###### Define Objective and Function
# first type of objective
objective1 = pypesto.Objective(fun=sp.optimize.rosen,
                               grad=sp.optimize.rosen_der,
                               hess=sp.optimize.rosen_hess)

# second type of objective
def rosen2(x):
    return (sp.optimize.rosen(x),
            sp.optimize.rosen_der(x),
            sp.optimize.rosen_hess(x))
objective2 = pypesto.Objective(fun=rosen2, grad=True, hess=True)

dim_full = 10
lb = -5 * np.ones((dim_full, 1))
ub = 5 * np.ones((dim_full, 1))

problem1 = pypesto.Problem(objective=objective1, lb=lb, ub=ub)
problem2 = pypesto.Problem(objective=objective2, lb=lb, ub=ub)


###### Illustration
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)
for j in range(0, x.shape[0]):
    for k in range(0, x.shape[1]):
        z[j,k] = objective1([x[j,k], y[j,k]], (0,))

fig = plt.figure()
fig.set_size_inches(*(14,10))
ax = plt.axes(projection='3d')
ax.plot_surface(X=x, Y=y, Z=z)
plt.xlabel('x axis')
plt.ylabel('y axis')
ax.set_title('cost function values')


###### Opimization
# create different optimizers
optimizer_bfgs = optimize.ScipyOptimizer(method='l-bfgs-b')
optimizer_tnc = optimize.ScipyOptimizer(method='TNC')
optimizer_dogleg = optimize.ScipyOptimizer(method='dogleg')

# set number of starts
n_starts = 20

# save optimizer trace
history_options = pypesto.HistoryOptions(trace_record=True)

# Run optimizaitons for different optimzers
result1_bfgs = optimize.minimize(
    problem=problem1, optimizer=optimizer_bfgs,
    n_starts=n_starts, history_options=history_options)
result1_tnc = optimize.minimize(
    problem=problem1, optimizer=optimizer_tnc,
    n_starts=n_starts, history_options=history_options)
result1_dogleg = optimize.minimize(
    problem=problem1, optimizer=optimizer_dogleg,
    n_starts=n_starts, history_options=history_options)

# Optimize second type of objective
result2 = optimize.minimize(
    problem=problem2, optimizer=optimizer_tnc, n_starts=n_starts)


###### Visualite and compare optimization results
# plot separated waterfalls
visualize.waterfall(result1_bfgs, size=(15,6))
visualize.waterfall(result1_tnc, size=(15,6))
visualize.waterfall(result1_dogleg, size=(15,6))

# plot one list of waterfalls
visualize.waterfall([result1_bfgs, result1_tnc],
                    legends=['L-BFGS-B', 'TNC'],
                    start_indices=10,
                    scale_y='lin')

# retrieve second optimum
all_x = result1_bfgs.optimize_result.get_for_key('x')
all_fval = result1_bfgs.optimize_result.get_for_key('fval')
x = all_x[19]
fval = all_fval[19]
print('Second optimum at: ' + str(fval))

# create a reference point from it
ref = {'x': x, 'fval': fval, 'color': [
    0.2, 0.4, 1., 1.], 'legend': 'second optimum'}
ref = visualize.create_references(ref)

# new waterfall plot with reference point for second optimum
visualize.waterfall(result1_dogleg, size=(15,6),
                    scale_y='lin', y_limits=[-1, 101],
                    reference=ref, colors=[0., 0., 0., 1.])


###### Visulaize Parameters
visualize.parameters([result1_bfgs, result1_tnc],
                     legends=['L-BFGS-B', 'TNC'],
                     balance_alpha=False)
visualize.parameters(result1_dogleg,
                     legends='dogleg',
                     reference=ref,
                     size=(15,10),
                     start_indices=[0, 1, 2, 3, 4, 5],
                     balance_alpha=False)

df = result1_tnc.optimize_result.as_dataframe(
    ['fval', 'n_fval', 'n_grad', 'n_hess', 'n_res', 'n_sres', 'time'])
df.head()


###### Optimizer History
# plot one list of waterfalls
visualize.optimizer_history([result1_bfgs, result1_tnc],
                            legends=['L-BFGS-B', 'TNC'],
                            reference=ref)
# plot one list of waterfalls
visualize.optimizer_history(result1_dogleg,
                            reference=ref)

# plot one list of waterfalls
visualize.optimizer_history([result1_bfgs, result1_tnc],
                            legends=['L-BFGS-B', 'TNC'],
                            reference=ref,
                            offset_y=0.)

# plot one list of waterfalls
visualize.optimizer_history([result1_bfgs, result1_tnc],
                            legends=['L-BFGS-B', 'TNC'],
                            reference=ref,
                            scale_y='lin',
                            y_limits=[-1., 11.])

### profiling is missing
