import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.colors
import pandas as pd


def celloracle_pca_plot(
    pca_results, 
    n_comps=None,
    n_comps_estimated=None,
    max_comps=None,
    ax=None
):
    ax = ax or plt.gca()
    
    plot_data = np.cumsum(pca_results['variance_ratio'])
    if max_comps is not None:
        plot_data = plot_data[:max_comps]
    
    ax.plot(plot_data)
    if n_comps is not None:
        ax.axvline(n_comps, c="k")
    
    if n_comps_estimated is not None:
        ax.axvline(n_comps_estimated, c="r")
    
    return ax


def celloracle_quiver_plot(
    vdata,
    celloracle_key,
    scale=None,
    ax=None,
    **kwargs,
):
    ax = ax or plt.gca()
    
    obsm_key_original = vdata.uns['celloracle'][celloracle_key]['obsm_key_original']
    obsm_key_delta = vdata.uns['celloracle'][celloracle_key]['obsm_key_delta']
    
    obsm_original = vdata.obsm[obsm_key_original]
    obsm_delta = vdata.obsm[obsm_key_delta]
    
    ax.quiver(
        obsm_original[:, 0], 
        obsm_original[:, 1], 
        obsm_delta[:, 0], 
        obsm_delta[:, 1], 
        scale=scale, 
        **kwargs
    )
    
    return ax


def _gridpoint_coordinates_and_total_p_mass(
    embedding,
    smooth: float=0.5, 
    steps: int=40,
    n_neighbors: int=100, 
    n_jobs: int=1, 
    xylim: Tuple=((None, None), (None, None))
) -> None:
        """Calculate the velocity using a points on a regular grid and a gaussian kernel

        Note: the function should work also for n-dimensional grid

        Arguments
        ---------
        embed: str, default=embedding
            The name of the attribute containing the embedding. It will be retrieved as getattr(self, embed)
            The difference vector is getattr(self, 'delta' + '_' + embed)
        smooth: float, smooth=0.5
            Higher value correspond to taking in consideration further points
            the standard deviation of the gaussian kernel is smooth * stepsize
        steps: tuple, default
            the number of steps in the grid for each axis
        n_neighbors:
            number of neighbors to use in the calculation, bigger number should not change too much the results..
            ...as soon as smooth is small
            Higher value correspond to slower execution time
        n_jobs:
            number of processes for parallel computing
        xymin:
            ((xmin, xmax), (ymin, ymax))

        """
        # Prepare the grid
        grs = []
        for dim_i in range(embedding.shape[1]):
            m, M = np.min(embedding[:, dim_i]), np.max(embedding[:, dim_i])

            if xylim[dim_i][0] is not None:
                m = xylim[dim_i][0]
            if xylim[dim_i][1] is not None:
                M = xylim[dim_i][1]

            m = m - 0.025 * np.abs(M - m)
            M = M + 0.025 * np.abs(M - m)
            gr = np.linspace(m, M, steps)
            grs.append(gr)

        meshes_tuple = np.meshgrid(*grs)
        gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
        nn.fit(embedding)
        dists, neighbors = nn.kneighbors(gridpoints_coordinates)

        std = np.mean([(g[1] - g[0]) for g in grs])
        # isotropic gaussian kernel
        gaussian_w = normal.pdf(loc=0, scale=smooth * std, x=dists)
        total_p_mass = gaussian_w.sum(1)
        
        return gridpoints_coordinates, total_p_mass, neighbors, gaussian_w


def _gridpoint_coordinates_and_total_p_mass_and_uz(
    embedding,
    delta_embedding,
    smooth: float=0.5, 
    steps: int=40,
    n_neighbors: int=100, 
    n_jobs: int=1, 
    xylim: Tuple=((None, None), (None, None))
) -> None:
    
    gridpoints_coordinates, total_p_mass, neighbors, gaussian_w = _gridpoint_coordinates_and_total_p_mass(
        embedding=embedding,
        smooth=smooth,
        steps=steps,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
        xylim=xylim,
    )
    UZ = (delta_embedding[neighbors] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, total_p_mass)[:, None]
        
    return gridpoints_coordinates, total_p_mass, UZ


def calculate_grid_properties(
    vdata,
    celloracle_key,
    smooth: float=0.5, 
    steps: int=40,
    n_neighbors: int=100, 
    min_mass: float=0.01,
    n_jobs: int=1, 
    xylim: Tuple=((None, None), (None, None))
) -> None:
        """Calculate the velocity using a points on a regular grid and a gaussian kernel

        Note: the function should work also for n-dimensional grid

        Arguments
        ---------
        embed: str, default=embedding
            The name of the attribute containing the embedding. It will be retrieved as getattr(self, embed)
            The difference vector is getattr(self, 'delta' + '_' + embed)
        smooth: float, smooth=0.5
            Higher value correspond to taking in consideration further points
            the standard deviation of the gaussian kernel is smooth * stepsize
        steps: tuple, default
            the number of steps in the grid for each axis
        n_neighbors:
            number of neighbors to use in the calculation, bigger number should not change too much the results..
            ...as soon as smooth is small
            Higher value correspond to slower execution time
        n_jobs:
            number of processes for parallel computing
        xymin:
            ((xmin, xmax), (ymin, ymax))

        Returns
        -------
        Nothing but it sets the attributes:
        flow_embedding: np.ndarray
            the coordinates of the embedding
        flow_grid: np.ndarray
            the gridpoints
        flow: np.ndarray
            vector field coordinates
        flow_magnitude: np.ndarray
            magnitude of each vector on the grid
        total_p_mass: np.ndarray
            density at each point of the grid

        """
        # Prepare the grid
        obsm_key = vdata.uns['celloracle'][celloracle_key]['obsm_key_original']
        obsm_key_delta = vdata.uns['celloracle'][celloracle_key]['obsm_key_delta']
        gridpoints_coordinates, total_p_mass, UZ = _gridpoint_coordinates_and_total_p_mass_and_uz(
            embedding=vdata.obsm[obsm_key],
            delta_embedding=vdata.obsm[obsm_key_delta],
            smooth=smooth,
            steps=steps,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
            xylim=xylim,
        )
        
        magnitude = np.linalg.norm(UZ, axis=1)
        
        flow = UZ
        flow_norm = UZ / np.percentile(magnitude, 99.5)
        
        return {
            'flow_grid': gridpoints_coordinates,
            'flow': flow,
            'flow_norm': flow_norm,
            'flow_norm_magnitude': np.linalg.norm(flow_norm, axis=1),
            'min_mass': min_mass,
            'mass_filter': total_p_mass < min_mass,
        }


def _grid_quiver_plot(
    grid_properties,
    grid_key='flow_grid',
    flow_key='flow',
    mass_filter_key='mass_filter',
    scale=None,
    ax=None,
    **kwargs,
):
    ax = ax or plt.gca()
    mass_filter = grid_properties[mass_filter_key]
    gridpoints_coordinates = grid_properties[grid_key]
    flow = grid_properties[flow_key]
    
    ax.quiver(
        gridpoints_coordinates[~mass_filter, 0],
        gridpoints_coordinates[~mass_filter, 1],
        flow[~mass_filter, 0],
        flow[~mass_filter, 1],
        #zorder=20000,
        scale=scale, 
        **kwargs
    )
    
    return ax

def celloracle_quiver_grid_plot(
    vdata,
    celloracle_key,
    grid_smooth=.8,
    grid_steps=40,
    n_neighbors=200,
    min_mass=0.01,
    **kwargs,
):
    grid_properties = calculate_grid_properties(
        vdata,
        celloracle_key,
        smooth=grid_smooth, 
        steps=grid_steps,
        n_neighbors=n_neighbors, 
        min_mass=min_mass,
        n_jobs=1, 
    )

    return _grid_quiver_plot(
        grid_properties,
        grid_key='flow_grid',
        flow_key='flow',
        **kwargs,
    )


def _polynomial_regression_sklearn(x, y, x_new, y_new, value, n_degree=3):
    data = np.stack([x, y], axis=1)
    data_new = np.stack([x_new, y_new], axis=1)

    pol = PolynomialFeatures(degree=n_degree, include_bias=False)
    data = pol.fit_transform(data)
    data_new = pol.transform(data_new)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Ridge(random_state=123)
        model.fit(data, value)

    return model.predict(data_new)


def _knn_regression(x, y, x_new, y_new, value, n_knn=30):

    data = np.stack([x, y], axis=1)

    model = KNeighborsRegressor(n_neighbors=n_knn)
    model.fit(data, value)

    data_new = np.stack([x_new, y_new], axis=1)
    return model.predict(data_new)


def calculate_pseudotime_grid_properties(
    vdata,
    celloracle_key,
    pseudotime_key,
    method='knn',
    n_poly=3,
    smooth: float=0.5, 
    steps: int=40,
    n_neighbors: int=100, 
    min_mass: float=0.01,
    n_jobs: int=1, 
    xylim: Tuple=((None, None), (None, None))
) -> None:
        """Calculate the velocity using a points on a regular grid and a gaussian kernel

        Note: the function should work also for n-dimensional grid

        Arguments
        ---------
        embed: str, default=embedding
            The name of the attribute containing the embedding. It will be retrieved as getattr(self, embed)
            The difference vector is getattr(self, 'delta' + '_' + embed)
        smooth: float, smooth=0.5
            Higher value correspond to taking in consideration further points
            the standard deviation of the gaussian kernel is smooth * stepsize
        steps: tuple, default
            the number of steps in the grid for each axis
        n_neighbors:
            number of neighbors to use in the calculation, bigger number should not change too much the results..
            ...as soon as smooth is small
            Higher value correspond to slower execution time
        n_jobs:
            number of processes for parallel computing
        xymin:
            ((xmin, xmax), (ymin, ymax))

        Returns
        -------
        Nothing but it sets the attributes:
        flow_embedding: np.ndarray
            the coordinates of the embedding
        flow_grid: np.ndarray
            the gridpoints
        flow: np.ndarray
            vector field coordinates
        flow_magnitude: np.ndarray
            magnitude of each vector on the grid
        total_p_mass: np.ndarray
            density at each point of the grid

        """
        # Prepare the grid
        if celloracle_key in vdata.uns['celloracle'].keys():
            obsm_key = vdata.uns['celloracle'][celloracle_key]['obsm_key_original']
        elif celloracle_key in vdata.obsm.keys():
            obsm_key = celloracle_key
        else:
            raise KeyError(f'{celloracle_key} not found in obsm or uns["celloracle"]')
        
        gridpoints_coordinates, total_p_mass, _, _ = _gridpoint_coordinates_and_total_p_mass(
            embedding=vdata.obsm[obsm_key],
            smooth=smooth,
            steps=steps,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
            xylim=xylim,
        )
        
        embedding = vdata.obsm[obsm_key]
        x, y = embedding[:, 0], embedding[:, 1]
        x_new, y_new = gridpoints_coordinates[:, 0], gridpoints_coordinates[:, 1]

        if method == "polynomial":
            value_on_grid =  _polynomial_regression_sklearn(
                x, 
                y, 
                x_new, 
                y_new, 
                vdata.obs[pseudotime_key].to_numpy(), 
                n_degree=n_poly,
        )
        elif method == "knn":
            value_on_grid = _knn_regression(
                x, 
                y, 
                x_new, 
                y_new, 
                vdata.obs[pseudotime_key].to_numpy(), 
                n_knn=n_neighbors,
        )
        
        # gradient
        n = int(np.sqrt(value_on_grid.shape[0]))
        value_on_grid_as_matrix = value_on_grid.reshape(n, n)
        dy, dx = np.gradient(value_on_grid_as_matrix)
        gradient = np.stack([dx.flatten(), dy.flatten()], axis=1)
        
        # normalise gradient
        size = np.sqrt(np.power(gradient, 2).sum(axis=1))
        size_sq = np.sqrt(size)
        size_sq[size_sq == 0] = 1
        factor = np.repeat(np.expand_dims(size_sq, axis=1), 2, axis=1)
        gradient /= factor
        
        # l2 norm gradient
        l2_norm = np.linalg.norm(gradient, ord=2, axis=1)
        scale_factor = 1 / l2_norm.mean()
        ref_flow = gradient * scale_factor
        
        return {
            'grid': gridpoints_coordinates,
            'pseudotime_grid': value_on_grid,
            'pseudotime_flow_norm': ref_flow,
            'min_mass': min_mass,
            'mass_filter': total_p_mass < min_mass,
        }


def pseudotime_quiver_grid_plot(
    vdata,
    celloracle_key,
    pseudotime_key,
    method='knn',
    n_poly=3,
    grid_smooth=.8,
    grid_steps=40,
    n_neighbors=200,
    min_mass=0.01,
    **kwargs,
):
    grid_properties = calculate_pseudotime_grid_properties(
        vdata,
        celloracle_key,
        pseudotime_key,
        method=method,
        n_poly=n_poly,
        smooth=grid_smooth, 
        steps=grid_steps,
        n_neighbors=n_neighbors, 
        min_mass=min_mass,
        n_jobs=1, 
    )

    return _grid_quiver_plot(
        grid_properties,
        grid_key='grid',
        flow_key='pseudotime_flow_norm',
        **kwargs,
    )


def celloracle_development_perturbations(
    vdata,
    celloracle_key,
    pseudotime_key,
    method='knn',
    n_poly=3,
    grid_smooth=.8,
    grid_steps=40,
    n_neighbors=200,
    min_mass=0.01,
    n_bins=10,
    **kwargs,
):
    grid_properties = calculate_grid_properties(
        vdata,
        celloracle_key,
        smooth=grid_smooth, 
        steps=grid_steps,
        n_neighbors=n_neighbors, 
        min_mass=min_mass,
        n_jobs=1, 
    )
    
    pseudotime_grid_properties = calculate_pseudotime_grid_properties(
        vdata,
        celloracle_key,
        pseudotime_key,
        method=method,
        n_poly=n_poly,
        smooth=grid_smooth, 
        steps=grid_steps,
        n_neighbors=n_neighbors, 
        min_mass=min_mass,
        n_jobs=1, 
    )
    
    inner_product = np.array(
        [
            np.dot(i, j) for i, j in zip(
                grid_properties['flow'], 
                pseudotime_grid_properties['pseudotime_flow_norm']
            )
        ]
    )
    
    mass_filter = grid_properties['mass_filter']
    min_ = pseudotime_grid_properties['pseudotime_grid'][~mass_filter].min()
    max_ = pseudotime_grid_properties['pseudotime_grid'][~mass_filter].max()
    width = (max_ - min_)/(n_bins)
    bins = np.arange(min_, max_ + width, width)[1:-1]
    
    data = {
            "score": inner_product,
            "pseudotime": pseudotime_grid_properties['pseudotime_grid'],
            "flow1": grid_properties['flow'][:, 0],
            "flow2": grid_properties['flow'][:, 1],
            "mass_filter": mass_filter,
            "pseudotime_binned": np.digitize(pseudotime_grid_properties['pseudotime_grid'], bins),
            "x": grid_properties['flow_grid'][:, 0],
            "y": grid_properties['flow_grid'][:, 1],
        }
    print(data)
    inner_product_df = pd.DataFrame(
        data
    )
    
    return inner_product_df


def celloracle_developmental_perturbation_plot_from_df(
    inner_product_df,
    ax=None,
    **kwargs
):
    ax = ax or plt.gca()
    
    mass_filter = inner_product_df['mass_filter']
    inner_product = inner_product_df['score']
    x = inner_product_df['x']
    y = inner_product_df['y']
    
    kwargs.setdefault('vmin', -1)
    kwargs.setdefault('vmax', 1)
    kwargs.setdefault('vcenter', 0)
    kwargs.setdefault(
        'norm', 
        matplotlib.colors.TwoSlopeNorm(
            vmin=kwargs.pop('vmin'), 
            vcenter=kwargs.pop('vcenter'), 
            vmax=kwargs.pop('vmax')
        )
    )
    kwargs.setdefault('s', 50)
    kwargs.setdefault('cmap', 'PiYG')
    
    ax.scatter(
        x[~mass_filter],
        y[~mass_filter],
        c=inner_product[~mass_filter],
        **kwargs,
    )
    return ax
