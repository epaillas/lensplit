import numpy as np
from pypower import ArrayMesh
from pandas import qcut


class LenSplit:
    def __init__(self, data_mesh, boxsize=None, randoms_mesh=None):

        self.data_mesh = data_mesh
        self.boxsize = boxsize

        if boxsize is None:
            raise NotImplementedError(
                'Have not figure out how to implement randoms yet!')

    def get_mesh(self):
        mesh  = CatalogMesh(array=data_mesh, boxsize=self.boxsize,
            nmesh=self.nmesh)
        return mesh


    def get_density(self, smooth_radius, nmesh, compensate=True,
        sampling='randoms', sampling_positions=None,
    ):
        self.nmesh = nmesh

        data_mesh = self.get_mesh().to_mesh()
        data_mesh = data_mesh.r2c().apply(TopHat(r=smooth_radius))
        data_mesh = data_mesh.c2r()
        norm = sum(sum(data_mesh))
        density_mesh = data_mesh/(norm/(nmesh**3)) - 1

        if self.boxsize is None:
            shift = self.mesh.boxsize / 2 - self.mesh.boxcenter
        else:
            shift = 0

        if sampling_positions is not None:
            self.density = density_mesh.readout(sampling_positions + shift)
            self.sampling_positions = sampling_positions
        else:
            raise NotImplementedError(
                'Have not figured out what to do when sampling positions are not provided...')
            
        return self.density

    def get_quantiles(self, nquantiles, return_density=False):
        quantiles_idx = qcut(self.density, nquantiles, labels=False)
        quantiles = []
        for i in range(nquantiles):
            quantiles.append(self.sampling_positions[quantiles_idx == i])
        self.quantiles = quantiles
        if return_density:
            density_quantiles = []
            for i in range(nquantiles):
                density_quantiles.append(
                    np.mean(self.density[quantiles_idx == i])
                )
            density_quantiles = np.asarray(density_quantiles, dtype=float)
            return quantiles, density_quantiles
        return quantiles


class TopHat(object):
    # adapted from https://github.com/bccp/nbodykit/
    def __init__(self, r):
        self.r = r
    def __call__(self, k, v):
        r = self.r
        k = sum(ki ** 2 for ki in k) ** 0.5
        kr = k * r
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 3 * (np.sin(kr) / kr ** 3 - np.cos(kr) / kr ** 2)
        w[k == 0] = 1.0
        return w * v


