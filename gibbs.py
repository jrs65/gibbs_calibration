import numpy as np
import h5py


def unique_to_full(y_unique, feedmap, feedmask=None):

    y_full = y_unique[feedmap]
    y_full[np.tril_indices(feedmap.shape[0])] = y_full[np.tril_indices(feedmap.shape[0])].conj()
    
    if feedmask is not None:
        y_full[np.where(np.logical_not(feedmask))] = 0.0

    return y_full

def full_to_unique(y_full, feedmap, feedmask=None):

    if feedmask is None:
        feedmask = np.ones(feedmap.shape, dtype=np.bool)

    y_full[np.tril_indices(feedmap.shape[0])] = y_full[np.tril_indices(feedmap.shape[0])].conj()
    y_unique = y_full[np.where(feedmask)][np.unique(feedmap[np.where(feedmask)], return_index=True)[1]]
    
    return y_unique


class GibbsCalibration(object):

    obs_y = None

    gains = None

    y_unique = None
    y_full = None

    noise = None

    telescope = None

    gain_samples = None
    y_samples = None
    chi2_samples = None

    gain_prior_mean = (1.0 + 0.0J)
    gain_prior_inv_var = 0.0

    y_prior_mean = (1.0 + 0.0J)
    y_prior_inv_var = 0.0

    def __init__(self, obs_y, noise, feedmap, feedmask, init_gain=None, init_y=None):

        # Set object parameters
        self.obs_y = obs_y
        self.feedmap = feedmap
        self.feedmask = feedmask
        self.noise = noise
        self.nfeed = obs_y.shape[0]
        self.nunique = np.unique(feedmap[np.where(feedmask)]).size

        # Set a few useful masks of the feed array
        self._trimask = np.zeros_like(obs_y, dtype=np.bool)
        self._trimask[np.triu_indices(self.nfeed)] = True
        self._fmaps = [np.where(np.logical_and(feedmap == i, self._trimask)) for i in range(self.nunique)]

        # Set initial gain solution (use ones if no guess provided)
        if init_gain is None:
            self.gains = np.ones(self.nfeed, dtype=np.complex128)
        else:
            self.gains = init_gain
        self.gains[0] = 1.0

        # Set initial visibilities guess (use conditional mean if nothing provided)
        if init_y is None:
            self.y_unique = self.y_mv()[0]
        else:
            self.y_unique = init_y

        self._y_unique_to_full()


    def _y_unique_to_full(self):
        self.y_full = unique_to_full(self.y_unique, self.feedmap, self.feedmask)
        


    def gain_mv(self, i):

        t1 = (self.gains * self.y_full[i].conj() * self.obs_y[i] / self.noise[i] * self.feedmask[i]).sum()
        t2 = (np.abs(self.gains * self.y_full[i].conj())**2 / self.noise[i] * self.feedmask[i]).sum()

        var = 1.0 / (t2 + self.gain_prior_inv_var)
        mean = (t1 + self.gain_prior_mean * self.gain_prior_inv_var) * var

        return mean, var
    
    def y_mv(self):

        t1ij = np.outer(self.gains.conj(), self.gains) * self.obs_y / self.noise
        t2ij = np.abs(np.outer(self.gains.conj(), self.gains))**2 / self.noise

        t1 = np.array([t1ij[self._fmaps[i]].sum() for i in range(self.nunique)])
        t2 = np.array([t2ij[self._fmaps[i]].sum() for i in range(self.nunique)])

        var = 1.0 / (t2 + self.y_prior_inv_var)
        mean = (t1 + self.y_prior_mean * self.y_prior_inv_var) * var

        return mean, var


    def _iterate_gains(self):

        # Restrict gain[0] to 1.0 to remove overall gain ambiguity
        self.gains[0] = 1.0

        for i in range(1, self.nfeed):
            mean, var = self.gain_mv(i)
            cvar = (np.random.standard_normal(2) * np.array([1.0, 1.0J])/ 2**0.5).sum()
            self.gains[i] = mean + cvar * var**0.5


    def _iterate_y(self):

        mean, var = self.y_mv()
        cvar = (np.random.standard_normal((self.nunique, 2)) * np.array([1.0, 1.0J])/ 2**0.5).sum(axis=1)
        self.y_unique = mean + cvar * var**0.5

        self._y_unique_to_full()


    def _save_sample(self):

        if self.gain_samples is None:
            self.gain_samples = self.gains.copy().reshape(1, self.nfeed)
            self.y_samples = self.y_unique.copy().reshape(1, self.nunique)
            self.chi2_samples = np.array([[self.chi2()]])
        else:
            self.gain_samples = np.vstack((self.gain_samples, self.gains.reshape(1, self.nfeed)))
            self.y_samples = np.vstack((self.y_samples, self.y_unique.reshape(1, self.nunique)))
            self.chi2_samples = np.vstack((self.chi2_samples, np.array([[self.chi2()]])))


    def step(self, save=True):

        self._iterate_gains()
        self._iterate_y()

        if save:
            self._save_sample()


    def run(self, niter, nburn=1000):

        for i in range(nburn):
            self.step(save=False)

        for i in range(niter):

            self.step()


    def chi2(self):

        return np.sum(self._trimask * np.abs(self.y_full * np.outer(self.gains, self.gains.conj()) - self.obs_y)**2 / self.noise)









si = 25


f = h5py.File('timeseries3.hdf5')

feedmap = f['feedmap'][:]
feedmask = f['feedmask'][:]

noise = f['noisepower'][:]

yf = f['visibility_timeseries'][:, :, si]
nf = f['noise_timeseries'][:, :, si]

nf = 0.5*(nf + nf.T.conj())

np.random.seed(1)
gains = 0.05 * (np.random.standard_normal([feedmap.shape[0], 2]) * np.array([1.0, 1.0J]) / 2**0.5).sum(axis=-1) + 1.0
gains[0] = 1.0
np.random.seed()

obs_y = np.outer(gains, gains.conj()) * yf + nf


nfd= 5

#oy = np.ones((nf, nf), dtype=np.complex128)
ns = np.ones((nfd, nfd), dtype=np.float64) * 1e-4
oy = 1.0 + (np.random.standard_normal((nfd, nfd, 2)) * np.array([1.0, 1.0J]) / 2**0.5).sum(axis=-1) * ns**0.5
oy = 0.5*(oy + oy.T.conj())
fm = np.identity(nfd, dtype=np.int) * -1
fa = (fm + 1).astype(np.bool)

gc = GibbsCalibration(obs_y, noise, feedmap, feedmask)#, init_gain=np.ones_like(gains), init_y=np.ones(13, dtype=np.complex128))
#gc = GibbsCalibration(oy, ns, fm, fa, init_gain=np.ones(nf, dtype=np.complex128), init_y=np.ones(1, dtype=np.complex128))

#gc.y_prior_inv_var = 1e20

gc.run(1000)