#
# Statistics tools
#
import numpy as np


class Summary:
    def __init__(self, data, axis=None, q=(2.5, 97.5)):
        """
        Get a bunch of statistics from a numpy array.

        :param data:
        :param axis:
        :param q:
        """

        self.data = np.asarray(data)

        # Which axis to go over
        self.axis = axis

        # Percentiles
        self.q = q

        # Stats including the NANs
        self.median = np.median(self.data, axis=self.axis)
        self.mean = np.mean(self.data, axis=self.axis)
        self.percentile = np.percentile(self.data, self.q, axis=self.axis)
        self.min = np.min(self.data, axis=self.axis)
        self.max = np.max(self.data, axis=self.axis)
        self.var = np.var(self.data, axis=self.axis)
        self.mad = np.median(np.abs(self.median - self.data), axis=self.axis)

        # Stats ignoring the NANs
        self.nanmedian = np.nanmedian(self.data, axis=self.axis)
        self.nanmean = np.nanmean(self.data, axis=self.axis)
        self.nanpercentile = np.nanpercentile(self.data, self.q, axis=self.axis)
        self.nanmin = np.nanmin(self.data, axis=self.axis)
        self.nanmax = np.nanmax(self.data, axis=self.axis)
        self.nanvar = np.nanvar(self.data, axis=self.axis)
        self.nanmad = np.nanmedian(np.abs(self.nanmedian - self.data), axis=self.axis)

        # How many NANs?
        self.n_nans = np.sum(np.isnan(self.data))
