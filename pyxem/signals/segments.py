# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

"""Signal classes for nanocrystal segmentation results obtained from
i) machine learning (LearningSegment) and ii) VDF images (VDFSegment).

"""

import numpy as np

from hyperspy.signals import Signal2D

from pyxem.utils.segment_utils import (
    norm_cross_corr,
    separate_watershed,
    get_gaussian2d,
)
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals import transfer_signal_axes

from sklearn.cluster import DBSCAN


class LearningSegment:
    _signal_type = "learning_segment"

    def __init__(self, factors, loadings):
        # Factors as Signal2D
        self.factors = factors
        # Corresponding loadings as Signal2D
        self.loadings = loadings

    def get_ncc_matrix(self):
        """Get the normalised correlation coefficient (NCC) matrix containing
        the NCC between each pair of segments.

        Returns
        -------
        ncc_matrix : Signal2D
            Normalised correlation coefficient matrix for loadings and factors.
        """
        # Set up empty matrices of correct size to store NCC values.
        num_comp = np.shape(self.loadings.data)[0]
        ncc_loadings = np.zeros((num_comp, num_comp))
        ncc_factors = np.zeros((num_comp, num_comp))
        # Iterate through loadings calculating NCC values.
        for i in np.arange(num_comp):
            ncc_loadings[i] = list(map(
                lambda x: norm_cross_corr(x, template=self.loadings.data[i]),
                self.loadings.data))
        # Iterate through factors calculating NCC values.
        for i in np.arange(num_comp):
            ncc_factors[i] = list(map(
                lambda x: norm_cross_corr(x, template=self.factors.data[i]),
                self.factors.data))
        # Convert matrix to Signal2D and set axes
        ncc_sig = Signal2D(np.array((ncc_loadings, ncc_factors)))
        ncc_sig.axes_manager.signal_axes[0].name = "index"
        ncc_sig.axes_manager.signal_axes[1].name = "index"
        ncc_sig.metadata.General.title = "Normalised Correlation Coefficient"

        return ncc_sig

    def correlate_learning_segments(self, corr_threshold=(0.4, 0.4), min_samples=(1, 1), distance_threshold=(0.6, 0.6),
                                    return_corr_list=False):
        """Performs correlation clustering of the components to sum similar components, subject to threshold criteria.
        The normalised cross-correlation matrices of the factors and loadings are calculated separately.
        The correlations are interpreted as distances (by taking the absolute of the correlations after subtracting
        by 1) and used for clustering separately. The clustering algorithm used is DBSCAN as implemented in
        sklearn.cluster. The factor clusters are checked against the loadings clusters, so that the final
        clusters only contain loadings and factors that were both assigned to the same cluster.
        If factors or loadings in a cluster have internal correlation scores below the respective correlation
        thresholds, they will be put in a new, separate cluster. Factors or loadings in a cluster fulfilling the
        correlation threshold criteria will be summed.

        Parameters
        ----------
        corr_threhold : tuple of float
            Correlation threshold for summation of factors and loadings, respectively.
            Must be between -1 and 1.
        min_samples : tuple of int
            The minimum number of factors and loadings, respectively, near a point for it to be considered a core point.
            See the documentation for sklearn.cluster.dbscan for details.
        distance_threshold : tuple of float
            The maximum correlation distance between two components for them to be considered to be in the same cluster.
            See the documentation for sklearn.cluster.dbscan for details.
        return_corr_list : bool, optional
            If True, an array containing the indices of the original components that each correlated component
            corresponds to, will be returned as the second variable.

        Returns
        -------
        learning_segment : LearningSegment
            LearningSegment where possibly some factors and loadings
            have been summed.
        corr_list : ndarray
            Array of the indices of the components that each correlated component corresponds to.
            The array is returned only if return_corr_list is True.

        """
        # If a mask was used during the decomposition, the factors and/or
        # loadings will contain nan, which must be converted to numbers prior
        # to the correlation calculations.
        factors = self.factors.map(np.nan_to_num, inplace=False)
        loadings = self.loadings.map(np.nan_to_num, inplace=False)

        # Calculate the normalised correlation matrix
        corr_mat = LearningSegment(factors, loadings).get_ncc_matrix().data
        # Simple correlation distance
        corr_mat_dist = abs(corr_mat - 1)

        # Find clusters of factors using DBSCAN
        factors_clusters = DBSCAN(min_samples=min_samples[0], eps=distance_threshold[0],
                                  metric="precomputed").fit(corr_mat_dist[0])

        # Find clusters of loadings using DBSCAN
        loadings_clusters = DBSCAN(min_samples=min_samples[1], eps=distance_threshold[1],
                                   metric="precomputed").fit(corr_mat_dist[1])

        # The final clusters are groups of components that are in the same cluster based on both the loadings and
        # factors. For each cluster found based on the factors, check which components are are in the same cluster in
        # loadings and group those into a final cluster.
        clusters = np.zeros_like(factors_clusters.labels_) - 1
        n = 0
        for cluster_num in np.arange(np.max(factors_clusters.labels_)):
            cluster_indices = factors_clusters.core_sample_indices_[factors_clusters.labels_ == cluster_num]
            cluster_num_loadings = np.sort(np.unique(loadings_clusters.labels_[cluster_indices]))
            for num_load in cluster_num_loadings:
                clusters[cluster_indices[np.where(loadings_clusters.labels_[cluster_indices] == num_load)]] = n
                n += 1

        # DBSCAN can assign some components to noise (label -1), but we want these to belong to separate clusters.
        if np.min(clusters) == -1:
            clusters[clusters == -1] = np.arange(n, n + len(clusters[clusters == -1]))

        # Extract the correlation matrix for the factors and loadings in each cluster and ensure that they are all
        # above the correlation thresholds.
        for cluster_num in np.sort(np.unique(clusters)):
            # Check the factors first and then loadings separately:
            for i in (0, 1):
                if len(clusters[clusters == cluster_num]) > 1:
                    corr_mat_cluster = corr_mat[i][:][np.where(clusters == cluster_num)]
                    corr_mat_cluster = corr_mat_cluster.T[np.where(clusters == cluster_num)].T

                    # Check if there are correlation scores below the correlation threshold.
                    # If so, put the components that gives lowest average value to all the others in a new cluster.
                    # Here, we assume that the initial DBSCAN clustering was good enough that the lowest average value
                    # components should not constitute their own new cluster, but be placed in separate ones for
                    # simplicity.
                    if np.min(corr_mat_cluster) < corr_threshold[i]:
                        out_indices = []
                        while np.min(corr_mat_cluster) < corr_threshold[i]:
                            corr_mat_cluster_averages = np.average(corr_mat_cluster, axis=1)
                            out_index = np.where(corr_mat_cluster_averages == np.min(corr_mat_cluster_averages))[0]
                            out_indices.append(out_index)
                            clusters[clusters == cluster_num][out_index] = n
                            corr_mat_cluster = np.delete(corr_mat_cluster, out_index, axis=0)
                            corr_mat_cluster = np.delete(corr_mat_cluster, out_index, axis=1)
                            n += 1

        correlated_factors = np.zeros(np.append(np.max(clusters) + 1, np.shape(factors.data)[1:]))
        correlated_loadings = np.zeros(np.append(np.max(clusters) + 1, np.shape(loadings.data)[1:]))
        corr_list = np.zeros(np.max(clusters) + 1, dtype=np.object)
        for m in np.unique(clusters):
            m_indices = np.where(clusters == m)[0]
            if len(m_indices) > 1:
                correlated_factors[m] = np.sum(self.factors.data[m_indices], axis=0)
                correlated_loadings[m] = np.sum(self.loadings.data[m_indices], axis=0)
            else:
                correlated_factors[m] = self.factors.data[m_indices]
                correlated_loadings[m] = self.loadings.data[m_indices]
            corr_list[m] = m_indices
        learning_segment = LearningSegment(factors=Signal2D(correlated_factors), loadings=Signal2D(correlated_loadings))
        if return_corr_list:
            return learning_segment, corr_list
        else:
            return learning_segment

    def separate_learning_segments(
        self,
        min_intensity_threshold=0,
        min_distance=2,
        min_size=10,
        max_size=np.inf,
        max_number_of_grains=np.inf,
        marker_radius=2,
        threshold=False,
        exclude_border=False,
    ):
        """Segmentation of loading maps by the watershed
        segmentation method implemented in scikit-image [1,2].

        Parameters
        ----------
        min_intensity_threshold : float
            Loading segments with a maximum intensity below
            min_intensity_threshold are discarded.
        min_distance: int
            Minimum distance (in pixels) between grains required for
            them to be considered as separate grains.
        min_size : float
            Grains with size (i.e. total number of pixels) below
            min_size are discarded.
        max_size : float
            Grains with size (i.e. total number of pixels) above
            max_size are discarded.
        max_number_of_grains : int
            Maximum number of grains included in the returned separated
            grains. If it is exceeded, those with highest peak
            intensities will be returned.
        marker_radius : float
            If 1 or larger, each marker for watershed is expanded to a disk
            of radius marker_radius. marker_radius should not exceed
            2*min_distance.
        threshold: bool
            If True, a mask is calculated by thresholding the VDF image
            by the Li threshold method in scikit-image. If False
            (default), the mask is the boolean VDF image.
        exclude_border : int or True, optional
            If non-zero integer, peaks within a distance of
            exclude_border from the boarder will be discarded. If True,
            peaks at or closer than min_distance of the boarder, will be
            discarded.

        References
        ----------
        [1] http://scikit-image.org/docs/dev/auto_examples/segmentation/
            plot_watershed.html
        [2] http://scikit-image.org/docs/dev/auto_examples/xx_applications/
            plot_coins_segmentation.html#sphx-glr-auto-examples-xx-
            applications-plot-coins-segmentation-py

        Returns
        -------
        learning_segment : LearningSegment
            LearningSegment where the loadings have been segmented and some
            factors have been repeated according to the new number of loading
            segments.
        """

        factors = self.factors.copy()
        loadings = self.loadings.copy()
        loadings_shape_x = loadings.data.shape[1]
        loadings_shape_y = loadings.data.shape[2]
        factors_shape_x = factors.data.shape[1]
        factors_shape_y = factors.data.shape[2]

        loadings_segments = np.array(
            loadings.map(
                separate_watershed,
                show_progressbar=True,
                inplace=False,
                min_distance=min_distance,
                min_size=min_size,
                max_size=max_size,
                max_number_of_grains=max_number_of_grains,
                marker_radius=marker_radius,
                threshold=threshold,
                exclude_border=exclude_border,
            ),
            dtype=np.object,
        )

        segments, factors_of_segments = [], []
        num_segs_tot = 0
        for loading_segment, factor in zip(loadings_segments, factors):
            segments = np.append(segments, loading_segment)
            num_segs = np.shape(loading_segment)[0]
            factors_of_segments = np.append(
                factors_of_segments,
                np.broadcast_to(factor, (num_segs, factors_shape_x, factors_shape_y)),
            )
            num_segs_tot += num_segs
        segments = segments.reshape((num_segs_tot, loadings_shape_y, loadings_shape_x))
        factors_of_segments = factors_of_segments.reshape(
            (num_segs_tot, factors_shape_x, factors_shape_y)
        )

        delete_indices = list(
            map(lambda x: x.max() < min_intensity_threshold, segments)
        )
        delete_indices = np.where(delete_indices)
        segments = np.delete(segments, delete_indices, axis=0)
        factors_of_segments = np.delete(factors_of_segments, delete_indices, axis=0)

        # if TraitError is raised, it is likely no segments were found
        segments = Signal2D(segments).transpose(navigation_axes=[0], signal_axes=[2, 1])
        factors_of_segments = Signal2D(factors_of_segments)
        learning_segment = LearningSegment(factors=factors_of_segments, loadings=segments)
        return learning_segment


class VDFSegment:
    _signal_type = "vdf_segment"

    def __init__(self, segments, vectors_of_segments, intensities=None):
        # Segments as Signal2D
        self.segments = segments
        # DiffractionVectors
        self.vectors_of_segments = vectors_of_segments
        # Intensities corresponding to each vector
        self.intensities = intensities

    def get_ncc_matrix(self):
        """Get the normalised correlation coefficient (NCC) matrix containing
        the NCC between each pair of segments.

        Returns
        -------
        ncc_matrix : Signal2D
            Normalised correlation coefficient matrix.
        """
        # TODO: This code should be factored out for reuse in other ncc method.
        # Set up an empty matrix of correct size to store NCC values.
        num_comp = np.shape(self.segments.data)[0]
        ncc_matrix = np.zeros((num_comp, num_comp))
        # Iterate through segments calculating NCC values.
        for i in np.arange(num_comp):
            ncc_matrix[i] = list(
                map(
                    lambda x: norm_cross_corr(x, template=self.segments.data[i]),
                    self.segments.data,
                )
            )
        # Convert matrix to Signal2D and set axes
        ncc_sig = Signal2D(ncc_matrix)
        ncc_sig.axes_manager.signal_axes[0].name = "segment index"
        ncc_sig.axes_manager.signal_axes[1].name = "segment index"
        ncc_sig.metadata.General.title = "Normalised Correlation Coefficient"
        return ncc_sig

    def correlate_vdf_segments(self, corr_threshold=0.7, vector_threshold=4,
                               segment_threshold=3, distance_threshold=0.3, min_samples=1,
                               return_corr_list=False):
        """Performs correlation matrix clustering of the segments to sum
        similar segments, subject to threshold criteria. The normalised
        cross-correlation matrix of the VDF segments is calculated.
        The correlations are interpreted as distances (by taking the absolute
        of the correlations after subtracting by 1) and used for clustering, in order
        to identify clusters of similar segments. The clustering algorithm used here is
        DBSCAN as implemented in sklearn.cluster. If segments in a cluster
        have correlation scores to other cluster segments that are smaller than the
        correlation threshold, they will be discarded. Segments fulfilling
        the correlation threshold criteria will be summed. The vectors of
        each segment sum will be updated accordingly, so that the vectors of
        each resulting segment sum are all the vectors of the original
        individual segments belonging to the cluster. Each vector is
        assigned an intensity that is the integrated intensity
        of the segment it originates from.

        Parameters
        ----------
        corr_threshold : float
            Segments in a cluster will be summed if they have a
            normalized cross-correlation above corr_threshold to all other
            segments belonging to the cluster, or otherwise the segment will
            be discarded. Must be between -1 and 1.
        vector_threshold : int, optional
            Summed segments having a number of vectors less than
            vector_threshold will be discarded.
        segment_threshold : int, optional
            Summed segment intensities that lie in a region where
            a number of segments less than segment_threshold have been
            found, are set to 0, i.e. the resulting segment will only
            have intensities above 0 where at least a number of
            segment_threshold segments have intensities above 0.
        distance_threshold : float, optional
            The maximum correlation distance between two segments for them to be considered to be in the same cluster.
            See the documentation for sklearn.cluster.dbscan for details.
        min_samples : int, optional
            The minimum number of segments near a point for it to be considered a core point.
            See the documentation for sklearn.cluster.dbscan for details.
        return_corr_list : bool, optional
            If True, an array containing the indices of the original segments that each correlated segment corresponds
            to, will be returned as the second variable.

        Returns
        -------
        vdfseg : VDFSegment
            The VDFSegment instance updated according to the image
            correlation results.
        corr_list : ndarray
            Array of the indices of the segments that each correlated segment corresponds to.
            The array is returned only if return_corr_list is True.

        """
        if segment_threshold > vector_threshold:
            raise ValueError("segment_threshold must be smaller than or "
                             "equal to vector_threshold.")

        vectors = self.vectors_of_segments.data
        segments = self.segments.data.copy()
        num_vectors = np.shape(vectors)[0]
        gvectors = np.array(np.empty(num_vectors, dtype=object))
        vector_indices = np.array(np.empty(num_vectors, dtype=object))

        for i in np.arange(num_vectors):
            gvectors[i] = np.array(vectors[i].copy())
            vector_indices[i] = np.array([i], dtype=int)
        # Calculate the normalised correlation matrix
        corr_mat = self.get_ncc_matrix().data
        # Simple correlation distance
        corr_mat_dist = abs(corr_mat - 1)

        # Find clusters of segments using DBSCAN
        clusters = DBSCAN(min_samples=min_samples, eps=distance_threshold,
                          metric="precomputed").fit(corr_mat_dist)
        cluster_indices_unique, cluster_counts = np.unique(clusters.labels_, return_counts=True)
        # Check which clusters have more than vector threshold number of entries:
        cluster_indices = cluster_indices_unique[np.where(cluster_counts >= vector_threshold)]
        num_clusters = np.shape(cluster_indices)[0]

        correlated_segments = np.zeros_like(segments[:1])
        correlated_vectors = np.array([0.0], dtype=object)
        correlated_vectors[0] = np.array(np.zeros_like(vectors[:1]))
        correlated_vector_indices = np.array([0], dtype=object)
        correlated_vector_indices[0] = np.array([0])
        # Iterate through the segment clusters that have more than
        # vector_threshold entries
        for num, cluster_index in zip(np.arange(num_clusters), cluster_indices):
            segment_cluster_indices = np.where(clusters.labels_ == cluster_index)
            # Extract the correlation matrix for only the segments in the cluster
            corr_mat_cluster = corr_mat[:][np.where(clusters.labels_ == cluster_index)]
            corr_mat_cluster = corr_mat_cluster.T[np.where(clusters.labels_ == cluster_index)].T
            # Check if there are correlation scores below the correlation threshold
            if not np.all(corr_mat_cluster[corr_mat_cluster != 1] < corr_threshold):
                if np.min(corr_mat_cluster) < corr_threshold:
                    # If correlation scores are below the threshold, delete
                    # the segment that gives lowest average value to all the others
                    while np.min(corr_mat_cluster) < corr_threshold:
                        corr_mat_cluster_averages = np.average(corr_mat_cluster, axis=1)
                        delete_index = np.where(corr_mat_cluster_averages == np.min(corr_mat_cluster_averages))
                        corr_mat_cluster = np.delete(corr_mat_cluster, delete_index, axis=0)
                        corr_mat_cluster = np.delete(corr_mat_cluster, delete_index, axis=1)
                        segment_cluster_indices = np.delete(segment_cluster_indices, delete_index, axis=1)
                # If there are a number of segments remaining that exceeds
                # or equates the vector threshold, sum the segments in the cluster
                # and add it to the list of correlated segments. Also collect the
                # vectors corresponding to the segments.
                if np.shape(segment_cluster_indices)[0] > 0:
                    if np.shape(segment_cluster_indices)[1] >= vector_threshold:
                        new_segment = np.array([np.sum(segments[segment_cluster_indices[0]], axis=0)])

                        if segment_threshold > 1:
                            segment_check = np.zeros_like(segments[segment_cluster_indices[0]], dtype=int)
                            segment_check[np.where(segments[segment_cluster_indices[0]])] = 1
                            segment_check = np.sum(segment_check, axis=0, dtype=int)
                            segment_mask = np.zeros_like(segments[0], dtype=bool)
                            segment_mask[np.where(segment_check >= segment_threshold)] = 1
                            new_segment = new_segment * segment_mask

                        correlated_segments = np.append(correlated_segments, new_segment, axis=0)
                        new_vectors = np.array([0], dtype=object)
                        new_vectors[0] = np.concatenate(gvectors[segment_cluster_indices[0]], axis=0).reshape(-1, 2)
                        correlated_vectors = np.append(correlated_vectors, new_vectors, axis=0)
                        new_indices = np.array([0], dtype=object)
                        new_indices[0] = np.concatenate(vector_indices[segment_cluster_indices[0]],
                                                        axis=0).reshape(-1, 1)
                        correlated_vector_indices = np.append(correlated_vector_indices, new_indices, axis=0)

        correlated_segments = np.delete(correlated_segments, 0, axis=0)
        correlated_vectors = np.delete(correlated_vectors, 0, axis=0)
        correlated_vector_indices = np.delete(correlated_vector_indices, 0, axis=0)
        correlated_vector_intensities = np.array(np.empty(len(correlated_vectors)), dtype=object)

        # Sum the intensities in the original segments and assign those to the
        # correct vectors by referring to vector_indices.
        # If segment_mask has been used, use the segments as masks too.
        if segment_threshold > 1:
            for i in range(len(correlated_vectors)):
                correlated_vector_intensities[i] = np.zeros(len(correlated_vector_indices[i]))
                segment_mask = np.zeros_like(segment_mask)
                segment_mask[np.where(correlated_segments[i])] = 1
                segment_intensities = np.sum(self.segments.data * segment_mask, axis=(1, 2))
                for n, index in zip(range(len(correlated_vector_indices[i])),
                                    correlated_vector_indices[i]):
                    correlated_vector_intensities[i][n] = np.sum(segment_intensities[index])
        else:
            segment_intensities = np.sum(self.segments.data, axis=(1, 2))
            for i in range(len(correlated_vectors)):
                correlated_vector_intensities[i] = np.zeros(len(correlated_vector_indices[i]))
                for n, index in zip(range(len(correlated_vector_indices[i])), correlated_vector_indices[i]):
                    correlated_vector_intensities[i][n] = np.sum(segment_intensities[index])

        vdfseg = VDFSegment(Signal2D(correlated_segments), DiffractionVectors(correlated_vectors),
                            correlated_vector_intensities)

        # Transfer axes properties of segments
        vdfseg.segments = transfer_signal_axes(vdfseg.segments, self.segments)
        n = vdfseg.segments.axes_manager.navigation_axes[0]
        n.name = "n"
        n.units = "number"
        if return_corr_list:
            return vdfseg, correlated_vector_indices
        else:
            return vdfseg

    def get_virtual_electron_diffraction(self, calibration, shape, sigma):
        """ Obtain a virtual electron diffraction signal that consists
        of one virtual diffraction pattern for each segment. The virtual
        diffraction pattern is composed of Gaussians centered at each
        vector position. If given, the integrated intensities of each
        vector will be taken into account by multiplication with the
        Gaussians.

        Parameters
        ----------
        calibration : float
            Reciprocal space calibration in inverse Angstrom per pixel.
        shape : tuple
            Shape of the signal, (shape_x, shape_y) in pixels, where
            shape_x and shape_y are integers.
        sigma : float
            The standard deviation of the Gaussians in inverse Angstrom
            per pixel. 'calibration' is a decent starting value.

        Returns
        -------
        virtual_ed : ElectronDiffraction2D
            Virtual electron diffraction signal consisting of one
            virtual diffraction pattern for each segment.
        """
        vectors = self.vectors_of_segments.data
        segments = self.segments.data
        num_segments = np.shape(segments)[0]

        if self.intensities is None:
            raise ValueError(
                "The VDFSegment does not have the attribute  "
                "intensities, required for this method."
            )
        else:
            intensities = self.intensities

        # TODO: Refactor this to use the diffsims simulation to plot functionality
        size_x, size_y = shape[0], shape[1]
        cx, cy = -size_x / 2 * calibration, -size_y / 2 * calibration
        x, y = np.indices((size_x, size_y))
        x, y = x * calibration + cx, y * calibration + cy
        virtual_ed = np.zeros((size_x, size_y, num_segments))

        for i in range(num_segments):
            # Allow plotting for segments that are only associated with
            # one vector.
            if np.shape(np.shape(vectors[i]))[0] <= 1:
                virtual_ed[..., i] = get_gaussian2d(
                    intensities[i],
                    vectors[i][..., 0],
                    vectors[i][..., 1],
                    x=x,
                    y=y,
                    sigma=sigma,
                )
            # Allow plotting for segments associated with several vectors.
            else:
                virtual_ed[..., i] = sum(
                    list(
                        map(
                            lambda a, xo, yo: get_gaussian2d(
                                a, xo, yo, x=x, y=y, sigma=sigma
                            ),
                            intensities[i],
                            vectors[i][..., 0],
                            vectors[i][..., 1],
                        )
                    )
                )

        virtual_ed = ElectronDiffraction2D(virtual_ed.T)

        return virtual_ed
