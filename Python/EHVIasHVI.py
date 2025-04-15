from __future__ import annotations

from abc import abstractmethod
from itertools import product 
from typing import List, Optional

import torch
from botorch.acquisition.multi_objective.analytic import MultiObjectiveAnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.utils.multi_objective.box_decompositions.utils import (
    _expand_ref_point,
    _pad_batch_pareto_frontier,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal

import itertools
from scipy import stats

class ExpectedHypervolumeImprovementAsHVI2(MultiObjectiveAnalyticAcquisitionFunction):
    # the init and point set pre-processing codes are duplicated from ExpectedHypervolumeImprovement class in BoTorch 
    def __init__(
        self,
        model: Model,
        ref_point: list[float],
        Y: Tensor,
        sort: bool = True,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        r"""Expected Hypervolume Improvement supporting m>=2 outcomes.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0]
            >>> EHVI = ExpectedHypervolumeImprovementAsHVI(model, ref_point, partitioning)
            >>> ehvi = EHVI(test_X)

        Args:
            model: A fitted model.
            ref_point: A list with `m` elements representing the reference point
                (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the outcome values (i.e., after
                applying `posterior_transform` if provided).
            posterior_transform: A `PosteriorTransform`.
        """        

        self.sort = torch.tensor(sort, dtype=torch.bool)
        
        if Y.isnan().any():
            raise ValueError(
                "NaN inputs are not supported. Got Y with "
                f"{Y.isnan().sum()} NaN values."
            )
        self._Y = Y
        self._validate_inputs()
        self.num_outcomes = len(ref_point)
        ref_point = torch.tensor(
            ref_point,
            dtype=self.Y.dtype,
            device=self.Y.device,
        )

        super().__init__(model=model, posterior_transform=posterior_transform)
        self.normal = Normal(0, 1)
        self.register_buffer("ref_point", ref_point)
        self._pareto_Y = self._compute_pareto_Y()

    @property
    def pareto_Y(self) -> Tensor:
        r"""This returns the non-dominated set.

        Returns:
            A `n_pareto x m`-dim tensor of outcomes.
        """
        return self._pareto_Y

    @property
    def Y(self) -> Tensor:
        r"""Get the raw outcomes.

        Returns:
            A `n x m`-dim tensor of outcomes.
        """
        return self._Y

    def _compute_pareto_Y(self) -> Tensor:
        # is_non_dominated assumes maximization
        if self.Y.shape[-2] == 0:
            return self.Y
        # assumes maximization
        pareto_Y = _pad_batch_pareto_frontier(
            Y=self.Y,
            ref_point=_expand_ref_point(
                ref_point=self.ref_point, batch_shape=self.batch_shape
            ),
        )
        if not self.sort:
            return pareto_Y
        # sort by the last objective
        pareto_Y = pareto_Y[torch.argsort(pareto_Y[:, -1])]

        return pareto_Y

    def _validate_inputs(self) -> None:
        self.batch_shape = self.Y.shape[:-2]
        if len(self.batch_shape) > 0:
            raise NotImplementedError(
                f"{type(self).__name__} only supports a single "
                f"batch dimension, but got {len(self.batch_shape)} "
                "batch dimensions."
            )

    def reset(self) -> None:
        r"""Reset non-dominated front and decomposition."""
        self._validate_inputs()
        self._pareto_Y = self._compute_pareto_Y()

    def _update_Y(self, Y: Tensor) -> bool:
        r"""Update the set of outcomes.

        Returns:
            A boolean indicating if _neg_Y was initialized.
        """
        if Y.isnan().any():
            raise ValueError(
                "NaN inputs are not supported. Got Y with "
                f"{Y.isnan().sum()} NaN values."
            )
        if Y.shape[-1] != self.num_outcomes:
            raise ValueError(
                "Dimension of Y does not match the dimension of reference point."
                )
        if self.Y is not None:
            self._Y = torch.cat([self.Y, Y], dim=-2)
            return False
        return True

    def update(self, Y: Tensor) -> None:
        r"""Update non-dominated front and decomposition.

        By default, the partitioning is recomputed. Subclasses can override
        this functionality.

        Args:
            Y: A `(batch_shape) x n x m`-dim tensor of new, incremental outcomes.
        """
        self._update_Y(Y=Y)
        self.reset()

    def set_ref_point(self, ref_point: list[float]) -> None:
        if len(ref_point) != self.num_outcomes:
            raise ValueError(
                "Dimension of reference point does not match the dimension of point set."
                )
        self.ref_point = torch.tensor(
            ref_point,
            dtype=self.pareto_Y.dtype,
            device=self.pareto_Y.device,
        )
        self.reset()

    # our implementation of QHV-II algorithm
    # see Jaszkiewicz, A. 2018. Improved quick hypervolume algorithm. Computers & Operations Research, 90: 72–83.
    def qhvii(self, points: Tensor, ub: Tensor, lb: Tensor, offset: Tensor) -> Tensor:
        n, m = points.shape
        if n == 0:
             return torch.tensor(0, dtype=points.dtype, device=points.device)
        if n == 1:
             return torch.prod(torch.minimum(points, ub) - lb)
        if n == 2:
             return torch.prod(torch.minimum(points, ub) - lb, axis=1).sum() - torch.prod(torch.minimum(torch.min(points, axis=0).values, ub) - lb)
        offset = offset + 1
        if offset >= m:
            offset = 0
        volumes = torch.prod(torch.minimum(points, ub) - lb, axis=1)
        pivot = torch.argmax(volumes)
        volume = volumes[pivot]
        newlb = torch.clone(lb)
        newub = torch.clone(ub)
        for i in range(0, m):
            j = i + offset
            if j >= m:
                j = j - m
            j2 = -1
            if i > 0:
                j2 = i + offset - 1
                if j2 >= m:
                    j2 = j2 - m
                newub[j2] = torch.min(ub[j2], points[pivot, j2])
                newlb[j2] = lb[j2]
            ind = torch.where(torch.minimum(points[:, j], ub[j]) > torch.min(points[pivot, j], ub[j]))[0]
            if ind.shape[0] > 0:
                newlb[j] = torch.min(ub[j], points[pivot, j])
                volume = volume + self.qhvii(points.index_select(-2, ind), newub, newlb, offset)
        return volume
    
    # compute hv of self.work_points with respect to self.work_points using QHV-II
    def hvmd(self) -> Tensor:        
        ub = torch.max(self.work_points, -2).values
        lb = torch.zeros(self.work_points.shape[-1], dtype=self.work_points.dtype, device=self.work_points.device)
        return self.qhvii(self.work_points, ub, lb, torch.zeros(1, dtype=int))
    
    # compute hv of A with respect to zero reference point using QHV-II
    def hvmd2(self, A: Tensor) -> Tensor:        
        ub = torch.max(A, -2).values
        lb = torch.zeros(A.shape[-1], dtype=A.dtype, device=A.device)
        return self.qhvii(A, ub, lb, torch.zeros(1, dtype=int))

    # interface for computing hv of self.work_points with respect to self.work_points
    def _compute_hypervolume(self) -> Tensor:
        r"""Compute hypervolume that is dominated by the Pareto Froniter.

        Returns:
            A `(batch_shape)`-dim tensor containing the hypervolume dominated by
                each Pareto frontier.
        """
        if self.work_points is None or self.work_points.shape[-2] == 0:
            return torch.zeros(
                self.pareto_Y.shape[:-2],
                dtype=self._neg_pareto_Y.dtype,
                device=self._neg_pareto_Y.device,
            )

        # 2-D HV is easy to handle
        # auto-differentiation is applicable to this calculation
        if self.work_points.shape[-1] == 2:
             if not self.sort:
                  self.work_points = self.work_points[torch.argsort(self.work_points[:, 1]), :]
             return self.work_points[-1, 0] * self.work_points[-1, 1] - torch.dot(torch.diff(self.work_points[:, 0]), self.work_points[:-1, 1])

        # auto-differentiation is not applicable to 3- and more dimensional cases because we use QHV-II
        if self.work_points.shape[-1] == 3:
             # TODO: implement HV3D algorithm for 3-D case
             hv = self.hvmd()
        else:
             hv = self.hvmd()
        
        return hv

    # interface for computing hv of A with respect to zeros
    # QHV-II can handle the case that points in A dominate each other
    def compute_hypervolume(self, A: Tensor) -> Tensor:
        r"""Compute hypervolume that is dominated by the Pareto Froniter.

        Returns:
            A `(batch_shape)`-dim tensor containing the hypervolume dominated by
                each Pareto frontier.
        """
        if A.shape[-2] == 0:
             return torch.tensor([0], dtype=A.dtype, device=A.device)
        if A.shape[-1] == 2:
             #hv = 0
             #A = A[torch.argsort(A[:, -1], descending=True)]
             #start = 0
             #ub = A[0, 0]
             #last = A[0, 1]
             #for i in range(A.shape[-2]):
             #      if A[i, 1] == ub:
             #             if A[i, 0] > ub:
             #                   ub = A[i, 0]
             #      else:
             #             start = i
             #             break
             #for i in range(start, A.shape[-2]):
             #      if A[i, 0] > ub:
             #             hv = hv + (last - A[i, 1]) * ub
             #             ub = A[i, 0]
             #             last = A[i, 1]
             #hv = hv + ub * last 
             #return hv
             indices = is_non_dominated(A)
             A = A[indices]
             A = A[torch.argsort(A[:, -1])]
             return A[-1, 0] * A[-1, 1] - torch.dot(torch.diff(A[:, 0]), A[:-1, 1])
        else:
             return self.hvmd2(A)

    # EI function
    def EI(self, T: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        r"""Compute ei function.

        Args:
            T: A `n_points x m`-dim tensor of points
            mu: A `batch_shape x 1 x m`-dim tensor of means
            sigma: A `batch_shape x 1 x m`-dim tensor of standard deviations (clamped).

        Returns:
            A `batch_shape x n_points x m`-dim tensor of values.
        """
        w_normalized = (mu - T) / torch.sqrt(sigma)
        return torch.sqrt(sigma) * (self.normal.log_prob(w_normalized).exp() + w_normalized * self.normal.cdf(w_normalized))

    # qMinEI function
    def qMinEI(self, T: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:    
    # Torch does not support computing c.d.f. of multivariate normal distribution, so we resort to scipy.stats
    # auto-differentiation is not applicable in the cases of q > 1
    # The implementation is referred to the method for qEI
    # see Chevalier, C.; and Ginsbourger, D. 2013. Fast Computation of the Multi-Points Expected Improvement with Applications in Batch Selection, LION 7, 59-69.
    
    # T: n x m tensor
    # mu: q x m tensor, mean for each objective
    # sigma: q x q x m tensor, covariance matrix for each objective
    # output: n x m tensor, each entry is qMinEI of corresponding entry in T
        q = mu.shape[-2]
        if q == 1:
            return self.EI(T, mu, sigma)
        n, m = T.shape
        minEI = torch.zeros((n, m), dtype=T.dtype, device=T.device)
        for dim in range(m):
            for k in range(q):
                L = -torch.eye(q, dtype=T.dtype, device=T.device)
                L[:, k] = 1
                L[k, k] = -1
                mu_Y = mu[:, dim] @ L.T
                sigma_Y = L @ sigma[:, :, dim] @ torch.transpose(L, 0, 1)
                b = torch.zeros((n, q), dtype=T.dtype, device=T.device) 
                b[:, k] = -T[:, dim]
                minEI[:, dim] = minEI[:, dim] + (mu[k, dim] - T[:, dim]) * stats.multivariate_normal.cdf(x=b.numpy(), mean=mu_Y.numpy(), cov=sigma_Y.numpy())
                for i in range(q):
                    sigma_Y_row = sigma_Y[i, :].unsqueeze(0)
                    sigma_Y_column = sigma_Y[i, :].unsqueeze(-1)
                    c_minusOne = b - mu_Y - (torch.tile(b[:, i], (q, 1)).transpose(0, 1) - mu_Y[i]) * sigma_Y_row / sigma_Y[i, i]
                    c1 = c_minusOne[:, 0:i]
                    c2 = c_minusOne[:, i+1:]
                    c_minusOne = torch.cat((c1, c2), dim=-1)
                    sigma_minusOne = sigma_Y - sigma_Y_column @ sigma_Y_row / sigma_Y[i, i]
                    sigma1 = sigma_minusOne[:, 0:i]
                    sigma2 = sigma_minusOne[:, i+1:]
                    sigma_minusOne = torch.cat((sigma1, sigma2), dim=-1)
                    sigma1 = sigma_minusOne[0:i, :]
                    sigma2 = sigma_minusOne[i+1:, :]
                    sigma_minusOne = torch.cat((sigma1, sigma2), dim=-2)
                    minEI[:, dim] = minEI[:, dim] + self.normal.log_prob((b[:, i]-mu_Y[i])/torch.sqrt(sigma_Y[i, i])).exp()/torch.sqrt(sigma_Y[i, i]) * sigma_Y[i, k] * stats.multivariate_normal.cdf(c_minusOne.numpy(), mean=torch.zeros(q-1).numpy(), cov=sigma_minusOne.numpy())
        return minEI

    # this function is used in calculating gradient of hypervolume with respect to Y
    def _project(
        self, axis: int, i: int, pareto_front: Tensor) -> Tuple[Tensor, Tensor]:
        """projecting the Pareto front along `axis` with respect to the i-th point"""
        y = pareto_front[i, :]
        # projection: drop the `axis`-th dimension
        y1 = y[0:axis]
        y2 = y[axis+1:]
        y_ = torch.cat((y1, y2), 0)
        idx = torch.nonzero(pareto_front[:, axis] > y[axis]).squeeze(-1)
        if idx.shape[0] == 0:
            return y_, torch.tensor([], dtype=pareto_front.dtype, device=pareto_front.device)
        pareto_front1 = pareto_front[idx, 0:axis]
        pareto_front2 = pareto_front[idx, axis+1:]
        pareto_front_ = torch.cat((pareto_front1, pareto_front2), 1)
        if pareto_front_.shape[1] == 1:
            pareto_indices = torch.tensor([torch.argmin(pareto_front_)])
        return y_, pareto_front_

    # compute HVI
    def _hypervolume_improvement(self, x: Tensor, pareto_front: Tensor) -> Tensor:
        return torch.prod(x) - self.compute_hypervolume(torch.minimum(pareto_front, x.unsqueeze(0)))

    def hypervolume_dY(self, pareto_front: Tensor) -> Tensor:
        """compute the gradient of hypervolume indicator in the objective space, i.e.,
        \partial HV / \partial Y
        Author: Hao Wang
        Downloaded and revised from: https://github.com/wangronin/HypervolumeDerivatives
        see Deutz, A.; Emmerich, Michael T. M.; Wang, H. The Hypervolume Indicator Hessian Matrix: Analytical Expression, Computational Time Complexity, and Sparsity, EMO2023, 405-418.

        Args:
            pareto_front: the Pareto front of shape (n_points, n_objectives)

        Returns:
            the hypervolume indicator gradient of shape (n_points, n_objective)
        """
        N, dim = pareto_front.shape

        if dim == 1:  # 1D case
            HVdY = torch.tensor([[1]], dtype=pareto_front.dtype, device=pareto_front.device)
        elif dim == 2:  # 2D case
            HVdY = torch.zeros((N, 2), dtype=pareto_front.dtype, device=pareto_front.device)
            # sort the pareto front with repsect to increasing y1 and decreasing y2
            # NOTE: weakly dominbated pointed are handled here
            _, tags = torch.unique(pareto_front[:, 0], return_inverse=True)
            idx1 = torch.argsort(-tags, stable=True)
            _, tags = torch.unique(pareto_front[idx1, 1], return_inverse=True)
            idx2 = torch.argsort(-tags, stable=True)
            idx = idx1[idx2]
            sorted_pareto_front = pareto_front[idx]
            y1 = sorted_pareto_front[:, 0]
            y2 = sorted_pareto_front[:, 1]
            HVdY[idx, 0] = y2 - torch.cat((y2[1:], torch.tensor([0.], dtype=pareto_front.dtype, device=pareto_front.device)), 0)
            HVdY[idx, 1] = y1 - torch.cat((torch.tensor([0.], dtype=pareto_front.dtype, device=pareto_front.device), y1[0:-1]), 0)
        else:
            HVdY = torch.zeros((N, dim), dtype=pareto_front.dtype, device=pareto_front.device)
            # higher dimensional cases: recursive computation
            for i in range(N):
                for k in range(dim):
                    y_, pareto_front_ = self._project(k, i, pareto_front)
                    if pareto_front_.shape[0] == 0:
                          HVdY[i, k] = torch.prod(y_)
                    else:
                          HVdY[i, k] = self._hypervolume_improvement(y_, pareto_front_)
        return HVdY

    # our implementation for computing gradient of EHVI
    # assuming GP model has been trained
    # X is batch_size x 1 x m tensor
    def gradient(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(
            X, posterior_transform=self.posterior_transform
        )            
        mu = posterior.mean
        mu = mu - self.ref_point
        posterior_sigma = posterior.variance
        sigma = posterior_sigma.clamp_min(1e-9)
        pareto_Y = self.pareto_Y - self.ref_point

        batch_size, m = mu.shape
        q = X.shape[-2]
        
        dim = X.shape[-1] 

        if q != 1:
            raise NotImplementedError(
                f"{type(self).__name__} only supports computing gradient in the case of q = 1 now"
            )
 
        self.work_ref = self.EI(T=torch.zeros((batch_size, self.num_outcomes), dtype=pareto_Y.dtype, device=pareto_Y.device), mu=mu, sigma=sigma)

        volume = torch.prod(self.work_ref, axis=-1)

        if pareto_Y.shape[0] == 0:
            volume.backward()
            return X.grad.clone()

        result = torch.zeros((batch_size, X.shape[-1]), dtype=pareto_Y.dtype, device=pareto_Y.device)
        grad = torch.zeros((1, X.shape[-1]), dtype=pareto_Y.dtype, device=pareto_Y.device)
        for i in range(batch_size):
            self.work_points = self.work_ref[i, :] - self.EI(T=pareto_Y, mu=mu[i, :], sigma=sigma[i, :])
            # compute partial derivative of HV(hat(A)) with respect to Y
            HVdY = self.hypervolume_dY(self.work_points).detach()
            # chain rule EHVI_dX = volume_dX - HV_dY * Y_dX using auto-differentiation
            EHVI_dX = (volume - torch.mul(HVdY, self.work_points).sum()).backward(retain_graph=True)
            result[i, :] = X.grad.clone() - grad
            grad = X.grad.clone()
        
        return result

    # our implementation of forward function for computing EHVI and qEHVI
    # assuming GP model has been trained
    # X is batch_size x q x m tensor
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(
            X, posterior_transform=self.posterior_transform
        )        

        mu = posterior.mean
        mu = mu - self.ref_point
        pareto_Y = self.pareto_Y - self.ref_point

        batch_size, q, m = mu.shape
        # EHVI using Equation (9) in the paper
        if q == 1:
                posterior_sigma = posterior.variance
                sigma = posterior_sigma.clamp_min(1e-9)

                self.work_ref = self.EI(T=torch.zeros((batch_size, 1, self.num_outcomes), dtype=pareto_Y.dtype, device=pareto_Y.device), mu=mu, sigma=sigma)
                volume = torch.prod(self.work_ref, axis=-1).squeeze(-1)

                if pareto_Y.shape[0] == 0:
                    return volume

                result = torch.zeros((batch_size,), dtype=pareto_Y.dtype, device=pareto_Y.device)                
                squeezed_mu = mu.squeeze(-2)
                squeezed_sigma = sigma.squeeze(-2)
                self.work_ref = self.work_ref.squeeze(-2)
                for i in range(batch_size):
                    self.work_points = self.work_ref[i, :] - self.EI(T=pareto_Y, mu=squeezed_mu[i, :], sigma=squeezed_sigma[i, :])
                    result[i] = volume[i] - self._compute_hypervolume()
        else:
                # qEHVI using Equation (17) in the paper
                result = torch.zeros((batch_size,), dtype=pareto_Y.dtype, device=pareto_Y.device)
                # joint distribution of q test samples
                covariance_matrix = posterior.mvn.covariance_matrix
                for i in range(batch_size):
                        sigma_qMinEI = torch.zeros((q, q, m), dtype=pareto_Y.dtype, device=pareto_Y.device)
                        for j in range(mu.shape[-1]):
                                sigma_qMinEI[:, :, j] = covariance_matrix.index_select(-3, torch.tensor([i])).index_select(-2, torch.arange(j*q, j*q+q)).index_select(-1, torch.arange(j*q, j*q+q)).squeeze(-3)
                        # compute terms in Equation (17) for I={1}, I={2}, ..., I={q}
                        # in this case, we do not need to use qMinEI but just EI
                        for j in range(q):
                                self.work_ref = self.EI(T=torch.zeros((1, self.num_outcomes), dtype=pareto_Y.dtype, device=pareto_Y.device), mu=mu[i, j, :], sigma=sigma_qMinEI[j, j, :])
                                volume = torch.prod(self.work_ref, axis=-1)
                                if pareto_Y.shape[0] == 0:
                                          result[i] = result[i] + volume
                                else:
                                          self.work_points = self.work_ref - self.EI(T=pareto_Y, mu=mu[i, j, :], sigma=sigma_qMinEI[j, j, :])
                                          result[i] = result[i] + volume - self._compute_hypervolume()

                        all_set = [i for i in range(q)]
                        # compute terms in Equation (17) for I \subset {1, ..., q} such that |I| >= 2
                        for j in range(2, q+1):
                                # generate all possible I such that |I| = j
                                for sub in itertools.combinations(all_set, j):
                                          comb = torch.tensor([k for k in sub])
                                          self.work_ref = self.qMinEI(T=torch.zeros((1, self.num_outcomes), dtype=pareto_Y.dtype, device=pareto_Y.device), mu=mu[i, :, :].index_select(-2, comb), sigma=sigma_qMinEI.index_select(-2, comb).index_select(-3, comb))
                                          if pareto_Y.shape[0] == 0:
                                                      if j % 2 == 1:
                                                                  result[i] = result[i] + torch.prod(self.work_ref, axis=-1)
                                                      else:
                                                                  result[i] = result[i] - torch.prod(self.work_ref, axis=-1)
                                          else:
                                                      self.work_points = self.work_ref - self.qMinEI(T=pareto_Y, mu=mu[i, :, :].index_select(-2, comb), sigma=sigma_qMinEI.index_select(-2, comb).index_select(-3, comb))
                                                      if j % 2 == 1:
                                                                  result[i] = result[i] + torch.prod(self.work_ref, axis=-1) - self._compute_hypervolume()
                                                      else:
                                                                  result[i] = result[i] - torch.prod(self.work_ref, axis=-1) + self._compute_hypervolume()
        return result