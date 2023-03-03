from argparse import Namespace
from pdb import set_trace as TT
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from clients.python.src.main.proto.minecraft_pb2 import AIR

from models.tensoRF import TensorVMSplit

from models.mc_blocks import init_mc_blocks, block_names_to_enum, block_enum_to_idx


class TensorVMSplitBlocks(TensorVMSplit):
    """Learn a voxel grid that maps to a grid of pre-defined 3D blocks (e.g. Minecraft blocks)"""

    def __init__(self, *args, **kwargs):
        cfg = Namespace(**kwargs)
        self.DUMMY_BLOCK_GRID = cfg.cfg.dummy_block_grid
        self.FULL_BLOCK_TEXTURE = cfg.cfg.full_block_texture
        self.IMPOSE_BLOCK_DENSITY = cfg.cfg.impose_block_density
        self.SNAP_TO_BLOCK_SURFACES = cfg.cfg.snap_to_block_surfaces

        self.blocks = init_mc_blocks(self.FULL_BLOCK_TEXTURE)
        self.n_blocks = len(self.blocks)

        super(TensorVMSplitBlocks, self).__init__(*args, **kwargs)

        if self.DUMMY_BLOCK_GRID:
            self._dummy_grid = self._init_dummy_grid(self.DUMMY_BLOCK_GRID)

        self.block_probs_mat = torch.nn.Linear(sum(self.app_n_comp), self.n_blocks, bias=False).to(self.device)
        self.blocks = self.blocks.to(self.device)
        self._has_computed = False

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        rays_pts, interpx, not_mask_outbox = super().sample_ray(rays_o, rays_d, is_train=is_train, N_samples=N_samples)


        if self.SNAP_TO_BLOCK_SURFACES:
            # FIXME: Isn't this snapping to the surfaces halfway inside the block?
            
            # print(rays_pts.max(0)[0].max(0)[0])
            # print(rays_pts.min(0)[0].min(0)[0])

            # Calculate intervals at which surfaces appear on the grid
            surface_intervals = self.units
            # surface_intervals = 1 / self.units

            # How far is each point from the nearest block surface along each axis?
            # How far back along the ray to nearest surface?
            # Account for aabb
            rays_pts = rays_pts + self.units / 2

            rays_pts_offsets_0 = - (rays_pts % surface_intervals)
            # How far forward?
            rays_pts_offsets_1 = surface_intervals + rays_pts_offsets_0
            rays_pts_offsets = torch.stack([rays_pts_offsets_0, rays_pts_offsets_1], dim=-1)

            # For each point and each axis, in which direction is the nearest surface?
            rays_pts_min_dir = torch.argmin(torch.abs(rays_pts_offsets), dim=-1)
            # Focus only on offsets in the relevant direction
            rays_pts_offsets = torch.gather(rays_pts_offsets, dim=-1, index=rays_pts_min_dir[..., None])[..., 0]  # ok copilot

            # Make sure offsets are not too large
            assert (rays_pts_offsets < (surface_intervals / 2)).all()

            # Convert the offsets to be in terms of ray direction vectors (i.e. how many applications of direction vector to
            # achieve offset in each axis).
            rays_pts_offsets_t = rays_pts_offsets / rays_d[:, None, :]
            # For each point, along which axis in the nearest surface?
            rays_pts_min_ax = torch.argmin(torch.abs(rays_pts_offsets_t), dim=-1)
            # How much to move along the ray to get to the nearest surface.
            rays_pts_t = torch.gather(rays_pts_offsets_t, dim=-1, index=rays_pts_min_ax[..., None])[..., 0]
            
            # Attempt to snap to grid (probably wrong??)
            # rays_pts += rays_pts_t[:,:,None] * torch.randn_like(rays_pts_t[:,:,None])/ 100 * rays_d[:,None,:]
            rays_pts += rays_pts_t[:,:,None] * rays_d[:,None,:]

            # Perturb the rays for fun (wait what)
            # rays_pts += (torch.randn_like(rays_pts_t[:,:,None]) * rays_d[:,None,:])

            # FIXME            rays_pts_offsets_0 = - (rays_pts % self.units)

            rays_pts_offsets_0 = - (rays_pts % surface_intervals)
            rays_pts_offsets_1 = surface_intervals + rays_pts_offsets_0
            rays_pts_offsets = torch.stack([rays_pts_offsets_0, rays_pts_offsets_1], dim=-1)
            # Assert that the points are now on surfaces
            if not rays_pts_offsets.abs().min(-1)[0].min(-1)[0].max() < 1e-3:
                TT()
            # assert (rays_pts % self.units).abs().min(-1)[0].max() < 1e-3

        # Round sampled points to nearest point along the border of some block
        # rays_pts = torch.round(rays_pts / self.units) * self.units
        # rays_pts = rays_pts + self.aabb[0] / 4
        rays_pts = rays_pts - self.units / 2

        return rays_pts, interpx, not_mask_outbox


    def forward(self, *args, **kwargs):
        ret = super().forward(*args, **kwargs)
        self._has_computed = False
        return ret

    # def compute_densityfeature(self, xyz_sampled):
    #     mask = None
    def compute_densityfeature(self, xyz_sampled, mask):
        self._app_densities = torch.zeros(xyz_sampled.view(-1, 3).shape[0], 4).to(self.device)

        if not self.IMPOSE_BLOCK_DENSITY:
            # return super().compute_densityfeature(xyz_sampled, mask=mask)
            return super().compute_densityfeature(xyz_sampled, mask)

        elif self.IMPOSE_BLOCK_DENSITY in (2, 3):
            # Compute density separately but over the same voxel grid
            
            if mask is not None:
                xyz_sampled = xyz_sampled[mask]

            # plane + line basis
            coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
            coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

            sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
            for idx_plane in range(len(self.density_plane)):
                plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                    align_corners=True, mode='nearest').view(-1, *xyz_sampled.shape[:1])
                line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                                align_corners=True, mode='nearest').view(-1, *xyz_sampled.shape[:1])
                sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

            if self.IMPOSE_BLOCK_DENSITY == 3:
                min_sigma, max_sigma = sigma_feature.min(), sigma_feature.max()
                # Stack densities with their negatives
                sigma_feature = torch.stack((sigma_feature, -sigma_feature), dim=-1)
                # Take gumbel softmax of the density, ultimately returning a binary array
                sigma_feature = F.gumbel_softmax(sigma_feature, tau=0.1, hard=True, dim=-1)
                # Turn oneohot into discrete
                sigma_feature *= torch.tensor([50, 0]).to(self.device)
                sigma_feature = sigma_feature.sum(-1)
                breakpoint()

            return sigma_feature

        elif self.IMPOSE_BLOCK_DENSITY == 1:

            if mask is not None:
                # This is only the case when computing densities independent of alpha, so shouldn't be a problem that
                # we would be missing original `xyz_sampled` shape when computing appearance features.
                xyz_sampled = xyz_sampled[mask]

                if not self._has_computed:
                    self._app_densities[mask.view(-1)] = self.compute_appdensityfeature(xyz_sampled)
                    # out = self.compute_appdensityfeature(xyz_sampled)
                    self._has_computed = True
                return self._app_densities[mask.view(-1), 0]

            else:
                if not self._has_computed:
                    self._app_densities = self.compute_appdensityfeature(xyz_sampled)
                    self._has_computed = True
                return self._app_densities[:, 0]

        else:
            raise NotImplementedError

    # TODO: Save app_densities then return subset of it when querying appearance (i.e. apply appearance mask directly to
    # app_densities, probably a level above this function.)
    def compute_appfeature(self, xyz_sampled, mask):
        xyz_sampled = xyz_sampled[mask]
    # def compute_appfeature(self, xyz_sampled):
        if not self._has_computed:
            # But this never happens prior to density (supposing we are also computing that using bloks)... right?
            if self.IMPOSE_BLOCK_DENSITY == 1:
                TT()
            self._app_densities[mask.view(-1)] = self.compute_appdensityfeature(xyz_sampled)
            # out = self.compute_appdensityfeature(xyz_sampled)
            self._has_computed = True
        return self._app_densities[mask.view(-1), 1:]
        # return out[:, 1:]

    def _compute_blockprobs_from_dummygrid(self, xyz_sampled):
        """Compute app and density features from a dummy grid"""
        block_probs = F.grid_sample(self._dummy_grid[None], xyz_sampled[None, :, None, None, :], align_corners=True, mode='nearest')[0, :, :, 0, 0].T
        return block_probs

    def _init_dummy_grid(self, mode):
        """Create a dummy grid to use for sampling block probabilities"""
        if mode == 1:
            dummy_grid = torch.zeros((self.n_blocks, *tuple(self.gridSize)), device=self.device)
            mid_point = self.gridSize // 2
            dummy_grid[0][tuple(mid_point)] = 1
            # dummy_grid[0][0, 0, 0] = 1
            return dummy_grid

        elif mode == 2:

            dummy_grid_disc = torch.randint(high=self.n_blocks, size=tuple(self.gridSize), device=self.device)
            # From discrete to onehot
            dummy_grid = torch.nn.functional.one_hot(dummy_grid_disc, self.n_blocks).float()
            dummy_grid = rearrange(dummy_grid, 'x y z c -> c x y z')
            return dummy_grid

        elif mode == 3:
            """Compute app and density features from a dummy grid"""
            dummy_grid = torch.zeros((self.n_blocks, *self.gridSize), device=self.device)
            dummy_grid[0] = 1.0  # stone
            dummy_grid[:, :, :, 1:2] = 0.0
            dummy_grid[1, :, :, 1:2] = 1.0
            dummy_grid[:, :, :, 2:3] = 0.0
            dummy_grid[2, :, :, 2:3] = 1.0
            dummy_grid[:, :, :, 3:4] = 0.0
            dummy_grid[3, :, :, 3:4] = 1.0
            return dummy_grid

    def get_block_grid(self):
        # Get the block probabilities over the entire grid
        xyz_grid = torch.argwhere(torch.ones(tuple(self.gridSize), device=self.device)).float()
        # Normalize xyz_grid to [-1, 1]
        xyz_grid = (xyz_grid / (self.gridSize - 1)) * 2 - 1

        block_probs = self._get_block_probs(xyz_grid)

        blocks_disc = torch.argmax(block_probs, dim=1).view(*self.gridSize)
        blocks_ec = torch.zeros_like(blocks_disc)
        blocks_ec[:] = AIR

        for block_name in block_names_to_enum:
            block_ec = block_names_to_enum[block_name]
            blocks_ec[blocks_disc == block_enum_to_idx[block_ec]] = block_ec


        blocks_alpha_grid = self.compute_densityfeature(xyz_grid, mask=None)
        blocks_alpha_grid = blocks_alpha_grid.view(*self.gridSize)
            
        return blocks_ec, blocks_alpha_grid
        
    def _get_block_probs(self, xyz_sampled):
        # print(f"Max x, y, z: {torch.max(xyz_sampled, dim=0)}")
        # print(f"Min x, y, z: {torch.min(xyz_sampled, dim=0)}")
        if self.DUMMY_BLOCK_GRID:
            block_probs = self._compute_blockprobs_from_dummygrid(xyz_sampled)
        else:
            # plane + line basis
            coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
            coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

            plane_coef_point,line_coef_point = [],[]
            for idx_plane in range(len(self.app_plane)):
                plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                    align_corners=True, mode='nearest').view(-1, *xyz_sampled.shape[:1]))
                line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                                align_corners=True, mode='nearest').view(-1, *xyz_sampled.shape[:1]))
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

            pts = plane_coef_point * line_coef_point
            # Take probabilities over block types
            block_probs = self.block_probs_mat(pts.T)

            block_probs = F.gumbel_softmax(block_probs, tau=0.1, dim=1, hard=True)

        return block_probs

    def compute_appdensityfeature(self, xyz_sampled):
        device = xyz_sampled.device

        block_probs = self._get_block_probs(xyz_sampled)
        # Convert xyz from normalized in [-1, 1] to world coordinates
        abs_xyz_sampled = (xyz_sampled + 1) / 2 * self.gridSize
        # Assert that the sampled points are inside the volume
        # assert torch.all(abs_xyz_sampled >= 0) and torch.all(abs_xyz_sampled < self.gridSize)
        # if not torch.all(abs_xyz_sampled >= 0) and torch.all(abs_xyz_sampled < self.gridSize):
        #     TT()
        # Get the coordinates of the nearest point on the grid to each sampled point
        grid_indices = torch.meshgrid([torch.arange(0, self.gridSize[0]), torch.arange(0, self.gridSize[1]), torch.arange(0, self.gridSize[2])])
        grid_indices = torch.stack(grid_indices, dim=0).to(device)
        # nearest_vertices_xyz_sampled = F.grid_sample(grid_indices[None,...].float(), xyz_sampled[None, :, None, None, :], mode='nearest')[0, :, :, 0, 0]
        # nearest_vertices_xyz_sampled = rearrange(nearest_vertices_xyz_sampled, 'xyz b -> b xyz')
        nearest_vertices_xyz_sampled = torch.round(abs_xyz_sampled).long()
        xyz_sampled_offsets = abs_xyz_sampled - nearest_vertices_xyz_sampled
        
        out = torch.zeros(xyz_sampled.shape[0], 4).to(device)
        # Get list of 3D blocks from the sampled points
        for i in range(len(self.blocks)):
            block = self.blocks[i][None]
            # out_i = block_probs[:, i] * self.blocks[i][..., :, :, :, None]
            # xyz_offsets_i_idxs = torch.where(block_probs[:, i] > 0)[0]

            if self.FULL_BLOCK_TEXTURE:
                # Assert xyz are in [-.5, .5]
                assert (xyz_sampled_offsets.min() >= -.5) and (xyz_sampled_offsets.max() <= .5)
                # print(xyz_sampled_offsets.min(), xyz_sampled_offsets.max())
                # Get point at relative position inside the block.
                pts_i = F.grid_sample(block, xyz_sampled_offsets[None, :, None, None, :] * 2, mode='nearest',
                    align_corners=False)[0, :, :, 0, 0]
                pts_i = rearrange(pts_i, 'xyz b -> b xyz')

            else:
                # Lazy debug: take point at corner of the block.
                pts_i = torch.tile(block[0, :, 0, 0, 0], (len(out), 1))

            pts_i = pts_i * block_probs[:, i, None]

            out += pts_i
        return out


    # @torch.no_grad()
    # def getDenseAlpha(self,gridSize=None):
    #     gridSize = self.gridSize if gridSize is None else gridSize

    #     # More samples to compensate for "scaling up" of blocks in underlying grid.
    #     gridSize = np.array(gridSize) * 16

    #     samples = torch.stack(torch.meshgrid(
    #         torch.linspace(0, 1, gridSize[0]),
    #         torch.linspace(0, 1, gridSize[1]),
    #         torch.linspace(0, 1, gridSize[2]),
    #     ), -1).to(self.device)
    #     dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

    #     # dense_xyz = dense_xyz
    #     # print(self.stepSize, self.distance_scale*self.aabbDiag)
    #     alpha = torch.zeros_like(dense_xyz[...,0])
    #     for i in range(gridSize[0]):
    #         alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
    #     return alpha, dense_xyz
        

    # def compute_alpha(self, xyz_locs, length=1):

    #     if self.alphaMask is not None:
    #         alphas = self.alphaMask.sample_alpha(xyz_locs)
    #         alpha_mask = alphas > 0
    #     else:
    #         alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

    #     sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

    #     if alpha_mask.any():
    #         xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
    #         sigma_feature = self.compute_densityfeature(xyz_sampled, None)
    #         # sigma_feature = self.compute_densityfeature(xyz_sampled)
    #         validsigma = self.feature2density(sigma_feature)
    #         sigma[alpha_mask] = validsigma
        

    #     alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

    #     return alpha 