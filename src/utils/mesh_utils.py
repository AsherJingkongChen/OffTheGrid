import copy
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from pathlib import Path

def clean_mesh(mesh, cleaning_threshold=10000, scale=1,
               keep_larger_island=True):
    """
    Clean the mesh by removing small clusters of triangles and save it.

    Returns: None
    """

    thres = cleaning_threshold / scale

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    cleaned_mesh = copy.deepcopy(mesh)

    if keep_larger_island:
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
    else:
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < thres

    cleaned_mesh.remove_triangles_by_mask(triangles_to_remove)
    cleaned_mesh.remove_unreferenced_vertices()
    return cleaned_mesh


def tsdf_fusion(rgbs, depths, alphas, extrinsics, intrinsics, n_images, width,
                height, output_path, quantile=0.9, cuda_device='CUDA:0',
                verbose=False):
    """
    Exports a 3D Mesh based on depthmaps.
    rgbs: A list of n_images rgb tensors (3, W, H) 
    depths: A list of n_images depthmap tensors (1, W, H) 
    alphas: A list of n_images depthmap tensors (W, H)
    extrinsics: (n_images, 3, 4) tensor
    intrinsics: (n_images, 3, 3) tensor
    n_images: int, 
    width: int
    height: int
    output_path: str, the path to save the mesh
    quantile: float, calculated the max_depth, default:0.9
    cuda_device='CUDA:0', CPU or CUDA for tsdf fusion, defaut: CUDA:0[CPU:0]
    verbose: Boolean, prints more messages, default: False
    """

    o3d_device = o3d.core.Device(cuda_device)
    # TODO
    # Add these values to arguments
    voxel_size = 0.002
    alpha_thres = 0.5
    depth_max_m = 1.0

    if verbose:
        print(f'Intrinsics shape {intrinsics.shape}')
        print(f'Extrinsics shape {extrinsics.shape}')

    vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000,
            device=o3d_device)

    q = torch.tensor([0.25, 0.5, 0.75]).cuda()
    tmp_vector = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(extrinsics[0].device)
    with torch.no_grad():
        for v in range(n_images):
            depth = depths[v][0:1]
            alpha = alphas[v].unsqueeze(0)
            rgb = rgbs[v]
            if verbose:
                print(f'Depth quantile {depth.quantile(q)}')
                print(f'Alpha {alpha.quantile(q)}')
                print(f'RGB {rgb.quantile(q)}')

            depth[(alpha < alpha_thres)] = 0
            max_depth = torch.quantile(depth, quantile)

            W = width
            H = height
            fx = float(intrinsics[v][0, 0])
            fy = float(intrinsics[v][1, 1])
            cx = float(intrinsics[v][0, 2])
            cy = float(intrinsics[v][1, 2])
            if verbose:
                print(f'fx: {fx} fy: {fy} cx: {cx} cy:{cy}')

            if verbose:
                print(f'Intrinsic {intrinsics[v]}')

            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H,
                                                          cx=cx, cy=cy,
                                                          fx=fx, fy=fy)

            # w2c
            if extrinsics[v].shape == (3,4):
                if verbose:
                    print(' Extrinsics shape is 3,4, add one more row')
                tmp_extrinsics = torch.cat([extrinsics[v], 
                                            tmp_vector.unsqueeze(0)], dim=0)
            else:
                if verbose:
                    print(' Extrinsics shape is 4, 4. No changes.')
                tmp_extrinsics = extrinsics[v]

            extrinsic_np = np.asarray((tmp_extrinsics.float()).cpu().numpy()).astype(np.float64)

            rgb_np = np.asarray(rgb.permute(1, 2, 0).cpu().numpy(), order="C")
            o3d_color = o3d.t.geometry.Image(rgb_np)
            o3d_color = o3d_color.to(o3d_device)

            depth = depth * depth_max_m
            if verbose:
                print("depth_m min/max:", float(depth.min()),
                      float(depth.max()))

            depth_np = np.asarray(depth.permute(1, 2, 0).cpu().numpy(),
                                  order="C")
            o3d_depth = o3d.t.geometry.Image(depth_np)
            o3d_depth = o3d_depth.to(o3d_device)

            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix,
                                        o3d.core.Dtype.Float64)
            extrinsic = o3d.core.Tensor(extrinsic_np, o3d.core.Dtype.Float64)

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                o3d_depth, intrinsic, extrinsic, 1.0, max_depth)

            vbg.integrate(frustum_block_coords, o3d_depth, o3d_color,
                          intrinsic, intrinsic, extrinsic, 1.0,
                          max_depth.item())

    mesh = vbg.extract_triangle_mesh().to_legacy()
    o3d.io.write_triangle_mesh(f"{Path(output_path)}/tsdf.ply", mesh)
    cleaned_mesh = clean_mesh(mesh)

    # write mesh
    o3d.io.write_triangle_mesh(f"{Path(output_path)}/clean_tsdf.ply",
                               cleaned_mesh)
