import trimesh
import numpy as np
import pyvista as pv
import open3d as o3d
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import os
import argparse
from scipy.spatial import cKDTree

def compute_rficp_speeds(field_values, k=1.0, c_target=0.1, lambda_eff=1.0, R=1.0, sigma=0.5, 
                         v_min=0.5, v_max=5.0):
    """
    Compute speeds based on Radiant-Field-Informed Coverage Planning (RFICP) formula.
    
    Parameters:
        field_values: Semantic field values A(p) at each point (e.g., aridity values)
        k: Proportional coefficient for desired coverage
        c_target: Target coverage level
        lambda_eff: Coverage efficiency constant
        R: Radius of influence circle
        sigma: Gaussian kernel standard deviation
        v_min: Minimum velocity constraint
        v_max: Maximum velocity constraint
    
    Returns:
        speeds: Array of speeds at each point
    """
    # Compute CDF value for Gaussian integral
    f_R_sigma = norm.cdf(R / sigma)
    gaussian_integral = 2 * f_R_sigma - 1
    
    # Initialize speeds array
    speeds = np.zeros_like(field_values)
    
    for i, A_p in enumerate(field_values):
        # Check validity conditions
        if k * A_p <= c_target:
            # If field value is too low, use maximum speed
            speeds[i] = v_max
        else:
            # Compute the argument for logarithm
            log_arg = 1 - (k * A_p - c_target) / gaussian_integral
            
            if log_arg <= 0 or log_arg >= 1:
                # Handle edge cases
                if log_arg <= 0:
                    speeds[i] = v_min  # Need maximum dwell time (minimum speed)
                else:
                    speeds[i] = v_max  # Need minimum dwell time (maximum speed)
            else:
                # Compute speed using RFICP formula
                # v(p) = -λ / ln(log_arg)
                dwell_time = -1.0 / lambda_eff * np.log(log_arg)
                
                # Avoid division by zero or negative dwell time
                if dwell_time <= 0:
                    speeds[i] = v_max
                else:
                    speeds[i] = 1.0 / dwell_time
    
    # Clip speeds to velocity constraints
    speeds = np.clip(speeds, v_min, v_max)
    
    # Apply smoothing for acceleration constraints
    if len(speeds) > 5:
        try:
            # Use Savitzky-Golay filter for smooth acceleration
            speeds = savgol_filter(speeds, window_length=min(5, len(speeds)), polyorder=2)
            speeds = np.clip(speeds, v_min, v_max)
        except:
            pass
    
    return speeds

def rotate_pointcloud_to_xy_plane(points):
    """
    Rotate point cloud from xz plane to xy plane (90 degrees around x-axis)
    This swaps Y and Z coordinates: (x, y, z) -> (x, z, -y)
    """
    print("Rotating point cloud from XZ plane to XY plane...")
    rotated_points = points.copy()
    # Swap Y and Z coordinates with proper sign
    rotated_points[:, 1] = points[:, 2]  # new Y = old Z
    rotated_points[:, 2] = points[:, 1]  # new Z = old Y
    return rotated_points

def load_pointcloud(file_path, auto_rotate=True):
    """
    Load point cloud file (supports PCD, PLY, XYZ, NPZ formats)
    Returns (points (N,3), colors (N,3) float in [0,1]) or (None, None) on failure.
    
    Parameters:
        file_path: path to point cloud file
        auto_rotate: if True, automatically detect and rotate from XZ to XY plane
    """
    print(f"Loading point cloud file: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pcd' or ext == '.ply':
            # Use Open3D primarily for PCD/PLY
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
            else:
                # try fallback via trimesh for potential vertex colors in some PLY variants
                if ext == '.ply':
                    try:
                        mesh_or_pc = trimesh.load(file_path, process=False)
                        if hasattr(mesh_or_pc, 'vertices'):
                            points = np.asarray(mesh_or_pc.vertices)
                        if hasattr(mesh_or_pc, 'colors') and mesh_or_pc.colors is not None:
                            colors = np.asarray(mesh_or_pc.colors)
                            # trimesh stores 0..255 typically
                            if colors.max() > 1.0:
                                colors = colors[:, :3] / 255.0
                        elif hasattr(mesh_or_pc, 'visual') and getattr(mesh_or_pc.visual, 'vertex_colors', None) is not None:
                            vc = np.asarray(mesh_or_pc.visual.vertex_colors)
                            colors = vc[:, :3] / 255.0 if vc.max() > 1.0 else vc[:, :3]
                        else:
                            colors = np.ones((len(points), 3)) * 0.5
                    except Exception:
                        colors = np.ones((len(points), 3)) * 0.5
                else:
                    colors = np.ones((len(points), 3)) * 0.5

        elif ext == '.xyz':
            data = np.loadtxt(file_path)
            if data.ndim == 1 and data.size >= 3:
                # single-row file
                data = data.reshape(1, -1)
            points = data[:, :3]
            if data.shape[1] >= 6:
                colors = data[:, 3:6].astype(float)
                if colors.max() > 1.0:
                    colors = colors / 255.0
            else:
                colors = np.ones((len(points), 3)) * 0.5
                print("  Warning: XYZ file has no color information, using default gray")

        elif ext in ['.npz', '.npy']:
            data = np.load(file_path, allow_pickle=True)
            if ext == '.npz':
                if 'points' in data:
                    points = data['points']
                elif 'xyz' in data:
                    points = data['xyz']
                else:
                    # try first array
                    keys = list(data.keys())
                    points = data[keys[0]]
                if 'colors' in data:
                    colors = data['colors']
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                else:
                    colors = np.ones((len(points), 3)) * 0.5
                    print("  Warning: NPZ file has no color information, using default gray")
            else:
                points = data
                if points.ndim == 1:
                    points = points.reshape(-1, 3)
                colors = np.ones((len(points), 3)) * 0.5
                print("  Warning: NPY file only contains coordinates, using default gray")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Basic validation
        if points is None or len(points) == 0:
            raise ValueError("Point cloud is empty or could not be read")

        points = np.asarray(points, dtype=float)
        colors = np.asarray(colors, dtype=float)
        if colors.shape[0] != points.shape[0]:
            # try to broadcast a single color
            if colors.size == 3:
                colors = np.tile(colors.reshape(1, 3), (len(points), 1))
            else:
                print("  Warning: Color array size mismatch, using default gray")
                colors = np.ones((len(points), 3)) * 0.5

        # Clamp color range and ensure in [0,1]
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

        print(f"Successfully loaded point cloud:")
        print(f"  - Number of points: {len(points):,}")
        print(f"  - Original coordinate ranges:")
        print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"  - Color range: [{colors.min():.3f}, {colors.max():.3f}]")
        
        # Auto-detect if point cloud needs rotation (terrain mainly in XZ plane)
        if auto_rotate:
            y_range = points[:, 1].max() - points[:, 1].min()
            z_range = points[:, 2].max() - points[:, 2].min()
            x_range = points[:, 0].max() - points[:, 0].min()
            
            # If Y range is much smaller than X and Z ranges, likely terrain is in XZ plane
            if y_range < 0.3 * min(x_range, z_range):
                print("\nDetected terrain in XZ plane, rotating to XY plane...")
                points = rotate_pointcloud_to_xy_plane(points)
                print(f"  - Rotated coordinate ranges:")
                print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
                print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
                print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

        return points, colors

    except Exception as e:
        print(f"Failed to load point cloud file: {e}")
        return None, None

def downsample_pointcloud(points, colors, target_num_points=25000, method='random'):
    """
    Downsample point cloud to target number of points
    """
    if len(points) <= target_num_points:
        print(f"Point count ({len(points)}) <= target ({target_num_points}), skipping downsample")
        return points, colors

    print(f"Downsampling point cloud: {len(points):,} -> {target_num_points:,} (method: {method})")

    if method == 'random':
        indices = np.random.choice(len(points), target_num_points, replace=False)
        return points[indices], colors[indices]

    elif method == 'voxel':
        # Voxel downsample using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        bbox = points.max(axis=0) - points.min(axis=0)
        # protect against zero bbox dimension
        bbox[bbox == 0] = 1e-6
        # heuristic voxel size: cube root of volume / target_count
        voxel_size = np.power(np.prod(bbox) / max(1, target_num_points), 1/3.0)
        voxel_size = max(voxel_size, 1e-4)  # avoid zero
        # slight multiplier to lean toward keeping slightly fewer points
        voxel_size *= 1.2

        down = pcd.voxel_down_sample(voxel_size)
        dp = np.asarray(down.points)
        dc = np.asarray(down.colors) if down.has_colors() else np.ones((len(dp), 3)) * 0.5
        print(f"  Voxel downsample produced {len(dp):,} points (voxel_size={voxel_size:.6f})")
        return dp, dc

    elif method == 'uniform':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        every_k_points = max(1, len(points) // target_num_points)
        down = pcd.uniform_down_sample(every_k_points)
        return np.asarray(down.points), np.asarray(down.colors)

    else:
        indices = np.random.choice(len(points), target_num_points, replace=False)
        return points[indices], colors[indices]

def reconstruct_surface_poisson(points, depth=9):
    """
    Poisson reconstruction (Open3D)
    """
    print(f"Reconstructing surface (Poisson depth={depth})...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals - use neighbor search param to be safe
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        # fallback: estimate with defaults
        pcd.estimate_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    bbox = pcd.get_axis_aligned_bounding_box()
    try:
        mesh = mesh.crop(bbox)
    except Exception:
        pass

    print(f"Surface reconstruction complete: {len(np.asarray(mesh.vertices))} vertices, {len(np.asarray(mesh.triangles))} triangles")
    return mesh

def generate_smooth_aridity_map(x_range, y_range, resolution):
    """
    Generate smoothly varying aridity map on xy plane.
    Returns x (len nx), y (len ny), aridity_map shape (nx, ny) properly aligned for interpolation.
    """
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)  # X and Y are shape (ny, nx)
    
    np.random.seed(42)

    # Generate multi-scale noise for more natural patterns
    large_scale = gaussian_filter(np.random.rand(resolution, resolution), sigma=20)
    medium_scale = gaussian_filter(np.random.rand(resolution, resolution), sigma=10) * 0.5
    small_scale = gaussian_filter(np.random.rand(resolution, resolution), sigma=5) * 0.25

    aridity_map = large_scale + medium_scale + small_scale

    # Add circular arid regions
    for _ in range(3):
        cx = np.random.uniform(*x_range)
        cy = np.random.uniform(*y_range)
        radius = np.random.uniform(0.2, 0.4) * (x_range[1] - x_range[0])
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        aridity_map += np.exp(-(dist ** 2) / (2 * (radius/2) ** 2)) * 0.5

    # Normalize to 0..1
    aridity_map = (aridity_map - aridity_map.min()) / (aridity_map.max() - aridity_map.min() + 1e-12)
    aridity_map = np.power(aridity_map, 1.5)

    # Transpose so that aridity_map[i,j] corresponds to (x[i], y[j])
    # This is critical for correct RegularGridInterpolator behavior
    aridity_map = aridity_map.T
    
    return x, y, aridity_map

def interpolate_aridity_map(x, y, aridity_map):
    """
    Create RegularGridInterpolator expecting points as (N,2) with order (x, y).
    aridity_map must have shape (len(x), len(y))
    """
    interp_func = RegularGridInterpolator((x, y), aridity_map, method='linear', bounds_error=False, fill_value=0.0)
    return interp_func

def generate_zigzag_path(terrain_points, x_range, y_range, line_spacing, point_spacing, altitude, 
                        aridity_interp, rficp_params=None):
    """
    Generate zigzag path on xy plane and adjust z values and speed based on terrain height and aridity.
    Robust to empty rows / NaN interpolations by falling back to nearest-neighbor.
    
    Parameters:
        terrain_points: Terrain point coordinates
        x_range: X axis range tuple
        y_range: Y axis range tuple
        line_spacing: Spacing between trajectory lines
        point_spacing: Spacing between points on each line
        altitude: Path altitude above terrain
        aridity_interp: Aridity interpolation function
        rficp_params: Dictionary of RFICP parameters (k, c_target, lambda_eff, R, sigma, v_min, v_max)
    """
    # Set default RFICP parameters if not provided
    if rficp_params is None:
        rficp_params = {
            'k': 1.0,
            'c_target': 0.1,
            'lambda_eff': 1.0,
            'R': 1.0,
            'sigma': 0.5,
            'v_min': 0.5,
            'v_max': 5.0
        }
    # Build interpolators
    lin_interp = LinearNDInterpolator(terrain_points[:, :2], terrain_points[:, 2])
    
    # Build KDTree for fast nearest-neighbor fallback
    kd = cKDTree(terrain_points[:, :2])

    y_start, y_end = y_range
    x_start, x_end = x_range

    # Ensure ranges are ordered
    if y_end < y_start:
        y_start, y_end = y_end, y_start
    if x_end < x_start:
        x_start, x_end = x_end, x_start

    # Create inclusive y lines
    if line_spacing <= 0:
        line_spacing = (y_end - y_start) / 10.0 if (y_end - y_start) != 0 else 0.1
    
    y_lines = np.arange(y_start, y_end + line_spacing, line_spacing)

    path_points = []
    speeds = []
    total_points = 0

    for i, y in enumerate(y_lines):
        # Create x sequence depending on row parity (zigzag)
        if i % 2 == 0:
            x_line = np.arange(x_start, x_end + point_spacing, point_spacing)
        else:
            x_line = np.arange(x_end, x_start - point_spacing, -point_spacing)

        if x_line.size == 0:
            continue

        line_points = np.column_stack((x_line, np.full_like(x_line, y)))
        z_line = lin_interp(line_points)
        
        # Handle NaN values with nearest neighbor fallback
        valid_idx = ~np.isnan(z_line)
        if not np.any(valid_idx):
            # All NaN, use nearest neighbor for all
            dists, idxs = kd.query(line_points, k=1)
            z_line = terrain_points[idxs, 2]
            valid_idx = np.ones(len(z_line), dtype=bool)
        elif np.any(~valid_idx):
            # Some NaN, fill them with nearest neighbor
            nan_mask = np.isnan(z_line)
            dists, idxs = kd.query(line_points[nan_mask], k=1)
            z_line[nan_mask] = terrain_points[idxs, 2]
            valid_idx = np.ones(len(z_line), dtype=bool)

        line_points = line_points[valid_idx]
        z_line = z_line[valid_idx] + altitude

        # Smooth z values
        if len(z_line) > 5:
            try:
                z_line_smoothed = savgol_filter(z_line, window_length=5, polyorder=2)
            except:
                z_line_smoothed = z_line
        else:
            z_line_smoothed = z_line

        # Compute field-informed speeds based on RFICP formula
        aridity_values = aridity_interp(line_points)
        speeds_line = compute_rficp_speeds(aridity_values, **rficp_params)

        line_points = np.column_stack((line_points, z_line_smoothed))
        path_points.append(line_points)
        speeds.append(speeds_line)
        total_points += len(line_points)

        # Add transitions between rows
        if i < len(y_lines) - 1:
            start_point = line_points[-1]
            next_y = y_lines[i+1]
            next_x = x_start if (i+1) % 2 == 0 else x_end
            next_line_point = np.array([next_x, next_y])
            
            # Interpolate z for next point
            next_z = lin_interp(next_line_point.reshape(1, -1))
            if np.isnan(next_z):
                dists, idxs = kd.query(next_line_point.reshape(1, -1), k=1)
                next_z = terrain_points[idxs[0], 2]
            else:
                next_z = next_z[0]
            next_z += altitude

            # Create transition
            num_transition_points = 10
            transition_x = np.linspace(start_point[0], next_line_point[0], num_transition_points)
            transition_y = np.linspace(start_point[1], next_line_point[1], num_transition_points)
            transition_xy = np.column_stack((transition_x, transition_y))
            
            transition_z = lin_interp(transition_xy)
            nan_mask = np.isnan(transition_z)
            if np.any(nan_mask):
                dists, idxs = kd.query(transition_xy[nan_mask], k=1)
                transition_z[nan_mask] = terrain_points[idxs, 2]
            
            transition_z = transition_z + altitude
            
            # Smooth transition z
            if len(transition_z) > 5:
                try:
                    transition_z_smoothed = savgol_filter(transition_z, window_length=5, polyorder=2)
                except:
                    transition_z_smoothed = transition_z
            else:
                transition_z_smoothed = transition_z
            
            transition_points = np.column_stack((transition_x, transition_y, transition_z_smoothed))
            path_points.append(transition_points)
            
            aridity_values = aridity_interp(transition_xy)
            speeds_line = compute_rficp_speeds(aridity_values, **rficp_params)
            speeds.append(speeds_line)
            total_points += len(transition_points)

    if len(path_points) == 0:
        print("Cannot generate valid trajectory within terrain boundaries")
        return None, None

    # Flatten lists into single arrays
    path = np.vstack(path_points)
    speeds = np.concatenate(speeds)

    # Ensure lengths match
    if path.shape[0] != speeds.shape[0]:
        nmin = min(path.shape[0], speeds.shape[0])
        path = path[:nmin]
        speeds = speeds[:nmin]

    # Smooth entire path z values
    if len(path) > 11:
        try:
            z_smoothed = savgol_filter(path[:, 2], window_length=11, polyorder=3)
            path[:, 2] = z_smoothed
        except:
            pass

    return path, speeds

def setup_aesthetic_plotter(title):
    plotter = pv.Plotter(title=title, window_size=[1200, 800])
    plotter.set_background('white')
    plotter.add_light(pv.Light(position=(5, 5, 5), light_type='scene light'))
    plotter.add_light(pv.Light(position=(-5, -5, 5), intensity=0.3))
    return plotter

def visualize_terrain(points, colors):
    plotter = setup_aesthetic_plotter('Terrain Map')
    point_cloud = pv.PolyData(points)
    # PyVista expects scalars length equal to n_points when using rgb True
    rgba = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    point_cloud["RGBA"] = rgba
    plotter.add_mesh(point_cloud, scalars='RGBA', rgb=True, point_size=3, render_points_as_spheres=True)
    plotter.camera_position = 'yz'
    plotter.add_axes(line_width=3, color='black')
    plotter.show()

def visualize_reconstructed_surface(reconstructed_mesh):
    plotter = setup_aesthetic_plotter('Surface Reconstruction')
    vertices = np.asarray(reconstructed_mesh.vertices)
    faces = np.asarray(reconstructed_mesh.triangles)
    faces_pv = np.hstack((np.full((len(faces), 1), 3, dtype=np.int64), faces)).astype(np.int64).flatten()
    tri_mesh = pv.PolyData(vertices, faces_pv)
    z_values = vertices[:, 2]
    tri_mesh["elevation"] = z_values
    plotter.add_mesh(tri_mesh, scalars="elevation", cmap='terrain', show_scalar_bar=False)
    plotter.camera_position = 'yz'
    plotter.add_axes(line_width=3, color='black')
    plotter.show()

def visualize_aridity_map(aridity_map, x, y):
    plotter = setup_aesthetic_plotter('Aridity Map')
    X, Y = np.meshgrid(x, y)  # X shape (ny,nx), Y shape (ny,nx)
    Z = np.zeros_like(X) + 0.01
    grid = pv.StructuredGrid(X, Y, Z)
    # Need to transpose aridity_map back for visualization since meshgrid gives (ny,nx)
    grid["aridity"] = aridity_map.T.flatten(order='C')
    plotter.add_mesh(grid, scalars="aridity", cmap='RdYlBu_r', opacity=0.9, show_scalar_bar=False)
    contours = grid.contour(isosurfaces=8, scalars="aridity")
    plotter.add_mesh(contours, color='#333333', line_width=2, opacity=0.6)
    plotter.camera_position = 'yz'
    plotter.add_axes(line_width=3, color='black')
    plotter.show()

def visualize_trajectory(path):
    plotter = setup_aesthetic_plotter('Trajectory Coverage')
    trajectory = pv.PolyData(path)
    plotter.add_mesh(trajectory, color='#ff4444', point_size=8, render_points_as_spheres=True)
    n_points = len(path)
    if n_points > 1:
        lines = np.hstack([np.full((n_points - 1, 1), 2, dtype=np.int64),
                          np.vstack((np.arange(n_points - 1, dtype=np.int64), np.arange(1, n_points, dtype=np.int64))).T]).flatten()
        path_line = pv.PolyData(path)
        path_line.lines = lines
        plotter.add_mesh(path_line, color='#ff8800', line_width=2, opacity=0.6)
    plotter.camera_position = 'yz'
    plotter.add_axes(line_width=3, color='black')
    plotter.show()

def visualize_trajectory_speed(path, speeds):
    plotter = setup_aesthetic_plotter('Speed Visualization')
    trajectory = pv.PolyData(path)
    trajectory["speed"] = speeds
    plotter.add_mesh(trajectory, scalars="speed", cmap='plasma', point_size=10, render_points_as_spheres=True, show_scalar_bar=False)
    n_points = len(path)
    if n_points > 1:
        lines = np.hstack([np.full((n_points - 1, 1), 2, dtype=np.int64),
                          np.vstack((np.arange(n_points - 1, dtype=np.int64), np.arange(1, n_points, dtype=np.int64))).T]).flatten()
        path_line = pv.PolyData(path)
        path_line.lines = lines
        path_line["speed"] = speeds
        plotter.add_mesh(path_line, scalars="speed", cmap='plasma', line_width=3, opacity=0.4, show_scalar_bar=False)
    plotter.camera_position = 'yz'
    plotter.add_axes(line_width=3, color='black')
    plotter.show()

def visualize_combined(points, colors, reconstructed_mesh, path, speeds, aridity_map, x, y):
    plotter = setup_aesthetic_plotter('Complete Visualization')
    # terrain points
    pc = pv.PolyData(points)
    rgba = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    pc["RGBA"] = rgba
    plotter.add_mesh(pc, scalars='RGBA', rgb=True, point_size=2, render_points_as_spheres=True, opacity=0.6)
    # reconstructed mesh
    vertices = np.asarray(reconstructed_mesh.vertices)
    faces = np.asarray(reconstructed_mesh.triangles)
    faces_pv = np.hstack((np.full((len(faces), 1), 3, dtype=np.int64), faces)).astype(np.int64).flatten()
    tri_mesh = pv.PolyData(vertices, faces_pv)
    tri_mesh["elevation"] = vertices[:, 2]
    plotter.add_mesh(tri_mesh, scalars="elevation", cmap='terrain', opacity=0.4, show_scalar_bar=False)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) + 0.02
    coverage_surface = pv.StructuredGrid(X, Y, Z)
    coverage_surface["aridity"] = aridity_map.T.flatten(order='C')
    plotter.add_mesh(coverage_surface, scalars="aridity", cmap='YlOrRd', opacity=0.5, show_scalar_bar=False)
    # trajectory
    trajectory = pv.PolyData(path)
    trajectory["speed"] = speeds
    plotter.add_mesh(trajectory, scalars="speed", cmap='plasma', point_size=8, render_points_as_spheres=True, show_scalar_bar=False)
    n_points = len(path)
    if n_points > 1:
        lines = np.hstack([np.full((n_points - 1, 1), 2, dtype=np.int64),
                          np.vstack((np.arange(n_points - 1, dtype=np.int64), np.arange(1, n_points, dtype=np.int64))).T]).flatten()
        path_line = pv.PolyData(path)
        path_line.lines = lines
        path_line["speed"] = speeds
        plotter.add_mesh(path_line, scalars="speed", cmap='plasma', line_width=2, opacity=0.3, show_scalar_bar=False)

    plotter.camera_position = 'yz'
    plotter.add_axes(line_width=4, color='black', ambient=0.5)
    plotter.add_text('Comprehensive Terrain Analysis System', position='upper_left', font_size=12, color='black')
    plotter.show()  

def main():
    parser = argparse.ArgumentParser(description='Terrain Analysis and Path Planning from Point Cloud')
    parser.add_argument('input', help='Input point cloud file (PCD, PLY, XYZ, NPZ, NPY)')
    parser.add_argument('--downsample', type=int, default=25000)
    parser.add_argument('--downsample-method', choices=['random', 'voxel', 'uniform'], default='random')
    parser.add_argument('--poisson-depth', type=int, default=9)
    parser.add_argument('--line-spacing', type=float, default=0.05)
    parser.add_argument('--point-spacing', type=float, default=0.1)
    parser.add_argument('--altitude', type=float, default=0.2)
    parser.add_argument('--aridity-resolution', type=int, default=100)
    parser.add_argument('--no-rotate', action='store_true', help='Disable automatic rotation from XZ to XY plane')
    
    # RFICP parameters
    parser.add_argument('--rficp-k', type=float, default=1.0, help='Proportional coefficient for desired coverage')
    parser.add_argument('--rficp-c-target', type=float, default=0.1, help='Target coverage level')
    parser.add_argument('--rficp-lambda', type=float, default=1.0, help='Coverage efficiency constant')
    parser.add_argument('--rficp-R', type=float, default=1.0, help='Radius of influence circle')
    parser.add_argument('--rficp-sigma', type=float, default=0.5, help='Gaussian kernel standard deviation')
    parser.add_argument('--v-min', type=float, default=0.5, help='Minimum velocity constraint')
    parser.add_argument('--v-max', type=float, default=5.0, help='Maximum velocity constraint')
    
    parser.add_argument('--visualize', choices=['all', 'terrain', 'surface', 'aridity',
                                               'trajectory', 'speed', 'combined'],
                        default='all')
    args = parser.parse_args()

    points, colors = load_pointcloud(args.input, auto_rotate=not args.no_rotate)
    if points is None:
        print("Error: Could not load point cloud file")
        return

    if len(points) > args.downsample:
        points, colors = downsample_pointcloud(points, colors, target_num_points=args.downsample, method=args.downsample_method)

    reconstructed_mesh = reconstruct_surface_poisson(points, depth=args.poisson_depth)

    x_range = (float(points[:, 0].min()), float(points[:, 0].max()))
    y_range = (float(points[:, 1].min()), float(points[:, 1].max()))
    x, y, aridity_map = generate_smooth_aridity_map(x_range, y_range, args.aridity_resolution)
    aridity_interp = interpolate_aridity_map(x, y, aridity_map)

    # Prepare RFICP parameters
    rficp_params = {
        'k': args.rficp_k,
        'c_target': args.rficp_c_target,
        'lambda_eff': args.rficp_lambda,
        'R': args.rficp_R,
        'sigma': args.rficp_sigma,
        'v_min': args.v_min,
        'v_max': args.v_max
    }
    
    print("\nRFICP Parameters:")
    print(f"  - k (proportional coefficient): {rficp_params['k']}")
    print(f"  - C_target (target coverage): {rficp_params['c_target']}")
    print(f"  - λ (coverage efficiency): {rficp_params['lambda_eff']}")
    print(f"  - R (influence radius): {rficp_params['R']}")
    print(f"  - σ (Gaussian std dev): {rficp_params['sigma']}")
    print(f"  - Velocity constraints: [{rficp_params['v_min']}, {rficp_params['v_max']}]")

    path, speeds = generate_zigzag_path(points, x_range, y_range, args.line_spacing, 
                                        args.point_spacing, args.altitude, aridity_interp, rficp_params)
    if path is None:
        print("Error: Could not generate trajectory path")
        return

    print(f"\nVisualization mode: {args.visualize}")
    print(f"Trajectory generated successfully:")
    print(f"  - Trajectory points: {len(path)}")
    print(f"  - Speed range: {speeds.min():.2f} - {speeds.max():.2f} m/s")
    print(f"  - Coverage area: {(x_range[1]-x_range[0])*(y_range[1]-y_range[0]):.2f} m²")
    
    if args.visualize == 'all':
        visualize_terrain(points, colors)
        visualize_reconstructed_surface(reconstructed_mesh)
        visualize_aridity_map(aridity_map, x, y)
        visualize_trajectory(path)
        visualize_trajectory_speed(path, speeds)
        visualize_combined(points, colors, reconstructed_mesh, path, speeds, aridity_map, x, y)
    elif args.visualize == 'terrain':
        visualize_terrain(points, colors)
    elif args.visualize == 'surface':
        visualize_reconstructed_surface(reconstructed_mesh)
    elif args.visualize == 'aridity':
        visualize_aridity_map(aridity_map, x, y)
    elif args.visualize == 'trajectory':
        visualize_trajectory(path)
    elif args.visualize == 'speed':
        visualize_trajectory_speed(path, speeds)
    elif args.visualize == 'combined':
        visualize_combined(points, colors, reconstructed_mesh, path, speeds, aridity_map, x, y)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()