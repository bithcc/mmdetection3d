import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_bin_point_cloud(file_path):
    """Load a point cloud from a .bin file."""
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 4))[:, :3]
    return points

def create_mesh(points):
    """Create a mesh using the Ball Pivoting algorithm."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Compute the mesh
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist   # Set radius for meshing
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    
    return mesh

def plot_and_save_mesh(mesh, output_file):
    """Plot the mesh and save it as a PNG file."""
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles, vertices[:, 2], shade=True, color=[0.5, 0.5, 0.5, 0.5])
    
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_axis_off()
    
    plt.savefig(output_file, dpi=300)
    plt.close()

def main():
    file_path = '/home/ps/huichenchen/mmdetection3d/scripts/000000.bin'
    output_file = '/home/ps/huichenchen/mmdetection3d/results2/0422_paint_mesh.png'

    points = load_bin_point_cloud(file_path)
    mesh = create_mesh(points)
    plot_and_save_mesh(mesh, output_file)

    print(f"Mesh visualization saved to {output_file}")

if __name__ == "__main__":
    main()
