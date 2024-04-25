import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import open3d as o3d

def load_bin_point_cloud(file_path):
    """Load a point cloud from a .bin file."""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    return points

def point_cloud_to_depth_image(points, img_height=64, img_width=1024, fov_up=3.0, fov_down=-25.0):
    """Convert point cloud to a depth image."""
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_up) + abs(fov_down)  # total field of view across vertical axis

    # Create depth image
    depth_img = np.zeros((img_height, img_width), dtype=np.float32)

    # Calculate angles and project points
    scan_radius = np.linalg.norm(points, 2, axis=1)
    scan_angles = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))

    # Map the angles to depth image rows
    row_indices = ((scan_angles - fov_down) / fov * img_height).astype(np.int32)
    row_indices = np.clip(row_indices, 0, img_height - 1)

    # Map x, y coordinates to depth image columns
    col_indices = (-np.arctan2(points[:, 0], points[:, 1]) / np.pi * 0.5 + 0.5) * img_width
    col_indices = np.clip(col_indices.astype(np.int32), 0, img_width - 1)

    # Assign depths
    for i in range(len(scan_radius)):
        if depth_img[row_indices[i], col_indices[i]] == 0 or depth_img[row_indices[i], col_indices[i]] > scan_radius[i]:
            depth_img[row_indices[i], col_indices[i]] = scan_radius[i]

    return depth_img

def save_depth_image(depth_img, output_file):
    """Save depth image to a PNG file."""
    fig = plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1])  # Increased ratio for image to colorbar

    # Create axes
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # Show image
    im = ax0.imshow(depth_img, cmap='hot')
    ax0.axis('off')  # No axis for image

    # Create colorbar
    plt.colorbar(im, cax=ax1)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    file_path = '/home/ps/huichenchen/mmdetection3d/scripts/000000.bin'
    output_file = '/home/ps/huichenchen/mmdetection3d/results2/0422_paint_range3.png'

    points = load_bin_point_cloud(file_path)
    depth_image = point_cloud_to_depth_image(points)
    save_depth_image(depth_image, output_file)

    print(f"Depth image saved to {output_file}")

if __name__ == "__main__":
    main()
