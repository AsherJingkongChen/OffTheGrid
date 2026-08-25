import tempfile
import subprocess
import os
import cv2
import json
import argparse
import torch
import numpy as np
import json
from einops import rearrange
import matplotlib.pyplot as plt
import torchvision

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def images_names_from_dir(directory, filter_str=None):
    try:
        image_names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return []
    if filter_str is not None:
        image_names = [i for i in image_names if filter_str in i]
    image_names = [i for i in image_names if ('.jpg' in i or '.png' in i or '.JPG' in i or '.PNG' in i)]
    image_names = [os.path.join(directory, x) for x in image_names]
    return image_names

def save_depth_map(depth_map, path, vmin, vmax):
    depth_map = depth_map.detach().cpu()

    if len(depth_map.shape)==3:
        depth_map = depth_map[0]
    elif len(depth_map.shape)==4:
        depth_map = rearrange(depth_map,'N C H W -> C H (N W)')[0]
    
    depth_map = (depth_map-vmin)/vmax
    depths_map_visu = plt.cm.magma_r(depth_map)[..., :3] 
    depths_map_visu = torch.from_numpy(depths_map_visu).permute(2, 0, 1).float()
    torchvision.utils.save_image(depths_map_visu, path)

def save_camera_poses(extrinsics, intrinsics, paths, file_path):
    num_frames = len(paths)
    cam_data = []
    
    for i in range(num_frames):
        frame_dict = {}
        
        frame_dict["file_path"] = paths[i]
        #frame_dict["colmap_id"] = int(path[-6:-4])
        matrix = extrinsics[i].tolist()
        frame_dict["extrinsics"] = matrix
        fx = torch.mean(intrinsics[i,0,0]).detach().cpu().item()
        fy = torch.mean(intrinsics[i,1,1]).detach().cpu().item()
        frame_dict["w"]= 518
        frame_dict["h"]= 518
        frame_dict["fx"]= fx
        frame_dict["fy"]= fy
        frame_dict["cx"]= 518//2#int(2.39 * 518 * 0.5)
        frame_dict["cy"]= 518//2
        cam_data.append(frame_dict)
        

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cam_data, f, ensure_ascii=False, indent=2)



def tensor_to_mp4(tensor, filename="output.mp4", fps=30, codec="libx264", crf=20, powerpoint_compatible=True):
    """
    Converts a PyTorch tensor containing a batch of images to an MP4 video file,
    without saving individual frames.  It uses a temporary file for intermediate
    storage, and then encodes that to the final MP4.

    Args:
        tensor (torch.Tensor): A PyTorch tensor of shape (batch_size, channels, height, width)
            or (batch_size, height, width, channels).  The images should typically
            have values in the range [0, 1] or [0, 255].  If the values are float,
            they are assumed to be in [0,1] and are converted to 8-bit unsigned
            integers. If the values are integers, it is checked that they are in
            the range [0, 255].
        filename (str, optional): The name of the output MP4 video file.
            Defaults to "output.mp4".
        fps (int, optional): Frames per second of the output video. Defaults to 30.
        codec (str, optional):  The video codec to use.  "libx264" (default, for
            CPU encoding) is a good general-purpose codec.  Other options
            include "libx265" (for potentially better compression, but slower
            encoding) and "h264_nvenc" (for hardware-accelerated encoding on
            NVIDIA GPUs).
        crf (int, optional): The Constant Rate Factor (CRF) for the output video.
            Lower values mean higher quality, but larger file sizes.  A good
            range is 17-28 for libx264; the default of 20 is a reasonable
            trade-off.  For libx265, use a slightly higher value.
        powerpoint_compatible (bool, optional):  If True, encodes the video in a
            format that is more compatible with PowerPoint (yuv420p pixel format).
            Defaults to True.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor must be a torch.Tensor")

    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must be 4D (batch_size, C, H, W) or (batch_size, H, W, C)")

    # Ensure the tensor is on the CPU and in the correct format.
    tensor = tensor.cpu().detach()

    # Handle different channel dimensions and value ranges
    if tensor.shape[1] == 3 or tensor.shape[1] == 4:
        # (B, C, H, W) to (B, H, W, C)
        tensor = tensor.permute(0, 2, 3, 1)
    elif tensor.shape[3] != 3 and tensor.shape[3] != 4:
        raise ValueError("The tensor should have 3 or 4 channels (RGB or RGBA) in the last or second dimension.")

    # Convert to numpy array and ensure correct data type
    images = tensor.numpy()

    if images.dtype == np.float32 or images.dtype == np.float64:
        images = (images * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    elif images.dtype == np.uint8:
        pass # Already in correct format
    else:
        raise TypeError(f"Input tensor should have dtype float32, float64 or uint8, but got {images.dtype}")
    
    if images.max() > 255 or images.min() < 0:
        raise ValueError("Input tensor has values outside the range [0, 255].")


    batch_size, height, width, channels = images.shape

    # Determine the correct pixel format for FFmpeg
    if powerpoint_compatible:
        pixel_format = "yuv420p"  # More compatible with PowerPoint
    elif channels == 3:
        pixel_format = "rgb24"
    elif channels == 4:
        pixel_format = "rgba"
    else:
        raise ValueError("Unsupported number of channels: {}".format(channels))

    # Create a temporary file to store the raw video data
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=True) as temp_file:
        temp_file_name = temp_file.name

        # Construct the FFmpeg command
        command = [
            "ffmpeg",
            "-y",  # Overwrite existing file without asking
            "-f", "rawvideo",
            "-pix_fmt", pixel_format,
            "-s:v", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",  # Read from stdin
            "-an",  # No audio
            "-vcodec", codec,
            "-crf", str(crf),  # Quality setting
            filename
        ]

        # Open a pipe to FFmpeg's stdin
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            # Write the raw video data to FFmpeg's stdin
            for image in images:
                # Convert RGB to YUV if necessary
                if powerpoint_compatible and channels == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420)
                elif powerpoint_compatible and channels == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2YUV_I420)

                process.stdin.write(image.tobytes())
            process.stdin.close()
            process.wait() # Wait for the process to finish.

            if process.returncode != 0:
                _, stderr = process.communicate()  # Get stderr
                raise RuntimeError(f"FFmpeg error (code {process.returncode}): {stderr.decode('utf-8')}")

        except Exception as e:
            print(f"Error during video conversion: {e}")
            # Clean up if there's an error.
            if os.path.exists(filename):
                os.remove(filename)
            raise
        finally:
            # Ensure the process is cleaned up
            if process.stdin:
                process.stdin.close()
            if process.stderr:
                process.stderr.close()
            if process:
                process.kill()
        print(f"Successfully saved video to {filename}")