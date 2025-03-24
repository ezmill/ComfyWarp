import torch
import numpy as np
from PIL import Image
import os

# Import the original warp_flow function from flow_utils.py
from .flow_utils import warp_flow

def custom_apply_warp(current_frame, flow, padding=0, pad_mode='reflect'):
    """
    Apply optical flow to warp a frame with customizable padding mode.
    
    Parameters:
        current_frame (torch.Tensor): The frame to warp
        flow (torch.Tensor): The optical flow field
        padding (float): Padding percentage (0.0 to 1.0)
        pad_mode (str): Padding mode, one of:
            - 'reflect': Reflect at boundary (default)
            - 'constant': Pad with zeros
            - 'edge': Pad with edge values
            - 'wrap': Wrap around (periodic padding)
    
    Returns:
        torch.Tensor: Warped frame
    """
    pad_pct = padding
    flow21 = flow 
    current_frame = current_frame[0]
    
    if pad_pct > 0:
        pad = int(max(flow21.shape) * pad_pct)
    else:
        pad = 0
        
    # Always use constant padding for flow
    flow21 = np.pad(flow21.numpy(), pad_width=((pad,pad),(pad,pad),(0,0)), mode='constant')
    
    # Use the specified padding mode for the frame
    current_frame = np.pad(
        current_frame.numpy().transpose(1,0,2), 
        pad_width=((pad,pad),(pad,pad),(0,0)),
        mode=pad_mode
    )
    
    # Apply warping
    warped_frame = warp_flow(current_frame, flow21).transpose(1,0,2)
    
    # Crop to original size
    if pad > 0:
        warped_frame = warped_frame[pad:warped_frame.shape[0]-pad, pad:warped_frame.shape[1]-pad, :]
    
    warped_frame = torch.from_numpy(warped_frame).cpu()
    
    return warped_frame[None, ]


class CustomWarpFrame:
    """
    Enhanced version of WarpFrame that supports different padding modes.
    """
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "previous_frame": ("IMAGE",), 
                        "flow": ("BACKWARD_FLOW",),
                        "padding": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "padding_mode": (["reflect", "constant", "edge", "wrap"], {"default": "reflect"}),
                    }
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "warp"

    def warp(self, previous_frame, flow, padding, padding_mode):
        warped_frame = custom_apply_warp(
            previous_frame, 
            flow, 
            padding=padding, 
            pad_mode=padding_mode
        )
        
        return (warped_frame, )

# Add this to your node registrations
NODE_CLASS_MAPPINGS = {
    "CustomWarpFrame": CustomWarpFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomWarpFrame": "WarpFrame (Custom)",
}