from .decoder import Decoder
from os.path import join, dirname, exists
from torch import Tensor, IntTensor, FloatTensor, inference_mode, load, save
import torch
import PIL.Image
import numpy as np

import inspect

int8_iinfo = torch.iinfo(torch.int8)
int8_range = int8_iinfo.max-int8_iinfo.min
int8_half_range = int8_range / 2


class FastLatentToImage:
    """
    A custom node for converting latents to images

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method.
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    decode(s) -> tuple || None:
        The entry point method.
    """
    def __init__(self):

        #get current directory
        class_file_path = inspect.getfile(self.__class__)
        #join with the directory name
        weights_path = join(dirname(class_file_path), "decoder_sdxl.pt")
        #weights_path="D:/img/comfy/ComfyUI_windows_portable/ComfyUI/custom_nodes/fastDecoderdecoder_sdxl.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Decoder()
        if exists(weights_path):
            self.model.load_state_dict(load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "decode"

    OUTPUT_NODE = False

    CATEGORY = "Custom"

    def decode(self, latent):
        latent=latent['samples']
        latent=latent.permute(0,2,3,1)        
        latent=latent.to(self.device)
        #predict
        predicts: Tensor = self.model(latent)
        # convert to correct type
        predicts = predicts
        predicts = predicts + 1
        predicts = predicts * int8_half_range
        predicts: Tensor = predicts.round().clamp(0, 255).to(dtype=torch.uint8).cpu()
        #comfy wants float32
        torch_image=(predicts.to(torch.float32)/255)
        return (torch_image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FastLatentToImage": FastLatentToImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FastLatentToImage": "Fast Latent To Image Node"
}
