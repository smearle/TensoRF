import os
from pdb import set_trace as TT

import numpy as np
from PIL import Image
import torch

from clients.python.src.main.proto.minecraft_pb2_grpc import *
from clients.python.src.main.proto.minecraft_pb2 import *


# block_names = ["stone", "dirt", "brick", "glass", "sand", "snow"]
block_names_to_enum = {
    "stone": STONE, 
    "dirt": DIRT, 
    "brick": BRICK_BLOCK, 
    # "sand": SAND, 
    "clay": CLAY,
    "snow": SNOW, 
    "glazed_terracotta_light_blue": LIGHT_BLUE_GLAZED_TERRACOTTA,
    "glazed_terracotta_yellow": YELLOW_GLAZED_TERRACOTTA,
    "redstone_block": REDSTONE_BLOCK, 
    "gold_block": GOLD_BLOCK, 
    "iron_block": IRON_BLOCK, 
    "diamond_block": DIAMOND_BLOCK, 
    "emerald_block": EMERALD_BLOCK,
    # "concrete_powder_yellow": CONCRETE_POWDER_YELLOW,
}
block_names = list(block_names_to_enum.keys())
block_enum_to_idx = {v: i for i, v in enumerate(block_names_to_enum.values())}
# block_names = ["stone", "dirt", "brick", "sand"]
# block_names = ["stone", "dirt"]
# block_names = ["black", "white"]
TEXTURE_PATH = "InventivetalentDev-minecraft-assets-3360cc4/assets/minecraft/textures/blocks/"

# Read all textures
# block_names = os.listdir(TEXTURE_PATH)
# block_names = [bn for bn in block_names if not (bn.endswith('.mcmeta'))]

#TODO: Assemble blocks made from multiple texture files.

def init_mc_blocks(FULL_BLOCK_TEXTURE):
    # Include empty block (will be last).
    blocks = torch.zeros(len(block_names_to_enum) + 1, 4, 16, 16, 16)

    # Transparent and white by default
    blocks[:, 0, :, :, :] = 0.0
    blocks[:, 1:, :, :, :] = 1.0

    # Named blocks only.
    # blocks = torch.zeros(len(block_names), 4, 16, 16, 16)

    i = 0
    for block_name in block_names_to_enum:

        block_texture = torch.Tensor(np.array(Image.open(TEXTURE_PATH + f"{block_name}.png"))).permute(2, 0, 1) / 255
        # try:
        #     block_texture = torch.Tensor(np.array(Image.open(TEXTURE_PATH + f"{block_name}"))).permute(2, 0, 1) / 255
        # except:
        #     continue

        # Check if correct dimensions
        if len(block_texture.shape) != 3 or block_texture.shape[1] != 16 or block_texture.shape[2] != 16:
            print(f"Warning: {block_name} is not 16x16")
            continue

        # Add alpha channel.
        if block_texture.shape[0] == 3:
            block_texture = torch.cat([block_texture, torch.ones(1, 16, 16)], dim=0)
        # print(f"Block texture {block_name}, shape: {block_texture.shape}")

        ### BEGIN DEBUG (comment out this chunk, uncomment line above to use Minecraft blocks. Bur first, let's try
        # to make it select black/white blocks based on eponymous text prompts.)
        # if i == 0:
        #     block_texture = torch.zeros(4, 16, 16).float()
        #     block_texture[-1] = 1
        # else:
        #     block_texture = torch.ones(4, 16, 16).float()
        #     block_texture[-1] = 1
        ### END DEBUG

        block_texture[-1] *= 100
        y = torch.zeros_like(block_texture)
        index = torch.LongTensor([3, 0, 1, 2])
        y = block_texture[index]
        block_texture = y
        # Lazy hack to make blocks look more solid(?)
        # blocks[i+1, :, 0:16, 0:16, 0:16] = torch.tile(block_texture.unsqueeze(1), (1, 16, 1, 1))

        # blocks[i+1, :, 1, 1:15, 1:15] = block_texture[:, 1:15, 1:15]
        # blocks[i+1, :, 2, 2:14, 2:14] = block_texture[:, 2:14, 2:14]
        # blocks[i+1, :, 3, 3:13, 3:13] = block_texture[:, 3:13, 3:13]
        # blocks[i+1, :, 4, 4:12, 4:12] = block_texture[:, 4:12, 4:12]
        # blocks[i+1, :, 5, 5:11, 5:11] = block_texture[:, 5:11, 5:11]
        # blocks[i+1, :, 6, 6:10, 6:10] = block_texture[:, 6:10, 6:10]
        # blocks[i+1, :, 7, 6:10, 6:10] = block_texture[:, 7:11, 7:11]

        if FULL_BLOCK_TEXTURE == 1:
            # Blocks are hollow
            n_reps=1

        elif FULL_BLOCK_TEXTURE == 2:
            # Weird hack to make blocks look more solid...
            n_reps = 1 if block_name == "glass" else 8

        # DEBUG solid blocks
        # blocks[i] = torch.tile(block_texture.unsqueeze(1), (1, 16, 1, 1))
        # blocks[i, :, 0] = block_texture
        # END DEBUG

        #TODO: Correctly orient block faces.

        for j in range(n_reps):
            blocks[i, :, j:16-j, j, j:16-j] = block_texture[:, j:16-j, j:16-j]
            blocks[i, :, j:16-j, 15-j, j:16-j] = block_texture[:, j:16-j, j:16-j]
            blocks[i, :, j:16-j, j:16-j, j] = block_texture[:, j:16-j, j:16-j]
            blocks[i, :, j:16-j, j:16-j, 15-j] = block_texture[:, j:16-j, j:16-j]
            blocks[i, :, j, j:16-j, j:16-j] = block_texture[:, j:16-j, j:16-j]
            blocks[i, :, 15-j, j:16-j, j:16-j] = block_texture[:, j:16-j, j:16-j]

        # blocks[i, 0, 1:15, 1:15, 1:15] = 100

        # blocks[i+1, :, 0, 0:16, 0:16] = block_texture
        # blocks[i+1, :, 15, 0:16, 0:16] = block_texture
        # blocks[i+1, :, 0:16, 0, 0:16] = block_texture
        # blocks[i+1, :, 0:16, 15, 0:16] = block_texture
        # blocks[i+1, :, 0:16, 0:16, 0] = block_texture
        # blocks[i+1, :, 0:16, 0:16, 15] = block_texture
        i += 1

    # Make the empty block white (though it is supposedly transparent...)
    blocks[-1, 1:] = 1

    print(f"Blocks shape: {blocks.shape}")

    return blocks

