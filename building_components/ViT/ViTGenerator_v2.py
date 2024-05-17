import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x):
        x = self.patch_embedding(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x


class Down(nn.Module):
    def __init__(self, input_hw, embed_dim):
        super(Down, self).__init__()
        self.input_hw = input_hw
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(4*embed_dim)
        self.reduction = nn.Linear(4*embed_dim, 2*embed_dim, bias=False)

    def forward(self, x):
        H, W = self.input_hw
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, self.embed_dim)
        x0 = x[:, 0::2, 0::2, :]                # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]                # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]                # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]                # (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)     # (B, H/2, W/2, C*4)
        x = x.view(B, -1, 4 * C)                # (B, L/4, C*4)

        x = self.norm(x)
        x = self.reduction(x)                   # (B, L/4, C*2)

        return x
    

class Up(nn.Module):
    def __init__(self, input_hw, embed_dim):
        super(Up, self).__init__()
        self.input_hw = input_hw
        self.embed_dim = embed_dim
        
        self.up = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim//2, kernel_size=2, stride=2)

    def forward(self, x):
        H, W = self.input_hw
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.transpose(1, 2)   # (B, C, L)
        x = x.view(B, C, H, W)  # (B, C, H, W)

        x = self.up(x)          # (B, C/2, H*2, W*2)
        x = x.flatten(2)        # (B, C/2, L*4)
        x = x.transpose(1, 2)   # (B, L*4, C/2)

        return x


class UnPatch(nn.Module):
    def __init__(self, input_hw, embed_dim, patch_size):
        super(UnPatch, self).__init__()
        self.input_hw = input_hw
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=3, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        H, W = self.input_hw
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.transpose(1, 2)   # (B, C, L)
        x = x.view(B, C, H, W)  # (B, C, H, W)
        x = self.conv(x)
    
        return x
    

class ViTGenerator(nn.Module):
    def __init__(self, n_channels):
        super(ViTGenerator, self).__init__()
        
        self.n_channels = n_channels
        embed_dim = 32  # Embedding dimension
        heads = 8  # Number of attention heads
        mlp_dim = 1024
    
        self.patch_embedding = PatchEmbedding(image_size=256, patch_size=16, in_channels=self.n_channels, embed_dim=embed_dim)      # (B, 256, 32)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches, embed_dim))                       # (1, 256, 32)
        self.tf_1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, heads, mlp_dim), num_layers=2)                      # (B, 256, 32)
        self.down_1 = Down((16, 16), embed_dim)                                                                                     # (B, 64, 64)
        self.tf_2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim*2, heads, mlp_dim), num_layers=2)                    # (B, 64, 64)        
        self.down_2 = Down((8, 8), embed_dim*2)                                                                                     # (B, 16, 128)
        self.tf_3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim*4, heads, mlp_dim), num_layers=2)                    # (B, 16, 128)        
        self.down_3 = Down((4, 4), embed_dim*4)                                                                                     # (B, 4, 256)
        self.tf_4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim*8, heads, mlp_dim), num_layers=2)                    # (B, 4, 256)
        self.up_3 = Up((2, 2), embed_dim*8)                                                                                         # (B, 16, 128)
        self.up_2 = Up((4, 4), embed_dim*4)                                                                                         # (B, 64, 64)
        self.up_1 = Up((8, 8), embed_dim*2)                                                                                         # (B, 256, 32)
        self.unpatch = UnPatch((16, 16), embed_dim, patch_size=16)

    def forward(self, base_image, base_heatmap, heatmaps):
        x = torch.cat([base_image, base_heatmap, heatmaps], dim=1)
        x = self.patch_embedding(x)
        x += self.positional_embedding
        x_1 = self.tf_1(x)
        x_1 = self.down_1(x_1)
        x_2 = self.tf_2(x_1)
        x_2 = self.down_2(x_2)
        x_3 = self.tf_3(x_2)
        x_3 = self.down_3(x_3)
        x = self.tf_4(x_3)
        x = self.up_3(x + x_3)
        x = self.up_2(x + x_2)
        x = self.up_1(x + x_1)
        x = self.unpatch(x)
        
        return x



if __name__=="__main__":
    model = ViTGenerator(503)

    img = torch.rand((4, 3, 256, 256))
    heatmaps = torch.rand((4, 500, 256, 256))
    
    print(model(img, heatmaps).shape)  # Output shape: (1, num_classes)
