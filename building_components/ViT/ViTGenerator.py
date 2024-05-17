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


class UnPatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super(UnPatchEmbedding, self).__init__()
        num_layers = int(math.log2(patch_size))
        model = []
        for i in range(num_layers):
            model += [nn.ConvTranspose2d(in_channels=embed_dim//(2**i), out_channels=embed_dim//(2**(i+1)), kernel_size=2, stride=2)]
        model += [nn.Conv2d(in_channels=embed_dim//(2**num_layers), out_channels=3, kernel_size=1)]
        self.model = nn.Sequential(*model)

    def forward(self, embedded_patches):
        batch_size, num_patches, embed_dim = embedded_patches.size()

        unflatten_patches = embedded_patches.transpose(1, 2)  # (batch_size, embed_dim, num_patches)
        unflatten_patches = unflatten_patches.view(
            batch_size, embed_dim, int(num_patches**0.5), int(num_patches**0.5)
            )  # (batch_size, embed_dim, num_patches_h, num_patches_w)

        image = self.model(unflatten_patches)  # (batch_size, in_channels, image_size, image_size)
        
        return image


class ViTGenerator(nn.Module):
    def __init__(self, n_channels):
        super(ViTGenerator, self).__init__()
        self.image_size = 256
        self.patch_size = 16
        self.n_channels = n_channels
        self.embed_dim = 512  # Embedding dimension
        self.depth = 6  # Depth of the transformer
        self.heads = 8  # Number of attention heads
        self.mlp_dim = 1024  # Dimension of the MLP hidden layer

        self.patch_embedding = PatchEmbedding(self.image_size, self.patch_size, self.n_channels, self.embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches, self.embed_dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.embed_dim, self.heads, self.mlp_dim), self.depth)
        self.un_patch_embedding = UnPatchEmbedding(self.patch_size, self.embed_dim)


    def forward(self, base_image, heatmaps):
        x = torch.cat([base_image, heatmaps],dim=1)
        x = self.patch_embedding(x)
        x += self.positional_embedding
        x = self.transformer(x)
        x = self.un_patch_embedding(x)

        return x



if __name__=="__main__":
    model = ViTGenerator(503)

    img = torch.rand((4, 3, 256, 256))
    heatmaps = torch.rand((4, 500, 256, 256))
    
    print(model(img, heatmaps).shape)  # Output shape: (1, num_classes)
