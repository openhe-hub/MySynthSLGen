import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_channels, heatmap_channels):
        super(Encoder, self).__init__()

        self.n_channels = n_channels
        self.heatmap_channels = heatmap_channels

        self.inc = DoubleConv(n_channels, 32)
        self.maxpool = nn.MaxPool2d(2)
        self.down1 = Down(self.heatmap_channels+32, 64)
        self.down2 = Down(self.heatmap_channels+64, 128)
        self.down3 = Down(self.heatmap_channels+128, 256)
        self.down4 = Down(self.heatmap_channels+256, 512)
        self.down5 = Down(self.heatmap_channels+512, 1024)
        self.fc_down1 = nn.Linear(4*4*(self.heatmap_channels+1024), 8192)
        self.fc_down2 = nn.Linear(8192, 200)

        ## adding 1*1 conb to reduce the number of heatmap channels for the fuly connected layer
        self.one_conv = nn.Conv2d(heatmap_channels, 10, kernel_size=(1,1), padding='same')
        self.fc_heatmap = nn.Linear(256*256*10, 100)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self, i, base_heatmap, heatmap):
        h = torch.cat([base_heatmap, heatmap],dim=1)
        x = torch.cat([i,h],dim=1)

        x1 = self.inc(x)
        x1 = self.maxpool(x1) #torch.Size([1, 64, 64, 64])
        h = heatmap
        h = F.interpolate(h, size=(x1.shape[2],x1.shape[3]))
        x1 = torch.cat([x1,h],dim=1)

        x2 = self.down1(x1) #torch.Size([1, 128, 32, 32])
        h = heatmap
        h = F.interpolate(h, size=(x2.shape[2],x2.shape[3]))
        x2 = torch.cat([x2,h],dim=1)

        x3 = self.down2(x2) #torch.Size([1, 256, 16, 16])
        h = heatmap
        h = F.interpolate(h, size=(x3.shape[2],x3.shape[3]))
        x3 = torch.cat([x3,h],dim=1)

        x4 = self.down3(x3) #torch.Size([1, 512, 8, 8])
        h = heatmap
        h = F.interpolate(h, size=(x4.shape[2],x4.shape[3]))
        x4 = torch.cat([x4,h],dim=1)

        x5 = self.down4(x4) #torch.Size([1, 1024, 4, 4])
        h = heatmap
        h = F.interpolate(h, size=(x5.shape[2],x5.shape[3]))
        x5 = torch.cat([x5,h],dim=1)

        x6 = self.down5(x5)
        h = heatmap
        h = F.interpolate(h, size=(x6.shape[2],x6.shape[3]))
        x6 = torch.cat([x6,h],dim=1)

        x6_flatten = x6.view(-1, self.num_flat_features(x6))
        v = self.fc_down1(x6_flatten)
        v = self.fc_down2(v) #[1,200]

        h_vector = heatmap
        h_vector = self.one_conv(h_vector)
        h_vector = h_vector.view(-1,self.num_flat_features(h_vector))
        h_vector = self.fc_heatmap(h_vector) #[1,100]

        v = torch.cat([v,h_vector],dim=1)
        v = v.unsqueeze(2).unsqueeze(2)

        return v,x1,x2,x3,x4,x5,x6


class Decoder(nn.Module):
    def __init__(self, n_channels, heatmap_channels):
        super(Decoder, self).__init__()
        
        self.heatmap_channels = heatmap_channels
        
        self.up1 = Up_init(300, 1024)
        self.up2 = Up(1024+1024+self.heatmap_channels, 512)
        self.up3 = Up(512+512+self.heatmap_channels, 256)
        self.up4 = Up(256+256+self.heatmap_channels, 128)
        self.up5 = Up(128+128+self.heatmap_channels, 64)
        self.up6 = Up(64+64+self.heatmap_channels, 32)
        self.up7 = Up(32+32+self.heatmap_channels, 3, bilinear=True)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self,v,x1,x2,x3,x4,x5,x6):
        x1_up = self.up1(v) #torch.Size([1, 1024, 4, 4])
        x1_up = torch.cat([x1_up, x6], dim=1) #torch.Size([1, 2058, 4, 4])

        x2_up = self.up2(x1_up) #torch.Size([1, 512, 8, 8])
        x2_up = torch.cat([x2_up,x5], dim=1) #torch.Size([1, 1034, 8, 8])

        x3_up = self.up3(x2_up) #torch.Size([1, 256, 16, 16])
        x3_up = torch.cat([x3_up,x4], dim=1) #torch.Size([1, 522, 16, 16])

        x4_up = self.up4(x3_up) #torch.Size([1, 128, 32, 32])
        x4_up = torch.cat([x4_up,x3], dim=1) #torch.Size([1, 266, 32, 32])

        x5_up = self.up5(x4_up) #torch.Size([1, 64, 64, 64])
        x5_up = torch.cat([x5_up,x2], dim=1) #torch.Size([1, 138, 64, 64])

        x6_up = self.up6(x5_up) #torch.Size([1, 3, 128, 128])
        x6_up = torch.cat([x6_up,x1], dim=1)

        logits = self.up7(x6_up)
        return logits


class UNetGenerator(nn.Module):
    def __init__(self, n_channels, heatmap_channels, bilinear=False, device_encoder=None, device_decoder=None):
        super(UNetGenerator, self).__init__()

        self.heatmap_channels = heatmap_channels
        self.encoder = Encoder(n_channels, heatmap_channels=self.heatmap_channels)
        self.decoder = Decoder(n_channels, heatmap_channels=self.heatmap_channels)
        self.device_encoder = device_encoder
        self.device_decoder = device_decoder

        if device_encoder is not None:
            self.encoder = self.encoder.to(device_encoder)
        if device_decoder is not None:
            self.decoder = self.decoder.to(device_decoder)

    def forward(self, i, base_heatmap, heatmap):
        if self.device_encoder is not None:
            v,x1,x2,x3,x4,x5,x6 = self.encoder(i.to(self.device_encoder), base_heatmap.to(self.device_encoder), heatmap.to(self.device_encoder))
        else:
            v,x1,x2,x3,x4,x5,x6 = self.encoder(i, base_heatmap, heatmap)

        if self.device_decoder is not None:
            v = v.to(self.device_decoder)
            x1 = x1.to(self.device_decoder)
            x2 = x2.to(self.device_decoder)
            x3 = x3.to(self.device_decoder)
            x4 = x4.to(self.device_decoder)
            x5 = x5.to(self.device_decoder)
            x6 = x6.to(self.device_decoder)

        self.decoder = self.decoder.to(self.device_decoder)

        return self.decoder(v,x1,x2,x3,x4,x5,x6)


class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with double conv and maxpool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_pool = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv_pool(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class Up_init(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=4, stride=4)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__=="__main__":
    import numpy as np
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    n_kp = 96
    model = UNetGenerator(2*n_kp+3, heatmap_channels=n_kp, device_encoder=device, device_decoder=device).to(device)
    dummy_img = torch.rand((batch_size, 3, 256, 256)).to(device)
    dummy_s_pose = torch.rand((batch_size, n_kp, 256, 256)).to(device)
    dummy_t_pose = torch.rand((batch_size, n_kp, 256, 256)).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_img, dummy_s_pose, dummy_t_pose)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_img, dummy_s_pose, dummy_t_pose)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)