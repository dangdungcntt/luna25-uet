from .model_3d import *
from copy import deepcopy
from torch import nn

class Unit3DTranspose(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        activation="relu",
        padding=(0, 0, 0),
        use_bias=False,
        use_bn=True,
        **kwargs,
    ):
        super(Unit3DTranspose, self).__init__()

        self.use_bn = use_bn
        self.activation_type = activation

        self.deconv3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
            **kwargs,
        )

        if use_bn:
            self.batch3d = nn.BatchNorm3d(out_channels)

        if activation == "relu":
            self.activation = F.relu
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.feat_processor = nn.Conv3d(out_channels, out_channels, 1, 1)
        self.out_conv = nn.Conv3d(out_channels * 2, out_channels, 1, 1)

    def forward(self, x, feat):
        x = self.deconv3d(x)
        
        feat = self.feat_processor(feat)
        x = torch.cat([x, feat], dim=1)
        x = self.out_conv(x)

        if self.use_bn:
            x = self.batch3d(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
    
class I3D_UNet(nn.Module):
    def __init__(
            self,
            num_classes,
            input_channels,
            out_channels,
            dropout_prob=0,
            pre_trained=True,
            freeze_bn=True,
            target_size=(64, 64, 64)
        ):
        super().__init__()
        self.target_size = target_size
        self.encoder = I3D(num_classes=num_classes, 
                           input_channels=input_channels, 
                           dropout_prob=dropout_prob, 
                           pre_trained=pre_trained, 
                           freeze_bn=freeze_bn)
        self.classifier = deepcopy(self.encoder.classifier)
        self.encoder.classifier = nn.Identity()

        self.up_1 = Unit3DTranspose(
            in_channels=1024,
            out_channels=832,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2)
        )
        self.up_2 = Unit3DTranspose(
            in_channels=832,
            out_channels=480,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            output_padding=(1, 1, 1)
        )
        self.up_3 = Unit3DTranspose(
            in_channels=480,
            out_channels=192,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            output_padding=(0, 1, 1)

        )
        self.up_4 = Unit3DTranspose(
            in_channels=192,
            out_channels=64,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            output_padding=(0, 1, 1)
        )
        # self.up_5 = nn.ConvTranspose3d(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=(7, 7, 7),
        #     stride=(2, 2, 2),
        #     padding=(3, 3, 3),
        #     output_padding=(1, 1, 1)
        # )
        self.segmentor = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x, extract_features=True)

        up_feature = features[0]
        up_feature = self.up_1(up_feature, features[1])
        up_feature = self.up_2(up_feature, features[2])
        up_feature = self.up_3(up_feature, features[3])
        up_feature = self.up_4(up_feature, features[4])

        # up_feature = self.up_5(up_feature)
        up_feature = F.interpolate(up_feature, size=self.target_size, mode='trilinear', align_corners=False)
        
        seg_logit = self.segmentor(up_feature)
        # with torch.no_grad():
        #     seg_logit_rescale = F.interpolate(seg_logit.detach().clone(), size=features[0].shape[-3:], mode="trilinear", align_corners=True)
        #     seg_logit_rescale = torch.sigmoid(seg_logit_rescale)
        
        class_logit = features[0] # * seg_logit_rescale
        class_logit = self.classifier(class_logit)

        return class_logit, seg_logit

if __name__ == "__main__":
    unet = I3D_UNet(num_classes=1, input_channels=3, out_channels=1)
    image = torch.randn(4, 3, 64, 128, 128)
    logit = unet(image)
    print(logit[0].shape)
    print(logit[1].shape)