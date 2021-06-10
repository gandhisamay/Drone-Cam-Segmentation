import torch.nn as nn 
import torch 
#Define the CNN block now
#Defined as per the U-net Structure 
#Made some modifications too to the original structure
class DoubleCNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            out_channels
        )
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            out_channels
        )
        self.act2 = nn.ReLU() 
    def forward(self,x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return out

class UpConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        ) 

    def forward(self,x,skip_connection):
        out = self.tconv(x)
        if out.shape != skip_connection.shape:
            out = TF.resize(out ,size=skip_connection.shape[2:])
        out = torch.cat([skip_connection,out],axis = 1)
        return out

class Bottom(nn.Module):
    def __init__(self,channel=[128,256]):
        super().__init__()
        self.channel=channel
        self.conv1 = nn.Conv2d(
            in_channels=self.channel[0],
            out_channels=self.channel[1],
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            self.channel[1]
        )
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.channel[1],
            out_channels=self.channel[1],
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            self.channel[1]
        )
        self.act2 = nn.ReLU()
        
        self.bottom = nn.Sequential(
            self.conv1,
            self.bn1,
            self.act1,
            self.conv2,
            self.bn2,
            self.act2
        )
    def forward(self,x):
#         out = self.act1(self.bn1(self.conv1(x)))
#         print("1:{}".format(out.shape))
#         out = self.act2(self.bn2(self.conv2(out)))
#         print("2:{}".format(out.shape))
        return self.bottom(x)

class Unet(nn.Module):
    def __init__(self,num_classes,filters=[16,32,64,128],input_channels=3):
        super().__init__()
        self.contract = nn.ModuleList()
        self.expand   = nn.ModuleList()                      #64 - #128 - #256 - #512 - #1024 -#512
        self.filters  = filters
        self.input_channels = input_channels
        self.num_classes = num_classes 
        
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        
        for filters in self.filters:
            self.contract.append(
                DoubleCNNBlock(
                    in_channels=input_channels,
                    out_channels=filters
                )
            )
            input_channels = filters
            
        for filters in reversed(self.filters):
            self.expand.append(
                UpConv(
                    in_channels=filters*2,
                    out_channels=filters
                )
            )
            self.expand.append(
                DoubleCNNBlock(
                    in_channels=filters*2,
                    out_channels=filters
                )
            )
            
        self.final = nn.Conv2d(
                    in_channels=self.filters[0],
                    out_channels=num_classes,
                    kernel_size=3,
                    padding=1,
                    stride=1
                    )
            
    def forward(self,x):
        skip_connections = []
        
        for downs in self.contract:
            out = downs(x)
            skip_connections.append(out)
            out   = self.pool(out)
            x = out
        
        bottom = Bottom()
        bottom.to(DEVICE)
        y = bottom(x)
        
        for idx in range(0,len(self.expand),2):
            skip_connection = skip_connections[len(skip_connections)-idx//2-1]
            y = self.expand[idx](y,skip_connection)
            y = self.expand[idx+1](y)
            
        return self.final(y)

model = Unet(num_classes=8)
model.to(DEVICE)
