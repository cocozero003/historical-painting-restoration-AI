
try:
    import torch, torch.nn as nn
except Exception:
    torch = None; nn = None
def _blk(i,o):
    return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
class UNet(nn.Module):
    def __init__(self,in_ch=3,out_ch=1,base=32):
        super().__init__()
        self.d1=_blk(in_ch,base); self.p1=nn.MaxPool2d(2)
        self.d2=_blk(base,base*2); self.p2=nn.MaxPool2d(2)
        self.d3=_blk(base*2,base*4); self.p3=nn.MaxPool2d(2)
        self.d4=_blk(base*4,base*8); self.p4=nn.MaxPool2d(2)
        self.b=_blk(base*8,base*16)
        self.u4=nn.ConvTranspose2d(base*16,base*8,2,2); self.c4=_blk(base*16,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.c3=_blk(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.c2=_blk(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.c1=_blk(base*2,base)
        self.out=nn.Conv2d(base,out_ch,1)
    def forward(self,x):
        d1=self.d1(x); d2=self.d2(self.p1(d1)); d3=self.d3(self.p2(d2)); d4=self.d4(self.p3(d3))
        b=self.b(self.p4(d4))
        u4=self.c4(torch.cat([self.u4(b),d4],1))
        u3=self.c3(torch.cat([self.u3(u4),d3],1))
        u2=self.c2(torch.cat([self.u2(u3),d2],1))
        u1=self.c1(torch.cat([self.u1(u2),d1],1))
        return self.out(u1)
