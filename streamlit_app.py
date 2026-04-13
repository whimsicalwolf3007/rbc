import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# ------------------------------
# Model definition (same as training)
# ------------------------------

class StandardCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, use_bn=True):
        super().__init__()
        p = ks // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=p),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=p),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=p),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DilatedCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, dilations=(1, 2, 4), use_bn=True):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, ks, padding=dilations[0], dilation=dilations[0])
        self.c2 = nn.Conv2d(out_ch, out_ch, ks, padding=dilations[1], dilation=dilations[1])
        self.c3 = nn.Conv2d(out_ch, out_ch, ks, padding=dilations[2], dilation=dilations[2])
        self.b1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.b2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.b3 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.b1(self.c1(x)))
        x = self.relu(self.b2(self.c2(x)))
        x = self.relu(self.b3(self.c3(x)))
        return x


class DepthwiseCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, use_bn=True):
        super().__init__()
        p = ks // 2

        def dw_pw(i, o):
            return nn.Sequential(
                nn.Conv2d(i, i, ks, padding=p, groups=i),
                nn.BatchNorm2d(i) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(i, o, 1),
                nn.BatchNorm2d(o) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
            )

        self.b1 = dw_pw(in_ch, out_ch)
        self.b2 = dw_pw(out_ch, out_ch)
        self.b3 = dw_pw(out_ch, out_ch)

    def forward(self, x):
        return self.b3(self.b2(self.b1(x)))


class MultiBranchModule(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, dilations=(1, 2, 4), use_bn=True):
        super().__init__()
        self.std = StandardCNNBlock(in_ch, out_ch, ks, use_bn)
        self.dil = DilatedCNNBlock(in_ch, out_ch, ks, dilations, use_bn)
        self.dw = DepthwiseCNNBlock(in_ch, out_ch, ks, use_bn)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.std(x), self.dil(x), self.dw(x)], dim=1))


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        mid = max(1, ch // r)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand=4, ks=3, use_se=True, use_bn=True):
        super().__init__()
        hid = in_ch * expand
        self.use_res = in_ch == out_ch
        layers = [
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid) if use_bn else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, hid, ks, padding=ks // 2, groups=hid, bias=False),
            nn.BatchNorm2d(hid) if use_bn else nn.Identity(),
            nn.SiLU(inplace=True),
        ]
        if use_se:
            layers.append(SEBlock(hid))
        layers += [
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class FusedMBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand=4, ks=3, use_bn=True):
        super().__init__()
        hid = in_ch * expand
        self.use_res = in_ch == out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hid, ks, padding=ks // 2, bias=False),
            nn.BatchNorm2d(hid) if use_bn else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = max(1, ch // 2)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([avg, mx], dim=1))


class ReducedUNet(nn.Module):
    def __init__(self, in_ch, base=16, use_bn=True, use_cbam=True):
        super().__init__()
        f1, f2, f3, fb = base, base * 2, base * 4, base * 8

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
            )

        self.enc1, self.enc2, self.enc3 = blk(in_ch, f1), blk(f1, f2), blk(f2, f3)
        self.pool = nn.MaxPool2d(2)
        self.bn_cnn = blk(f3, fb)
        self.bn_cbam = CBAM(f3) if use_cbam else nn.Identity()
        self.bn_fuse = nn.Conv2d(fb + f3, fb, 1)
        self.up3, self.dec3 = nn.ConvTranspose2d(fb, f3, 2, 2), blk(f3 + f3, f3)
        self.up2, self.dec2 = nn.ConvTranspose2d(f3, f2, 2, 2), blk(f2 + f2, f2)
        self.up1, self.dec1 = nn.ConvTranspose2d(f2, f1, 2, 2), blk(f1 + f1, f1)
        self.out = nn.Conv2d(f1, in_ch, 1) if f1 != in_ch else nn.Identity()

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        b = self.bn_fuse(torch.cat([self.bn_cnn(p3), self.bn_cbam(p3)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1) + x


class AdvancedDenoisingModel(nn.Module):
    def __init__(self, in_ch=1, s1=16, s2=32, unet_base=16, s4=16, use_bn=True, use_cbam=True):
        super().__init__()
        self.stage1 = MultiBranchModule(in_ch, s1, use_bn=use_bn)
        self.stage2 = MBConv(in_ch + s1, s2, use_se=False, use_bn=use_bn)
        self.stage3 = ReducedUNet(s2, base=unet_base, use_bn=use_bn, use_cbam=use_cbam)
        self.stage4 = FusedMBConv(s2, s4, use_bn=use_bn)
        self.refine = MultiBranchModule(s4, s4, use_bn=use_bn)
        self.output = nn.Conv2d(s4, in_ch, 1)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(torch.cat([x, s1], dim=1))
        s3 = self.stage3(s2)
        s4 = self.refine(self.stage4(s3))
        return x + self.output(s4)


# ------------------------------
# App utilities
# ------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "final_model.pth"
IMG_SIZE = (256, 256)


def to_tensor(pil_img):
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = arr.mean(axis=2, keepdims=True).transpose(2, 0, 1)
    return torch.from_numpy(arr)


def to_pil(t):
    t = t.detach().cpu().clamp(0, 1)
    if t.ndim == 4:
        t = t[0]
    img = (t[0].numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="L")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedDenoisingModel(in_ch=1, s1=16, s2=32, unet_base=16, s4=16).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, device


def denoise(model, device, pil_img):
    img = pil_img.convert("L").resize(IMG_SIZE, Image.NEAREST)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x).clamp(0, 1)
    return img, to_pil(y)


def denoise_rgb_channels(model, device, pil_img):
    rgb = pil_img.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    r, g, b = rgb.split()
    _, r_dn = denoise(model, device, r)
    _, g_dn = denoise(model, device, g)
    _, b_dn = denoise(model, device, b)
    return rgb, Image.merge("RGB", (r_dn, g_dn, b_dn))


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="CVDL Denoising Evaluation", layout="wide")
st.title("CVDL Denoising Evaluation")

st.markdown("Upload one or more noisy phase images. The app loads `models/final_model.pth`.")

try:
    model, device = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

use_rgb = st.checkbox("Process RGB channels separately", value=True)

uploads = st.file_uploader(
    "Upload noisy image(s)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploads:
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Noisy Input")
    with cols[1]:
        st.subheader("Denoised Output")

    for up in uploads:
        raw = up.read()
        pil_img = Image.open(io.BytesIO(raw))
        if use_rgb and pil_img.mode in {"RGB", "RGBA"}:
            noisy, denoised = denoise_rgb_channels(model, device, pil_img)
        else:
            noisy, denoised = denoise(model, device, pil_img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(noisy, caption=up.name, use_column_width=True)
        with col2:
            st.image(denoised, caption=f"Denoised - {up.name}", use_column_width=True)

        buf = io.BytesIO()
        denoised.save(buf, format="PNG")
        st.download_button(
            label=f"Download {up.name} denoised",
            data=buf.getvalue(),
            file_name=f"denoised_{Path(up.name).stem}.png",
            mime="image/png",
        )
else:
    st.info("Upload images to run evaluation.")
