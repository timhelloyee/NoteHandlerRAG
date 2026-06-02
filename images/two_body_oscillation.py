"""Render a two-body oscillation as an animated GIF.

Two masses (m1, m2) connected by a spring oscillate about their common
centre of mass. The COM stays fixed, so amplitudes are inversely
proportional to mass: the lighter mass swings farther.
"""
import math
from PIL import Image, ImageDraw, ImageFont

W, H = 640, 260
YC = H // 2
N_FRAMES = 72
PERIOD = N_FRAMES  # one full oscillation per loop

# --- physics setup ---------------------------------------------------------
m1, m2 = 1.0, 2.0                       # masses (m2 heavier -> moves less)
x1_eq, x2_eq = 190, 450                 # equilibrium x positions (px)
L0 = x2_eq - x1_eq                      # equilibrium separation
COM = (m1 * x1_eq + m2 * x2_eq) / (m1 + m2)
dL = 90                                 # separation amplitude (px)

r1 = 20                                 # radii scaled ~ by mass
r2 = 26

BG = (18, 20, 28)
SPRING = (150, 160, 180)
COL1 = (90, 200, 255)
COL2 = (255, 140, 90)
COM_COL = (120, 200, 140)
TEXT = (235, 235, 235)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except OSError:
    font = fsm = ImageFont.load_default()


def spring_points(xa, xb, yc, coils=12, amp=14, lead=14):
    """Zigzag coil from (xa,yc) to (xb,yc)."""
    pts = [(xa, yc), (xa + lead, yc)]
    span = (xb - lead) - (xa + lead)
    for i in range(1, 2 * coils):
        x = xa + lead + span * i / (2 * coils)
        y = yc + (amp if i % 2 else -amp)
        pts.append((x, y))
    pts += [(xb - lead, yc), (xb, yc)]
    return pts


frames = []
for f in range(N_FRAMES):
    theta = 2 * math.pi * f / PERIOD
    sep = L0 + dL * math.cos(theta)
    x1 = COM - (m2 / (m1 + m2)) * sep
    x2 = COM + (m1 / (m1 + m2)) * sep

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # ground/reference line
    d.line([(0, YC + 70), (W, YC + 70)], fill=(45, 48, 60), width=2)

    # fixed centre-of-mass marker (dashed vertical line)
    for y in range(20, H - 20, 14):
        d.line([(COM, y), (COM, y + 7)], fill=COM_COL, width=2)
    d.text((COM - 18, 14), "COM", font=fsm, fill=COM_COL)

    # spring between the inner edges of the masses
    pts = spring_points(x1 + r1, x2 - r2, YC, coils=12, amp=14)
    d.line(pts, fill=SPRING, width=3, joint="curve")

    # masses
    d.ellipse([x1 - r1, YC - r1, x1 + r1, YC + r1], fill=COL1, outline=(255, 255, 255), width=2)
    d.ellipse([x2 - r2, YC - r2, x2 + r2, YC + r2], fill=COL2, outline=(255, 255, 255), width=2)
    d.text((x1 - 8, YC - 10), "m", font=font, fill=(0, 0, 0))
    d.text((x2 - 8, YC - 10), "2m", font=font, fill=(0, 0, 0))

    d.text((14, H - 26), "Two-body oscillation  (COM fixed, lighter mass swings farther)",
           font=fsm, fill=TEXT)

    frames.append(img)

out = "/home/user/NoteHandlerRAG/images/two_body_oscillation.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)
print("wrote", out)
