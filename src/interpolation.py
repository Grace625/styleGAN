import argparse
import math

import jittor as jt
jt.flags.use_cuda = True
jt.flags.log_silent = True

from model import StyledGenerator

@jt.no_grad()
def get_mean_style(generator):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(jt.randn(1024, 512))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to checkpoint file')

args = parser.parse_args()

generator = StyledGenerator(512)
ckpt = jt.load(args.path)
generator.load_state_dict(ckpt)
generator.eval()
mean_style = get_mean_style(generator)

step = int(math.log(128, 2)) - 2

code_1 = jt.randn(50, 512)
code_2 = jt.randn(50, 512)

inter_times = 2000

with jt.no_grad():
    img_1 = generator(
        code_1,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
with jt.no_grad():
    img_2 = generator(
        code_1,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
delta = (code_2 - code_1)/inter_times

for i in range(inter_times):
    image = generator(
        code_1 + delta * i,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
        # mixing_range=(0, 1),
    )
    jt.save_image(image, f'./output/sample_{i}.png', nrow=10, normalize=True, range=(-1, 1))