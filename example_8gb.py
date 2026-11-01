"""
Image to 3D on an 8 GB GPU.

Produces a low face count GLB with a baked normal map, so the asset keeps
high-resolution surface detail.

    python example_8gb.py --image assets/example_image/T.png --out sample.glb
    python example_8gb.py --image my.png --faces 50000 --texture-size 4096 --video
"""
import argparse
import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from trellis2.utils import vram_profiles

vram_profiles.apply_runtime_settings()

import time
import torch
from PIL import Image
import o_voxel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', default='assets/example_image/T.png')
    parser.add_argument('--out', default='sample.glb')
    parser.add_argument('--model', default='microsoft/TRELLIS.2-4B')
    parser.add_argument('--profile', default='8gb', choices=list(vram_profiles.PROFILES))
    parser.add_argument('--resolution', choices=['512', '1024', '1536'], default=None,
                        help='Override the profile resolution.')
    parser.add_argument('--faces', type=int, default=None, help='Target face count of the exported mesh.')
    parser.add_argument('--texture-size', type=int, default=None, choices=[1024, 2048, 4096])
    parser.add_argument('--no-normal-map', action='store_true', help='Skip normal map baking.')
    parser.add_argument('--steps', type=int, default=None,
                        help='Sampling steps for every stage. Fewer steps run faster (default 12).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--video', action='store_true', help='Also render a preview video.')
    args = parser.parse_args()

    profile = vram_profiles.get_profile(args.profile)
    glb_args = dict(profile['glb'])
    if args.faces is not None:
        glb_args['decimation_target'] = args.faces
    if args.texture_size is not None:
        glb_args['texture_size'] = args.texture_size
    if args.no_normal_map:
        glb_args['bake_normal_map'] = False
    pipeline_type = {'512': '512', '1024': '1024_cascade', '1536': '1536_cascade', None: None}[args.resolution]

    sampler = {'steps': args.steps} if args.steps else {}

    t0 = time.time()
    pipeline = vram_profiles.load_pipeline(args.model, profile)
    print(f'Pipeline loaded in {time.time() - t0:.1f}s')

    image = Image.open(args.image)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mesh = vram_profiles.run_with_fallback(
        pipeline, image, profile,
        pipeline_type=pipeline_type,
        seed=args.seed,
        sparse_structure_sampler_params=sampler,
        shape_slat_sampler_params=sampler,
        tex_slat_sampler_params=sampler,
    )[0]
    print(f'Generation: {time.time() - t0:.1f}s, {vram_profiles.peak_memory_report()}')

    if args.video:
        import cv2
        import imageio
        from trellis2.renderers import EnvMap
        from trellis2.utils import render_utils
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        preview = mesh
        if mesh.faces.shape[0] > profile['preview_max_faces']:
            import copy
            preview = copy.copy(mesh)
            preview.simplify(profile['preview_max_faces'])
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(preview, envmap=envmap))
        video_path = os.path.splitext(args.out)[0] + '.mp4'
        imageio.mimsave(video_path, video, fps=15)
        print(f'Video saved to {video_path}')
        del preview, envmap
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        verbose=True,
        **glb_args,
    )
    glb.export(args.out, extension_webp=True)
    print(f'Export: {time.time() - t0:.1f}s, {vram_profiles.peak_memory_report()}')
    print(f'Saved {args.out} ({glb_args["decimation_target"]} target faces, '
          f'{glb_args["texture_size"]}px textures, normal map: {glb_args["bake_normal_map"]})')


if __name__ == '__main__':
    main()
