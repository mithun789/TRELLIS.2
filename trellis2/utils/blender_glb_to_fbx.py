"""
Convert a TRELLIS.2 GLB to FBX with embedded PBR textures.

Runs inside Blender (or the `bpy` pip package):
    blender -b --factory-startup -P blender_glb_to_fbx.py -- input.glb output.fbx

glTF packs roughness (G) and metallic (B) into one texture. The FBX exporter
only picks up image textures wired straight into a Principled BSDF input, so
this script splits that texture into separate grayscale roughness and
metallic images first. Base color, alpha and normal maps go through as-is.

No imports from trellis2: Blender's bundled Python does not have them.
"""
import os
import sys
import tempfile

import bpy


def _linked_node(socket):
    return socket.links[0].from_node if socket.is_linked else None


def _packed_png(name, width, height, rgba, colorspace, workdir):
    """
    Write pixels to a real PNG file, load it back, and pack it.
    Images generated in memory do not embed correctly in FBX.
    """
    path = os.path.join(workdir, f'{bpy.path.clean_name(name)}.png')
    tmp = bpy.data.images.new(name + '_tmp', width=width, height=height, alpha=True)
    tmp.pixels.foreach_set(rgba)
    tmp.filepath_raw = path
    tmp.file_format = 'PNG'
    tmp.save()
    bpy.data.images.remove(tmp)
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = colorspace
    img.pack()
    return img


def _new_gray_image(name, width, height, values, workdir):
    """values: flat float list of length width*height (one channel)."""
    rgba = [1.0] * (width * height * 4)
    rgba[0::4] = values
    rgba[1::4] = values
    rgba[2::4] = values
    return _packed_png(name, width, height, rgba, 'Non-Color', workdir)


def _split_metallic_roughness(material, workdir):
    """Rewire glTF metallicRoughness (Separate Color node) into two direct textures."""
    if not material.use_nodes:
        return
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        return
    sep = _linked_node(bsdf.inputs['Roughness']) or _linked_node(bsdf.inputs['Metallic'])
    if sep is None or sep.type not in ('SEPARATE_COLOR', 'SEPRGB'):
        return
    tex = _linked_node(sep.inputs[0])
    if tex is None or tex.type != 'TEX_IMAGE' or tex.image is None:
        return

    src = tex.image
    width, height = src.size
    pixels = [0.0] * (width * height * 4)
    src.pixels.foreach_get(pixels)

    # Separate Color outputs: 0=R, 1=G, 2=B. glTF: G=roughness, B=metallic.
    channel_of = {}
    for out_idx, out in enumerate(sep.outputs):
        for link in out.links:
            if link.to_node == bsdf:
                channel_of[link.to_socket.name] = out_idx

    for input_name, channel in channel_of.items():
        img = _new_gray_image(f'{material.name}_{input_name.lower()}', width, height, pixels[channel::4], workdir)
        node = nodes.new('ShaderNodeTexImage')
        node.image = img
        node.location = (tex.location[0], tex.location[1] - 300 * (channel - 1))
        links.new(node.outputs['Color'], bsdf.inputs[input_name])

    nodes.remove(sep)
    if not tex.outputs[0].is_linked:
        nodes.remove(tex)


def _ensure_packed_png(image, workdir):
    """Make every texture a packed PNG so the FBX embeds a widely supported format."""
    if image.type != 'IMAGE':
        return
    if image.file_format == 'PNG':
        if not image.packed_file:
            image.pack()
        return
    # WebP/JPEG etc: copy pixels into a new PNG image and point all users at it.
    width, height = image.size
    pixels = [0.0] * (width * height * 4)
    image.pixels.foreach_get(pixels)
    png = _packed_png(os.path.splitext(image.name)[0] + '_png', width, height, pixels,
                      image.colorspace_settings.name, workdir)
    image.user_remap(png)
    bpy.data.images.remove(image)


def convert(input_glb: str, output_fbx: str) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(input_glb))

    with tempfile.TemporaryDirectory(prefix='trellis2_fbx_') as workdir:
        for material in bpy.data.materials:
            _split_metallic_roughness(material, workdir)
        for image in list(bpy.data.images):
            if image.users > 0:
                _ensure_packed_png(image, workdir)
        _export_fbx(output_fbx)


def _export_fbx(output_fbx):
    bpy.ops.export_scene.fbx(
        filepath=os.path.abspath(output_fbx),
        use_selection=False,
        object_types={'MESH', 'EMPTY'},
        apply_unit_scale=True,
        mesh_smooth_type='FACE',
        path_mode='COPY',
        embed_textures=True,
        axis_forward='-Z',
        axis_up='Y',
    )


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    if len(argv) != 2:
        print('Usage: blender -b -P blender_glb_to_fbx.py -- input.glb output.fbx')
        sys.exit(1)
    convert(argv[0], argv[1])
    print(f'FBX saved to {argv[1]}')
