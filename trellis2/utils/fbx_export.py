"""
FBX export for TRELLIS.2 assets.

FBX is a closed format with no good pure-Python writer, so the conversion
runs through Blender. Two ways to provide it:

1. The `bpy` pip package (Blender as a Python module):
       pip install bpy          # needs the Python version bpy was built for
2. A Blender install, 4.2 or newer. Blender 4.x and 5.x both work.
   `. ./setup.sh --blender` installs 5.2 LTS.
   Or put `blender` on PATH, or set
       export TRELLIS2_BLENDER=/path/to/blender

Usage:
    glb = o_voxel.postprocess.to_glb(...)
    export_fbx(glb, 'model.fbx')

    glb_to_fbx('model.glb', 'model.fbx')   # convert an existing GLB
"""
import functools
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blender_glb_to_fbx.py')

MIN_BLENDER_VERSION = (4, 2, 0)

_SEARCH_PATTERNS = (
    # setup.sh --blender (any version)
    os.path.expanduser('~/.local/share/trellis2/blender-*-linux-x64/blender'),
    # data_toolkit/dump_mesh.py
    '/tmp/blender-*-linux-x64/blender',
    # macOS
    '/Applications/Blender*.app/Contents/MacOS/Blender',
    # Windows
    'C:/Program Files/Blender Foundation/Blender */blender.exe',
)


def _version_from_text(text: str) -> Optional[Tuple[int, ...]]:
    match = re.search(r'(\d+)\.(\d+)(?:\.(\d+))?', text)
    if match is None:
        return None
    return tuple(int(g or 0) for g in match.groups())


@functools.lru_cache(maxsize=None)
def blender_version(exe: str) -> Optional[Tuple[int, ...]]:
    """Version of a Blender executable, e.g. (5, 2, 3). None if it does not run."""
    try:
        out = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.startswith('Blender'):
            return _version_from_text(line)
    return None


def _is_executable(path: Optional[str]) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _discovered_blenders():
    found = []
    for pattern in _SEARCH_PATTERNS:
        found += [p for p in glob.glob(pattern) if _is_executable(p)]
    # Newest version first, judged by the folder name (blender-5.2.3-linux-x64).
    return sorted(set(found), key=lambda p: _version_from_text(os.path.basename(os.path.dirname(p))) or (0,), reverse=True)


def find_blender(blender: Optional[str] = None) -> Optional[str]:
    """
    Return a Blender executable path (4.2 or newer), or None.

    Order: the `blender` argument, `TRELLIS2_BLENDER`, `blender` on PATH, then
    known install folders, newest version first.
    """
    explicit = blender or os.environ.get('TRELLIS2_BLENDER')
    if explicit:
        if not _is_executable(explicit):
            raise RuntimeError(f'Blender not found or not executable: {explicit}')
        version = blender_version(explicit)
        if version is not None and version < MIN_BLENDER_VERSION:
            raise RuntimeError(f'{explicit} is Blender {".".join(map(str, version))}. FBX export needs 4.2 or newer.')
        return explicit
    for candidate in [shutil.which('blender'), *_discovered_blenders()]:
        if not _is_executable(candidate):
            continue
        version = blender_version(candidate)
        if version is None or version >= MIN_BLENDER_VERSION:
            return candidate
    return None


def bpy_available() -> bool:
    spec = importlib.util.find_spec('bpy')
    # A plain folder named `bpy` shows up as a namespace package with no origin.
    return spec is not None and spec.origin is not None


def fbx_available(blender: Optional[str] = None) -> bool:
    return find_blender(blender) is not None or bpy_available()


def glb_to_fbx(glb_path: str, fbx_path: str, blender: Optional[str] = None, timeout: int = 900, verbose: bool = False) -> str:
    """
    Convert a GLB file to FBX with embedded textures.

    Prefers a Blender executable, then the `bpy` package. Both run in a
    separate process, so Blender never shares memory or CUDA state with torch.
    """
    if not os.path.isfile(glb_path):
        raise FileNotFoundError(glb_path)
    exe = find_blender(blender)
    if exe is not None:
        cmd = [exe, '-b', '--factory-startup', '-P', SCRIPT, '--', glb_path, fbx_path]
    elif bpy_available():
        cmd = [sys.executable, SCRIPT, glb_path, fbx_path]
    else:
        raise RuntimeError(
            'FBX export needs Blender 4.2+ or 5.x. Run `. ./setup.sh --blender`, '
            'install the bpy package (pip install bpy), put blender on PATH, or set TRELLIS2_BLENDER.'
        )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 or not os.path.isfile(fbx_path):
        raise RuntimeError(f'FBX conversion failed (exit {result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}')
    if verbose:
        print(result.stdout)
    return fbx_path


def export_fbx(mesh, fbx_path: str, blender: Optional[str] = None, verbose: bool = False) -> str:
    """
    Export a trimesh object (the return value of `o_voxel.postprocess.to_glb`) to FBX.

    Textures are written as PNG, since most FBX readers do not support WebP.
    """
    with tempfile.TemporaryDirectory(prefix='trellis2_fbx_') as tmp:
        glb_path = os.path.join(tmp, 'asset.glb')
        mesh.export(glb_path, extension_webp=False)
        return glb_to_fbx(glb_path, fbx_path, blender=blender, verbose=verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert GLB to FBX with embedded textures.')
    parser.add_argument('input', help='Input .glb')
    parser.add_argument('output', nargs='?', help='Output .fbx (default: same name)')
    parser.add_argument('--blender', default=None, help='Path to the Blender executable')
    args = parser.parse_args()
    out = args.output or os.path.splitext(args.input)[0] + '.fbx'
    print(f'Saved {glb_to_fbx(args.input, out, blender=args.blender)}')
