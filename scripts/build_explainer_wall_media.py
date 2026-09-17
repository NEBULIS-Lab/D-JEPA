"""CPU-only presentation crops of published videos; originals remain unchanged."""
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

root = Path(__file__).resolve().parents[1] / 'docs'
out = root / 'static/videos/explainer-wall'
out.mkdir(exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
filters = {
    'reacher': '[0:v]split=2[a][b];[a]crop=960:960:30:95,scale=320:320[l];[b]crop=960:960:1950:95,scale=320:320[r];[l][r]hstack=inputs=2,fps=12[v]',
    'granular': '[0:v]split=2[a][b];[a]crop=960:960:30:95,scale=320:320[l];[b]crop=960:960:1950:95,scale=320:320[r];[l][r]hstack=inputs=2,fps=12[v]',
    'shape': '[0:v]split=2[a][b];[a]crop=960:960:30:95,scale=320:320[l];[b]crop=960:960:3870:95,scale=320:320[r];[l][r]hstack=inputs=2,fps=12[v]',
    'robotwin': '[0:v]crop=1920:690:0:72,scale=800:-2,fps=12[v]',
    'driving': '[0:v]scale=800:-2,fps=12[v]',
}
entries = []
for name, filtering in filters.items():
    source = root / f'static/videos/{name}.mp4'
    target = out / f'{name}.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-threads', '2',
        '-i', str(source), '-filter_complex_threads', '1', '-filter_complex', filtering,
        '-map', '[v]', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
        '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
    poster = out / f'{name}.jpg'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-threads', '2',
        '-i', str(target), '-frames:v', '1', '-threads', '1', str(poster)], check=True)
    entries.append(dict(source=str(source.relative_to(root)), path=str(target.relative_to(root)),
        filter=filtering, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size,
        poster=str(poster.relative_to(root))))
    print(name, target.stat().st_size, flush=True)
(out / 'manifest.json').write_text(json.dumps(dict(
    description='Presentation previews derived from the full published videos. Camera crops preserve recorded actions and playback speed. Baseline left, D-JEPA right; shape omits the middle fusion panel. Driving retains shared camera context and both simulated trajectories.',
    entries=entries), indent=2) + '\n')
