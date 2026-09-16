#!/usr/bin/env python3
"""Drive-JEPA resource checks, scene manifests and candidate export.

No upstream files are edited. Oracle labels and observed futures are separate from inputs.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .contract import ROOT, admission, config, camera_paths, partition_rows, read_rows, write_csv, write_json, source_identity


def activate(cfg):
    repo = Path(cfg['repo']) / 'navsim_v1'
    if Path(cfg['assets']).name != 'Drive-JEPA-cache':
        raise ValueError('Upstream expects assets directory basename Drive-JEPA-cache')
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ.update(NAVSIM_EXP_ROOT=str(Path(cfg['assets']).parent),
                      NAVSIM_DEVKIT_ROOT=str(repo), NUPLAN_MAPS_ROOT=cfg['maps'],
                      NUPLAN_MAP_VERSION='nuplan-maps-v1.0', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg')
    import navsim
    if not Path(navsim.__file__).resolve().is_relative_to(repo.resolve()):
        raise RuntimeError('Wrong navsim import; WoTE and Drive-JEPA must not mix')


def make_loader(cfg, rows=None, sensors=False):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    filter_path = Path(cfg['repo']) / ('navsim_v1/navsim/planning/script/config/common/'
                                     f'train_test_split/scene_filter/{cfg["source_split"]}.yaml')
    filt = instantiate(OmegaConf.load(filter_path))
    if rows:
        requested_logs = {r['log_name'] for r in rows}
        if filt.log_names is not None and not requested_logs <= set(filt.log_names):
            raise ValueError('Manifest logs outside upstream scene filter')
        requested_tokens = {r['token'] for r in rows}
        if filt.tokens is not None and not requested_tokens <= set(filt.tokens):
            raise ValueError('Manifest tokens outside upstream scene filter')
        filt.log_names, filt.tokens = sorted(requested_logs), sorted(requested_tokens)
    filt.max_scenes = None
    sensor_cfg = SensorConfig.build_no_sensors()
    if sensors:
        sensor_cfg = SensorConfig(cam_f0=[2, 3], cam_b0=[3], cam_l0=[3], cam_r0=[3],
                                  cam_l1=[], cam_l2=[], cam_r1=[], cam_r2=[], lidar_pc=[])
    loader = SceneLoader(Path(cfg['logs']), Path(cfg['sensors']), filt, sensor_cfg)
    if rows and set(loader.tokens) != {r['token'] for r in rows}:
        raise ValueError('Missing requested scene; refusing silently reduced sample count')
    return loader


def inventory(cfg, args):
    loader = make_loader(cfg)
    pool = [dict(token=t, log_name=frames[3]['log_name'])
            for t, frames in loader.scene_frames_dicts.items()]
    # Freeze selection from metadata, before either outcomes or sensor availability.
    selected = partition_rows(pool, cfg['counts'], cfg['seed'])
    write_csv(args.output / 'scenes.csv', selected)
    requests = []
    for r in selected:
        frames = loader.scene_frames_dicts[r['token']]
        for purpose, paths in [('inference', camera_paths(frames)),
                               ('continuous_media', camera_paths(frames, media=True))]:
            for p in paths:
                requests.append({**r, 'purpose': purpose, 'relative_path': p,
                                 'present': (Path(cfg['sensors']) / p).is_file()})
    write_csv(args.output / 'camera_requests.csv', requests)
    for purpose in ('inference', 'continuous_media'):
        paths = sorted({r['relative_path'] for r in requests if r['purpose'] == purpose})
        with open(args.output / f'{purpose}_files.txt', 'x') as f:
            f.write('\n'.join(paths) + '\n')
    write_json(args.output / 'summary.json', {'state': 'manifest_prepared', 'counts': cfg['counts'],
        'protocol': cfg['protocol'], 'selection': 'metadata and seed only; no outcome filtering',
        'missing_file_references': sum(not r['present'] for r in requests),
        'note': 'File manifest does not reduce bytes of an indivisible remote tar archive'})


def load_agent(cfg):
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Native Drive-JEPA export requires CUDA')
    agent = DriveJEPAAgent(DriveJEPAConfig(), lr=1e-4, checkpoint_path=cfg['checkpoint'])
    agent.initialize()  # official strict state loading; no partial checkpoint acceptance
    agent.requires_grad_(False).eval().to('cuda')
    return agent


def to_local(poses, origin):
    import numpy as np
    value = np.asarray(poses).copy()
    c, s = np.cos(origin[2]), np.sin(origin[2])
    value[..., :2] = (value[..., :2] - origin[:2]) @ np.array([[c, -s], [s, c]])
    value[..., 2] = (value[..., 2] - origin[2] + np.pi) % (2*np.pi) - np.pi
    return value


def scene_context(scene, frames, cfg, origin):
    """Privileged visualization-only payload in a fixed initial rear-axle frame."""
    import numpy as np
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    annotations = []
    for frame in scene.frames[3:12]:
        ego = frame.ego_status.ego_pose
        boxes = frame.annotations.boxes
        c, s = np.cos(ego[2]), np.sin(ego[2])
        global_poses = np.column_stack((boxes[:, :2] @ np.array([[c, s], [-s, c]]) + ego[:2],
                                       boxes[:, 6] + ego[2]))
        local = to_local(global_poses, origin)
        annotations.append({'time': (frame.timestamp-scene.frames[3].timestamp)/1e6,
                            'boxes': np.column_stack((local, boxes[:, 3:5])).tolist(),
                            'names': list(frame.annotations.names)})
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    roads = scene.map_api.get_proximal_map_objects(Point2D(*origin[:2]), 100, layers)
    lines = []
    for layer in layers:
        for road in roads[layer]:
            xy = np.asarray(road.baseline_path.linestring.coords)[:, :2]
            lines.append(to_local(np.column_stack((xy, np.zeros(len(xy)))), origin)[:, :2].tolist())
    return {'origin_global': origin.tolist(), 'annotations': annotations, 'road_centerlines': lines,
            'camera_paths': [str(Path(cfg['sensors']) / p) for p in camera_paths(frames, media=True)],
            'camera_times': [(f['timestamp']-frames[3]['timestamp'])/1e6 for f in frames],
            'semantics': 'Logged actors and cameras; simulated ego alternatives are NOT counterfactual camera video',
            'actor_interpolation': 'nearest recorded frame; no synthesized actor response'}


def cache(cfg, args):
    import lzma
    import pickle
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.evaluate.pdm_score import pdm_score

    all_rows = read_rows(cfg['manifest'])
    rows = [r for r in all_rows if r['role'] == args.role]
    if not rows:
        raise ValueError('No rows for requested role')
    rows.sort(key=lambda r: r['token'])
    if args.limit is not None:
        if args.role != 'fit' or not 0 < args.limit <= len(rows):
            raise ValueError('Only fitting smoke may limit the requested population')
        rows = rows[:args.limit]
    write_csv(args.output / 'manifest.csv', all_rows)
    write_csv(args.output / 'processed_manifest.csv', rows)
    agent = load_agent(cfg)
    model = agent._pad_model
    captured = {}

    def capture(module, inputs):
        proposals, features = inputs
        b, k, t, _ = proposals.shape
        captured['query'] = features.reshape(b, k, t, -1).amax(2).detach()

    hook = model.scorer.register_forward_pre_hook(capture)
    loader = make_loader(cfg, rows, sensors=True)
    scoring = OmegaConf.load(Path(cfg['repo']) / 'navsim_v1/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml')
    simulator, scorer = instantiate(scoring.simulator), instantiate(scoring.scorer)
    processor = MetricCacheProcessor(str(args.output / 'metric_cache'), False)
    metric_paths = {}
    if Path(cfg['metric_index']).is_file():
        with open(cfg['metric_index'], newline='') as f:
            for r in csv.DictReader(f):
                p = Path(r['path'])
                if not p.is_absolute() or r['token'] in metric_paths:
                    raise ValueError('Metric index requires unique tokens and absolute paths')
                metric_paths[r['token']] = p
    arrays, labels, contexts = [args.output / d for d in ('predictions', 'labels', 'contexts')]
    for p in (arrays, labels, contexts):
        p.mkdir()
    native = []
    simulated = {}
    official_simulate = simulator.simulate_proposals

    def recording_simulator(*a, **kw):
        states = official_simulate(*a, **kw)
        simulated['states'] = states.copy()
        return states

    simulator.simulate_proposals = recording_simulator
    try:
        for row in rows:
            token = row['token']
            frames = loader.scene_frames_dicts[token]
            missing = [p for p in camera_paths(frames) if not (Path(cfg['sensors']) / p).is_file()]
            if missing:
                raise FileNotFoundError(f'{token}: missing {missing}')
            features = agent.get_feature_builders()[0].compute_features(loader.get_agent_input_from_token(token))
            features = {k: v.unsqueeze(0).to('cuda') for k, v in features.items()}
            with torch.inference_mode():
                pred = agent(features)
            poses = pred['proposals'][0].cpu().numpy()
            score = pred['pdm_score'][0].cpu().numpy()
            factors = pred['pred_logit'][0].sigmoid().cpu().numpy()
            query = captured['query'][0].cpu().numpy()
            if poses.shape != (cfg['proposal_count'], 8, 3) or query.shape[0] != len(score):
                raise ValueError('Unexpected multi-proposal contract')
            if not all(np.isfinite(a).all() for a in (poses, score, factors, query)):
                raise ValueError('Nonfinite prediction')
            if not np.allclose(pred['trajectory'][0].cpu().numpy(), poses[score.argmax()]):
                raise ValueError('Export selector differs from official forward')
            prediction_file = arrays / f'{token}.npz'
            np.savez_compressed(prediction_file, candidate_id=np.arange(len(score)), trajectory=poses,
                                native_score=score, predicted_factors=factors, query_feature=query)
            scene = Scene.from_scene_dict_list(frames, Path(cfg['sensors']), 4, 10, SensorConfig.build_no_sensors())
            if token in metric_paths:
                metric_path = metric_paths[token]
            else:
                scenario = NavSimScenario(scene, map_root=cfg['maps'], map_version='nuplan-maps-v1.0')
                metric_path = Path(processor.compute_metric_cache(scenario).file_name)
            with lzma.open(metric_path, 'rb') as f:
                metric = pickle.load(f)
            if np.linalg.norm(np.asarray(metric.ego_state.rear_axle.serialize())[:2] - scene.frames[3].ego_status.ego_pose[:2]) > 0.1:
                raise ValueError('Metric cache initial position does not match scene')
            if metric.ego_state.time_point.time_us != scene.frames[3].timestamp:
                raise ValueError('Metric cache timestamp does not match scene')
            origin = np.asarray(metric.ego_state.rear_axle.serialize())
            results, rollouts = [], []
            for i, trajectory in enumerate(poses):
                result = pdm_score(metric, Trajectory(trajectory), simulator.proposal_sampling, simulator, scorer)
                result = {k: float(v) for k, v in asdict(result).items()}
                results.append(dict(candidate_id=i, **result))
                # Exact states scored by official simulator (index 0 is reference).
                rollouts.append(to_local(simulated['states'][1, :, :3], origin))
            write_json(labels / f'{token}.json', {'token': token, 'role': row['role'], 'labels': results,
                       'metric_cache': str(metric_path), 'prediction_sha256': hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
                       'semantics': 'Each candidate evaluated against official reference; not joint-pool progress normalization'})
            np.savez_compressed(contexts / f'{token}.rollouts.npz', simulated_ego=np.stack(rollouts),
                                times=np.arange(len(rollouts[0])) * simulator.proposal_sampling.interval_length)
            write_json(contexts / f'{token}.json', scene_context(scene, frames, cfg, origin))
            native.append({**row, **results[int(score.argmax())]})
            print(f'{token}: {len(results)} candidates; full simulation horizon cached', flush=True)
        write_csv(args.output / 'native_results.csv', native)
        write_json(args.output / 'summary.json', {'state': 'complete', 'role': args.role, 'count': len(rows),
            'PDMS_0_to_1': float(np.mean([r['score'] for r in native])), 'protocol': cfg['protocol'],
            'selection': cfg['selection'], 'feature_identity': 'proposal query, NOT explicit future-scene latent'})
    finally:
        hook.remove()
        simulator.simulate_proposals = official_simulate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['doctor', 'inventory', 'cache'], required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--role', choices=['fit', 'calibration', 'test'], default='fit')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    cfg = config(args.config)
    if args.stage == 'doctor':
        print(json.dumps(admission(cfg), indent=2))
        return
    if args.output is None:
        p.error('--output required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', cfg)
    try:
        write_json(args.output / 'provenance.json', {
            **source_identity(cfg['repo']),
            'python': sys.version,
            'checkpoint': cfg['checkpoint'],
            'checkpoint_bytes': Path(cfg['checkpoint']).stat().st_size if Path(cfg['checkpoint']).exists() else None,
            'manifest_sha256': hashlib.sha256(Path(cfg['manifest']).read_bytes()).hexdigest() if Path(cfg['manifest']).is_file() else None,
            'selection': cfg['selection'], 'protocol': cfg['protocol']})
        activate(cfg)
        import torch
        write_json(args.output / 'framework.json', {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                   'device': 'cuda'})
        (inventory if args.stage == 'inventory' else cache)(cfg, args)
    except Exception:
        write_json(args.output / 'failure.json', {'state': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
