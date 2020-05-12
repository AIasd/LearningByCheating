from pathlib import Path

import pandas as pd
import numpy as np
import tqdm
import time

import bird_view.utils.bz_utils as bzu
import bird_view.utils.carla_utils as cu
import carla

from carla import LaneType

from bird_view.models.common import crop_birdview

import scipy.misc
import datetime
import re
import os

def _paint(observations, control, diagnostic, debug, env, show=False):
    import cv2
    import numpy as np


    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2

    birdview = cu.visualize_birdview(observations['birdview'])
    birdview = crop_birdview(birdview)

    if 'big_cam' in observations:
        canvas = np.uint8(observations['big_cam']).copy()
        rgb = np.uint8(observations['rgb']).copy()
    else:
        canvas = np.uint8(observations['rgb']).copy()

    def _stick_together(a, b, axis=1):

        if axis == 1:
            h = min(a.shape[0], b.shape[0])

            r1 = h / a.shape[0]
            r2 = h / b.shape[0]

            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

            return np.concatenate([a, b], 1)

        else:
            h = min(a.shape[1], b.shape[1])

            r1 = h / a.shape[1]
            r2 = h / b.shape[1]

            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

            return np.concatenate([a, b], 0)

    def _write(text, i, j, canvas=canvas, fontsize=0.4):
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 9) for x in range(9+1)]
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, WHITE, 1)

    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(observations['command'], '???')

    if 'big_cam' in observations:
        fontsize = 0.8
    else:
        fontsize = 0.4

    _write('Command: ' + _command, 1, 0, fontsize=fontsize)
    _write('Velocity: %.1f' % np.linalg.norm(observations['velocity']), 2, 0, fontsize=fontsize)

    _write('Steer: %.2f' % control.steer, 4, 0, fontsize=fontsize)
    _write('Throttle: %.2f' % control.throttle, 5, 0, fontsize=fontsize)
    _write('Brake: %.1f' % control.brake, 6, 0, fontsize=fontsize)

    _write('Collided: %s' % diagnostic['collided'], 1, 6, fontsize=fontsize)
    _write('Invaded: %s' % diagnostic['invaded'], 2, 6, fontsize=fontsize)
    _write('Lights Ran: %d/%d' % (env.traffic_tracker.total_lights_ran, env.traffic_tracker.total_lights), 3, 6, fontsize=fontsize)
    _write('Goal: %.1f' % diagnostic['distance_to_goal'], 4, 6, fontsize=fontsize)

    _write('Time: %d' % env._tick, 5, 6, fontsize=fontsize)
    _write('FPS: %.2f' % (env._tick / (diagnostic['wall'])), 6, 6, fontsize=fontsize)

    for x, y in debug.get('locations', []):
        x = int(X - x / 2.0 * CROP_SIZE)
        y = int(Y + y / 2.0 * CROP_SIZE)

        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED

    for x, y in debug.get('locations_world', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED

    for x, y in debug.get('locations_birdview', []):
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED

    for x, y in debug.get('locations_pixel', []):
        S = R // 2
        if 'big_cam' in observations:
            rgb[y-S:y+S+1,x-S:x+S+1] = RED
        else:
            canvas[y-S:y+S+1,x-S:x+S+1] = RED

    for x, y in debug.get('curve', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        try:
            birdview[x,y] = [155, 0, 155]
        except:
            pass

    if 'target' in debug:
        x, y = debug['target'][:2]
        x = int(X - x * 4)
        y = int(Y + y * 4)
        birdview[x-R:x+R+1,y-R:y+R+1] = [0, 155, 155]

    ox, oy = observations['orientation']
    rot = np.array([
        [ox, oy],
        [-oy, ox]])
    u = observations['node'] - observations['position'][:2]
    v = observations['next'] - observations['position'][:2]
    u = rot.dot(u)
    x, y = u
    x = int(X - x * 4)
    y = int(Y + y * 4)
    v = rot.dot(v)
    x, y = v
    x = int(X - x * 4)
    y = int(Y + y * 4)

    if 'big_cam' in observations:
        _write('Network input/output', 1, 0, canvas=rgb)
        _write('Projected output', 1, 0, canvas=birdview)
        full = _stick_together(rgb, birdview)
    else:
        full = _stick_together(canvas, birdview)

    if 'image' in debug:
        full = _stick_together(full, cu.visualize_predicted_birdview(debug['image'], 0.01))

    if 'big_cam' in observations:
        full = _stick_together(canvas, full, axis=0)

    if show:
        bzu.show_image('canvas', full)
    bzu.add_to_video(full)


def run_single(env, weather, start, target, agent_maker, seed, autopilot, show=False, model_path=None, suite_name=None):
    # addition from agent.py
    from skimage.io import imread
    _road_map = imread('PythonAPI/agents/navigation/%s.png' % env._map.name)
    WORLD_OFFSETS = {
        'Town01' : (-52.059906005859375, -52.04995942115784),
        'Town02' : (-57.459808349609375, 55.3907470703125)
    }
    PIXELS_PER_METER = 5
    def _world_to_pixel(vehicle, location, offset=(0, 0)):
        world_offset = WORLD_OFFSETS[env._map.name]
        x = PIXELS_PER_METER * (location.x - world_offset[0])
        y = PIXELS_PER_METER * (location.y - world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]
    def _is_point_on_sidewalk(vehicle, loc):
        # Convert to pixel coordinate
        pixel_x, pixel_y = _world_to_pixel(vehicle, loc)
        pixel_y = np.clip(pixel_y, 0, _road_map.shape[0]-1)
        pixel_x = np.clip(pixel_x, 0, _road_map.shape[1]-1)
        point = _road_map[pixel_y, pixel_x, 0]

        return point == 0

    # ------------------------------------------------
    # HACK: deterministic vehicle spawns.
    env.seed = seed

    # modifications
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])

    if not autopilot:
        agent = agent_maker()
    else:
        agent = agent_maker(env._player, resolution=1, threshold=7.5)
        agent.set_route(env._start_pose.location, env._target_pose.location)

    diagnostics = list()
    result = {
            'weather': weather,
            'start': start, 'target': target,
            'success': None, 't': None,
            'total_lights_ran': None,
            'total_lights': None,
            'collided': None,
            }

    # modifications
    all_data_folder = 'collected_data'
    if not os.path.exists(all_data_folder):
        os.mkdir(all_data_folder)
    data_folder = all_data_folder+'/'+suite_name
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    trial_folder = data_folder+'/'+'0'
    counter = 0
    while os.path.exists(trial_folder):
        counter += 1
        trial_folder = data_folder+'/'+str(counter)
    os.mkdir(trial_folder)

    image_folder = trial_folder+'/'+'images'
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    all_misbehavior_logfile_path = data_folder+'/'+'all_misbehavior_driving_log.csv'

    logfile_path = trial_folder+'/'+'driving_log.csv'
    misbehavior_logfile_path = trial_folder+'/'+'misbehavior_driving_log.csv'


    title = ','.join(['FrameId', 'center', 'steering', 'throttle', 'brake', 'speed', 'command', 'Self Driving Model','Suit Name', 'Weather', 'Crashed', 'Crashed Type', 'Tot Crashes', 'Tot Lights Ran', 'Tot Lights', 'dist_ped', 'dist_vehicle', 'offroad', 'road_type', 'status'])

    with open(logfile_path, 'a+') as f_out:
        f_out.write(title+'\n')
    with open(misbehavior_logfile_path, 'a+') as f_out:
        f_out.write(title+','+'problem_type'+'\n')
    with open(all_misbehavior_logfile_path, 'a+') as f_out:
        f_out.write('counter'+','+title+','+'problem_type'+'\n')

    total_lights = 0
    total_lights_ran = 0
    collided = False
    total_crashes = 0
    frame_id = 0
    LAST_ROUND = False
    while env.tick() and not LAST_ROUND:
        observations = env.get_observations()
        control = agent.run_step(observations)
        diagnostic = env.apply_control(control)

        _paint(observations, control, diagnostic, agent.debug, env, show=show)

        diagnostic.pop('viz_img')
        diagnostics.append(diagnostic)

        # modification
        status = 'progress'
        if env.is_failure() or env.is_success():
            result['success'] = env.is_success()
            result['total_lights_ran'] = env.traffic_tracker.total_lights_ran
            result['total_lights'] = env.traffic_tracker.total_lights
            result['collided'] = env.collided
            result['t'] = env._tick
            LAST_ROUND = True
            if env.is_failure():
                status = 'failure'
            elif env.is_success():
                status = 'success'

        # additions
        from agents.tools.misc import is_within_distance_ahead, compute_yaw_difference
        total_lights = env.traffic_tracker.total_lights
        total_lights_ran = env.traffic_tracker.total_lights_ran
        collided = env.collided


        _proximity_threshold_vehicle = 5.5 # 9.5 for autopilot to avoid crash
        _proximity_threshold_ped = 2.5 # 9.5 for autopilot to avoid crash
        actor_list = env._world.get_actors()
        vehicle_list = actor_list.filter('*vehicle*')
        # lights_list = actor_list.filter('*traffic_light*')
        walkers_list = actor_list.filter('*walker*')


        ego_vehicle_location = env._player.get_location()
        ego_vehicle_orientation = env._player.get_transform().rotation.yaw
        ego_vehicle_waypoint = env._map.get_waypoint(ego_vehicle_location, project_to_road=False, lane_type=LaneType.Any)


        offroad = False
        if not ego_vehicle_waypoint:
            print('-'*100, 'no lane', '-'*100)
            offroad = True
        elif ego_vehicle_waypoint.lane_type != LaneType.Driving:
            print('-'*100, ego_vehicle_waypoint.lane_type, '-'*100)
            offroad = True



        dist_ped = 10000
        dist_vehicle = 10000

        # crash_type
        crash_type = 'none'
        if collided:
            crash_type = 'other'
            total_crashes += 1

        for walker in walkers_list:
            loc = walker.get_location()
            cur_dist_ped = loc.distance(ego_vehicle_location)
            degree = 162 / (np.clip(dist_ped, 1.5, 10.5)+0.3)
            if _is_point_on_sidewalk(env._player, loc):
                continue

            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        env._player.get_transform().rotation.yaw, _proximity_threshold_vehicle, degree=cur_dist_ped):
                crash_type = 'pedestrian'
            if dist_ped > cur_dist_ped:
                dist_ped = cur_dist_ped

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == env._player.id:
                continue

            loc = target_vehicle.get_location()
            ori = target_vehicle.get_transform().rotation.yaw

            target_vehicle_waypoint = env._map.get_waypoint(target_vehicle.get_location())

            if compute_yaw_difference(ego_vehicle_orientation, ori) <= 150 and is_within_distance_ahead(loc, ego_vehicle_location,
                                        env._player.get_transform().rotation.yaw, _proximity_threshold_vehicle, degree=45):
                crash_type = 'vehicle'
            cur_dist_vehicle = np.linalg.norm(np.array([
                loc.x - ego_vehicle_location.x,
                loc.y - ego_vehicle_location.y]))
            if dist_vehicle > cur_dist_vehicle:
                dist_vehicle = cur_dist_vehicle

        prev_total_lights_ran = 0
        with open(logfile_path, 'a+') as f_out:
            dt = str(datetime.datetime.now())
            m = re.search("(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+).\d+", dt)
            if m:
                time_info = '_'.join(m.groups())
            else:
                time_info = ''


            center_address = image_folder+'/'+'center_'+str(frame_id)+'_'+time_info+'.jpg'
            scipy.misc.toimage(observations['rgb'], cmin=0.0, cmax=...).save(center_address)

            log_text = ','.join([str(frame_id), center_address, str(control.steer), str(control.throttle), str(control.brake), str(diagnostic['speed']), str(observations['command']), str(model_path), suite_name, str(weather), str(collided), crash_type, str(total_crashes), str(total_lights_ran), str(total_lights), str(dist_ped), str(dist_vehicle), str(offroad), str(ego_vehicle_waypoint.lane_type), status])

            f_out.write(log_text+'\n')


        if collided or offroad or total_lights_ran > prev_total_lights_ran:
            if collided:
                problem_type = 'collision'
            elif offroad:
                problem_type = 'offroad'
            elif total_lights_ran > prev_total_lights_ran:
                problem_type = 'light_ran'
            with open(misbehavior_logfile_path, 'a+') as f_out:
                f_out.write(log_text+','+problem_type+'\n')
            with open(all_misbehavior_logfile_path, 'a+') as f_out:
                f_out.write(str(counter)+','+log_text+','+problem_type+'\n')

        frame_id += 1
        prev_total_lights_ran = total_lights_ran
    if env.is_failure():
        print('+'*100, 'collision:', str(result['collided']), '+'*100)
    # -------------------------------------------------------------
    return result, diagnostics


def run_benchmark(agent_maker, env, benchmark_dir, seed, autopilot, resume, max_run=5, show=False, model_path=None, suite_name=None):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    summary_csv = benchmark_dir / 'summary.csv'
    diagnostics_dir = benchmark_dir / 'diagnostics'
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = list()
    total = len(list(env.all_tasks))

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()



    num_run = 0

    for weather, (start, target), run_name in tqdm.tqdm(env.all_tasks, total=total):
        print('+'*200)
        print(weather, start, target)
        print('+'*200)
        if resume and len(summary) > 0 and ((summary['start'] == start) \
                       & (summary['target'] == target) \
                       & (summary['weather'] == weather)).any():
            print (weather, start, target)
            continue


        diagnostics_csv = str(diagnostics_dir / ('%s.csv' % run_name))

        bzu.init_video(save_dir=str(benchmark_dir / 'videos'), save_path=run_name)

        result, diagnostics = run_single(env, weather, start, target, agent_maker, seed, autopilot, show=show, model_path=model_path, suite_name=suite_name)

        summary = summary.append(result, ignore_index=True)

        # Do this every timestep just in case.
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)

        num_run += 1

        if num_run >= max_run:
            break
