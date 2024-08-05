#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
from pickle import TRUE
import sys
import time
import subprocess
from unittest import TextTestRunner

##### Execture CARLA #####
carla_path = "D:/carla-0.9.11_1/Unreal/CarlaUE4/Binaries/Win64/"
os.system(f"d:")
os.system(f"cd {carla_path}")

terminal_command = ".\CarlaUE4.exe.lnk -carla-rpc-port=2000 -fps=30 -directx12 -prefernvidia" 
process = subprocess.Popen(terminal_command, shell=True)
print(f'==== Loading CARLA ====')
time.sleep(15)
print(f'==== Loading Done ====')
##### Call CARLA #####
sys.path.append(
    'D:/carla-0.9.11_1/PythonAPI/carla/dist/carla-0.9.11-py3.7-win-amd64.egg')

import carla

from carla import VehicleLightState as vls
import math
import argparse
import logging
from numpy import random
import numpy as np
import cv2
from queue import Queue
from queue import Empty
import psutil

def killcarla():
    for proc in psutil.process_iter():
        # 프로세스 이름, PID값 가져오기
        processName = proc.name()
        processID = proc.pid
        #print(processName , ' - ', processID)

        if processName == "CarlaUE4.exe":
            parent_pid = processID  #PID
            parent = psutil.Process(parent_pid) # PID 찾기
            for child in parent.children(recursive=True):  #자식-부모 종료
                child.kill()
            parent.kill()

WEATHER_PRESETS = {
    'clear': [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.2, 0.0],
    'night': [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.2, 0.0],
    'dawn': [30.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.9, 6.0],
    'overcast': [30.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.9, 6.0], # 'overcast': [80.0, 0.0, 0.0, 50.0, 2.0, 0.0, 0.9, 10.0]
    'overcast1': [50.0, 0.0, 0.0, 20.0, 2.0, 0.0, 0.9, 9.0],
    'overcast2': [80.0, 0.0, 0.0, 30.0, 3.0, 0.0, 0.9, 12.0],
    'rain': [100.0, 40.0, 30.0, 15.0, 0.0, 0.0, 0.9, 100.0],
    'rain1': [100.0, 80.0, 90.0, 40.0, 2.0, 0.0, 0.9, 100.0]}

def apply_weather_presets(args_weather, weather):
    """Uses weather presets to set the weather parameters"""
    if args_weather is not None:
        if args_weather in WEATHER_PRESETS:
            weather.cloudiness = WEATHER_PRESETS[args_weather][0]
            weather.precipitation = WEATHER_PRESETS[args_weather][1]
            weather.precipitation_deposits = WEATHER_PRESETS[args_weather][2]
            weather.wind_intensity = WEATHER_PRESETS[args_weather][3]
            weather.fog_density = WEATHER_PRESETS[args_weather][4]
            weather.fog_distance = WEATHER_PRESETS[args_weather][5]
            weather.fog_falloff = WEATHER_PRESETS[args_weather][6]
            weather.wetness = WEATHER_PRESETS[args_weather][7]
        else:
            print("[ERROR]: Command [--weather | -w] '" + args_weather + "' not known")
            sys.exit(1)

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost', # 127.0.0.1
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=100,
        type=int,
        help='number of vehicles (default: 100)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50, # 50
        type=int,
        help='number of walkers (default: 0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        default=True,
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        default=True,
        help='Enanble')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        default=1123, #1102
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=True,
        help='Enanble car lights')
    argparser.add_argument(
        '--Map_sh',
        default='Town10HD',
        action="store",
    )
    argparser.add_argument(
        '--index',
        default=1,
        action="store",
    )
    argparser.add_argument(
        '--SpawnP_sh',
        default=118, # t2 91
        action="store",
    )
    argparser.add_argument(
        '--Vehicle_type_sh',
        default='vehicle.bmw.isetta',
        action="store",
    )
    argparser.add_argument(
        '--Camera_set_sh',
        default='Baseline_RD',
        action="store",
    )
    argparser.add_argument(
        '--Weather_sh',
        default='clear',
        action="store",
    )
    argparser.add_argument(
        '--Num_car_sh',
        default=1.5,
        action="store",
    )
    args = argparser.parse_args()
    
    framenum = 0
    def sensor_callback(sensor_data, sensor_queue, sensor_name, frame = framenum):
        sensor_queue.put((sensor_data, sensor_name))
        sensor_data.save_to_disk('E:\Dataset\\raw_dataset\GV360_train_' + str(args.Map_sh) + '_' + str(args.Camera_set_sh) + '_'
                                    + str(args.index) + '_' + str(args.Weather_sh) + '\%05d_%s.png' %(framenum, sensor_name)) #sensor_data.save_to_disk G:\CARLA_Dataset H:\Dataset

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    sensor_queue = Queue()
    record = True
    image_w = '960' # 1920 960 480
    image_h = '768' # 1536 768 384
    fovset = '185' #gear fov 195 focallength 140 # normal 185 280
    focallength = 280 # 560 280 140
    focallengh_x = str(focallength)
    focallengh_y = str(focallength)
    optical_x = str(int(image_w) // 2)
    optical_y = str(int(image_h) // 2)
    
    def setup_camera(bp_lib, world, vehicle_sh, camera_bp, camera_transform, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y):
        camera_bp = bp_lib.find('sensor.camera.fisheye') 
        camera_bp.set_attribute('x_size', str(image_w))  # Example: '1280'
        camera_bp.set_attribute('y_size', str(image_h))  # Example: '720'
        camera_bp.set_attribute('max_angle', str(fovset))
        camera_bp.set_attribute('d_1', str(0.08309221636708493))
        camera_bp.set_attribute('d_2', str(0.01112126630599195))
        camera_bp.set_attribute('d_3', str(-0.008587261043925865))
        camera_bp.set_attribute('d_4', str(0.0008542188930970716))
        camera_bp.set_attribute('f_x', str(focallengh_x))  # Example: '320'
        camera_bp.set_attribute('f_y', str(focallengh_y))  # Example: '320'
        camera_bp.set_attribute('c_x', str(optical_x))  # Example: '640'
        camera_bp.set_attribute('c_y', str(optical_y))  # Example: '480'
        return world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_sh)
    
    def rect_setup_camera(bp_lib, world, vehicle_sh, camera_bp, camera_transform, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y):
        camera_bp = bp_lib.find('sensor.camera.rgb') 
        camera_bp.set_attribute('image_size_x', str(image_w)) #1280
        camera_bp.set_attribute('image_size_y', str(image_h)) #720
        camera_bp.set_attribute('fov', str(fovset))
        return world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_sh)
    
    try:
        world = client.get_world()
        
        if args.Map_sh != 'Town10HD':
            world = client.load_world(args.Map_sh)
            print(f'Map set done\n')
            
        if args.Weather_sh == 'clear': #overcast
            weather = world.get_weather()
            weather = carla.WeatherParameters(sun_azimuth_angle=30.0, sun_altitude_angle=15.0) #(sun_azimuth_angle=20.0, sun_altitude_angle=120.0)
            apply_weather_presets(args.Weather_sh, weather)
            world.set_weather(weather)
            print(f'Weather set done\n')
        elif args.Weather_sh == 'overcast': #overcast
            weather = world.get_weather()
            weather = carla.WeatherParameters(sun_azimuth_angle=60.0, sun_altitude_angle=25.0) #(sun_azimuth_angle=20.0, sun_altitude_angle=120.0)
            apply_weather_presets(args.Weather_sh, weather)
            world.set_weather(weather)
            print(f'Weather set done\n')
        elif args.Weather_sh == 'overcast1':
            weather = world.get_weather()
            weather = carla.WeatherParameters(sun_azimuth_angle=90.0, sun_altitude_angle=35.0)
            apply_weather_presets(args.Weather_sh, weather)
            world.set_weather(weather)
            print(f'Weather set done\n')
        elif args.Weather_sh == 'overcast2':
            weather = world.get_weather()
            weather = carla.WeatherParameters(sun_azimuth_angle=120.0, sun_altitude_angle=145.0)
            apply_weather_presets(args.Weather_sh, weather)
            world.set_weather(weather)
            print(f'Weather set done\n')
        elif args.Weather_sh == 'rain':
            weather = world.get_weather()
            weather = carla.WeatherParameters(sun_azimuth_angle=150.0, sun_altitude_angle=135.0)
            apply_weather_presets(args.Weather_sh, weather)
            world.set_weather(weather)
            print(f'Weather set done\n')
        elif args.Weather_sh == 'rain1':
            weather = world.get_weather()
            weather = carla.WeatherParameters(sun_azimuth_angle=180.0, sun_altitude_angle=155.0)
            apply_weather_presets(args.Weather_sh, weather)
            world.set_weather(weather)
            print(f'Weather set done\n')
        elif args.Weather_sh == 'night':
            weather = world.get_weather()
            weather = carla.WeatherParameters(sun_azimuth_angle=180.0, sun_altitude_angle=-90.0)
            apply_weather_presets(args.Weather_sh, weather)
            world.set_weather(weather)
            print(f'Weather set done\n')
        elif args.Weather_sh == 'dawn':
            weather = world.get_weather()
            weather = carla.WeatherParameters(sun_azimuth_angle=180.0, sun_altitude_angle=10.0)
            apply_weather_presets(args.Weather_sh, weather)
            world.set_weather(weather)
            print(f'Weather set done\n')
        
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(0.9) #1.0 #distance length Curiosity 0.8 Indago Town05 Town03 0.9
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)


        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)
        
        spawn_points = world.get_map().get_spawn_points()
        bp_lib = world.get_blueprint_library().filter('sensor.camera.*')
        vehicle_bp = blueprints.find(args.Vehicle_type_sh)
        vehicle_sh = world.try_spawn_actor(vehicle_bp, spawn_points[int(args.SpawnP_sh)])
        vehicle_sh.set_autopilot(True)
        
        if record:
            # Spawn the camera into the world and attach it relative to the origin of the vehicle
            # Attached as the circumference of a circle from the center of the vehicle
            if args.Camera_set_sh == 'Baseline_RU':
                print('====RU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=1.24), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-1.24), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.64), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.64), carla.Rotation(yaw=90))

                camera_init_trans_RU1 = carla.Transform(carla.Location(z=1.762, x=(1.2337), y=(0.0647)), carla.Rotation(yaw=3.0)) # distance=(1.2353)
                camera_init_trans_RU2 = carla.Transform(carla.Location(z=1.762, x=(1.2151), y=(0.1277)), carla.Rotation(yaw=6.0)) # distance=(1.2218)
                camera_init_trans_RU3 = carla.Transform(carla.Location(z=1.762, x=(1.1854), y=(0.1878)), carla.Rotation(yaw=9.0)) # distance=(1.2002)
                camera_init_trans_RU4 = carla.Transform(carla.Location(z=1.762, x=(1.1466), y=(0.2437)), carla.Rotation(yaw=12.0)) # distance=(1.1722)
                camera_init_trans_RU5 = carla.Transform(carla.Location(z=1.762, x=(1.1005), y=(0.2949)), carla.Rotation(yaw=15.0)) # distance=(1.1394)
                camera_init_trans_RU6 = carla.Transform(carla.Location(z=1.762, x=(1.0494), y=(0.3410)), carla.Rotation(yaw=18.0)) # distance=(1.1034)
                camera_init_trans_RU7 = carla.Transform(carla.Location(z=1.762, x=(0.9950), y=(0.3819)), carla.Rotation(yaw=21.0)) # distance=(1.0658)
                camera_init_trans_RU8 = carla.Transform(carla.Location(z=1.762, x=(0.9389), y=(0.4180)), carla.Rotation(yaw=24.0)) # distance=(1.0278)
                camera_init_trans_RU9 = carla.Transform(carla.Location(z=1.762, x=(0.8824), y=(0.4496)), carla.Rotation(yaw=27.0)) # distance=(0.9904)
                camera_init_trans_RU10 = carla.Transform(carla.Location(z=1.762, x=(0.8264), y=(0.4771)), carla.Rotation(yaw=30.0)) # distance=(0.9543)
                camera_init_trans_RU11 = carla.Transform(carla.Location(z=1.762, x=(0.7715), y=(0.5010)), carla.Rotation(yaw=33.0)) # distance=(0.9199)
                camera_init_trans_RU12 = carla.Transform(carla.Location(z=1.762, x=(0.7181), y=(0.5218)), carla.Rotation(yaw=36.0)) # distance=(0.8877)
                camera_init_trans_RU13 = carla.Transform(carla.Location(z=1.762, x=(0.6665), y=(0.5397)), carla.Rotation(yaw=39.0)) # distance=(0.8576)
                camera_init_trans_RU14 = carla.Transform(carla.Location(z=1.762, x=(0.6167), y=(0.5552)), carla.Rotation(yaw=42.0)) # distance=(0.8298)
                camera_init_trans_RU15 = carla.Transform(carla.Location(z=1.762, x=(0.5687), y=(0.5687)), carla.Rotation(yaw=45.0)) # distance=(0.8043)
                camera_init_trans_RU16 = carla.Transform(carla.Location(z=1.762, x=(0.5226), y=(0.5804)), carla.Rotation(yaw=48.0)) # distance=(0.7810)
                camera_init_trans_RU17 = carla.Transform(carla.Location(z=1.762, x=(0.4782), y=(0.5905)), carla.Rotation(yaw=51.0)) # distance=(0.7598)
                camera_init_trans_RU18 = carla.Transform(carla.Location(z=1.762, x=(0.4354), y=(0.5993)), carla.Rotation(yaw=54.0)) # distance=(0.7407)
                camera_init_trans_RU19 = carla.Transform(carla.Location(z=1.762, x=(0.3941), y=(0.6068)), carla.Rotation(yaw=57.0)) # distance=(0.7235)
                camera_init_trans_RU20 = carla.Transform(carla.Location(z=1.762, x=(0.3541), y=(0.6133)), carla.Rotation(yaw=60.0)) # distance=(0.7082)
                camera_init_trans_RU21 = carla.Transform(carla.Location(z=1.762, x=(0.3154), y=(0.6190)), carla.Rotation(yaw=63.0)) # distance=(0.6947)
                camera_init_trans_RU22 = carla.Transform(carla.Location(z=1.762, x=(0.2777), y=(0.6237)), carla.Rotation(yaw=66.0)) # distance=(0.6828)
                camera_init_trans_RU23 = carla.Transform(carla.Location(z=1.762, x=(0.2410), y=(0.6278)), carla.Rotation(yaw=69.0)) # distance=(0.6725)
                camera_init_trans_RU24 = carla.Transform(carla.Location(z=1.762, x=(0.2051), y=(0.6312)), carla.Rotation(yaw=72.0)) # distance=(0.6637)
                camera_init_trans_RU25 = carla.Transform(carla.Location(z=1.762, x=(0.1699), y=(0.6340)), carla.Rotation(yaw=75.0)) # distance=(0.6563)
                camera_init_trans_RU26 = carla.Transform(carla.Location(z=1.762, x=(0.1352), y=(0.6362)), carla.Rotation(yaw=78.0)) # distance=(0.6504)
                camera_init_trans_RU27 = carla.Transform(carla.Location(z=1.762, x=(0.1010), y=(0.6379)), carla.Rotation(yaw=81.0)) # distance=(0.6458)
                camera_init_trans_RU28 = carla.Transform(carla.Location(z=1.762, x=(0.0672), y=(0.6391)), carla.Rotation(yaw=84.0)) # distance=(0.6426)
                camera_init_trans_RU29 = carla.Transform(carla.Location(z=1.762, x=(0.0335), y=(0.6398)), carla.Rotation(yaw=87.0)) # distance=(0.6406)
                
            elif args.Camera_set_sh == 'Baseline_LU':
                print('====LU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=1.24), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-1.24), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.64), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.64), carla.Rotation(yaw=90))

                camera_init_trans_LU29 = carla.Transform(carla.Location(z=1.762, x=(0.0335), y=(-0.6398)), carla.Rotation(yaw=-87.0)) # distance=(0.6406)
                camera_init_trans_LU28 = carla.Transform(carla.Location(z=1.762, x=(0.0672), y=(-0.6391)), carla.Rotation(yaw=-84.0)) # distance=(0.6426)
                camera_init_trans_LU27 = carla.Transform(carla.Location(z=1.762, x=(0.1010), y=(-0.6379)), carla.Rotation(yaw=-81.0)) # distance=(0.6458)
                camera_init_trans_LU26 = carla.Transform(carla.Location(z=1.762, x=(0.1352), y=(-0.6362)), carla.Rotation(yaw=-78.0)) # distance=(0.6504)
                camera_init_trans_LU25 = carla.Transform(carla.Location(z=1.762, x=(0.1699), y=(-0.6340)), carla.Rotation(yaw=-75.0)) # distance=(0.6563)
                camera_init_trans_LU24 = carla.Transform(carla.Location(z=1.762, x=(0.2051), y=(-0.6312)), carla.Rotation(yaw=-72.0)) # distance=(0.6637)
                camera_init_trans_LU23 = carla.Transform(carla.Location(z=1.762, x=(0.2410), y=(-0.6278)), carla.Rotation(yaw=-69.0)) # distance=(0.6725)
                camera_init_trans_LU22 = carla.Transform(carla.Location(z=1.762, x=(0.2777), y=(-0.6237)), carla.Rotation(yaw=-66.0)) # distance=(0.6828)
                camera_init_trans_LU21 = carla.Transform(carla.Location(z=1.762, x=(0.3154), y=(-0.6190)), carla.Rotation(yaw=-63.0)) # distance=(0.6947)
                camera_init_trans_LU20 = carla.Transform(carla.Location(z=1.762, x=(0.3541), y=(-0.6133)), carla.Rotation(yaw=-60.0)) # distance=(0.7082)
                camera_init_trans_LU19 = carla.Transform(carla.Location(z=1.762, x=(0.3941), y=(-0.6068)), carla.Rotation(yaw=-57.0)) # distance=(0.7235)
                camera_init_trans_LU18 = carla.Transform(carla.Location(z=1.762, x=(0.4354), y=(-0.5993)), carla.Rotation(yaw=-54.0)) # distance=(0.7407)
                camera_init_trans_LU17 = carla.Transform(carla.Location(z=1.762, x=(0.4782), y=(-0.5905)), carla.Rotation(yaw=-51.0)) # distance=(0.7598)
                camera_init_trans_LU16 = carla.Transform(carla.Location(z=1.762, x=(0.5226), y=(-0.5804)), carla.Rotation(yaw=-48.0)) # distance=(0.7810)
                camera_init_trans_LU15 = carla.Transform(carla.Location(z=1.762, x=(0.5687), y=(-0.5687)), carla.Rotation(yaw=-45.0)) # distance=(0.8043)
                camera_init_trans_LU14 = carla.Transform(carla.Location(z=1.762, x=(0.6167), y=(-0.5552)), carla.Rotation(yaw=-42.0)) # distance=(0.8298)
                camera_init_trans_LU13 = carla.Transform(carla.Location(z=1.762, x=(0.6665), y=(-0.5397)), carla.Rotation(yaw=-39.0)) # distance=(0.8576)
                camera_init_trans_LU12 = carla.Transform(carla.Location(z=1.762, x=(0.7181), y=(-0.5218)), carla.Rotation(yaw=-36.0)) # distance=(0.8877)
                camera_init_trans_LU11 = carla.Transform(carla.Location(z=1.762, x=(0.7715), y=(-0.5010)), carla.Rotation(yaw=-33.0)) # distance=(0.9199)
                camera_init_trans_LU10 = carla.Transform(carla.Location(z=1.762, x=(0.8264), y=(-0.4771)), carla.Rotation(yaw=-30.0)) # distance=(0.9543)
                camera_init_trans_LU9 = carla.Transform(carla.Location(z=1.762, x=(0.8824), y=(-0.4496)), carla.Rotation(yaw=-27.0)) # distance=(0.9904)
                camera_init_trans_LU8 = carla.Transform(carla.Location(z=1.762, x=(0.9389), y=(-0.4180)), carla.Rotation(yaw=-24.0)) # distance=(1.0278)
                camera_init_trans_LU7 = carla.Transform(carla.Location(z=1.762, x=(0.9950), y=(-0.3819)), carla.Rotation(yaw=-21.0)) # distance=(1.0658)
                camera_init_trans_LU6 = carla.Transform(carla.Location(z=1.762, x=(1.0494), y=(-0.3410)), carla.Rotation(yaw=-18.0)) # distance=(1.1034)
                camera_init_trans_LU5 = carla.Transform(carla.Location(z=1.762, x=(1.1005), y=(-0.2949)), carla.Rotation(yaw=-15.0)) # distance=(1.1394)
                camera_init_trans_LU4 = carla.Transform(carla.Location(z=1.762, x=(1.1466), y=(-0.2437)), carla.Rotation(yaw=-12.0)) # distance=(1.1722)
                camera_init_trans_LU3 = carla.Transform(carla.Location(z=1.762, x=(1.1854), y=(-0.1878)), carla.Rotation(yaw=-9.0)) # distance=(1.2002)
                camera_init_trans_LU2 = carla.Transform(carla.Location(z=1.762, x=(1.2151), y=(-0.1277)), carla.Rotation(yaw=-6.0)) # distance=(1.2218)
                camera_init_trans_LU1 = carla.Transform(carla.Location(z=1.762, x=(1.2337), y=(-0.0647)), carla.Rotation(yaw=-3.0)) # distance=(1.2353)
                
            elif args.Camera_set_sh == 'Baseline_RD':
                print('====RD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=1.24), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-1.24), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.64), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.64), carla.Rotation(yaw=90))

                camera_init_trans_RD1 = carla.Transform(carla.Location(z=1.762, x=(-0.0335), y=(0.6398)), carla.Rotation(yaw=93.0)) # distance=(0.6406)
                camera_init_trans_RD2 = carla.Transform(carla.Location(z=1.762, x=(-0.0672), y=(0.6391)), carla.Rotation(yaw=96.0)) # distance=(0.6426)
                camera_init_trans_RD3 = carla.Transform(carla.Location(z=1.762, x=(-0.1010), y=(0.6379)), carla.Rotation(yaw=99.0)) # distance=(0.6458)
                camera_init_trans_RD4 = carla.Transform(carla.Location(z=1.762, x=(-0.1352), y=(0.6362)), carla.Rotation(yaw=102.0)) # distance=(0.6504)
                camera_init_trans_RD5 = carla.Transform(carla.Location(z=1.762, x=(-0.1699), y=(0.6340)), carla.Rotation(yaw=105.0)) # distance=(0.6563)
                camera_init_trans_RD6 = carla.Transform(carla.Location(z=1.762, x=(-0.2051), y=(0.6312)), carla.Rotation(yaw=108.0)) # distance=(0.6637)
                camera_init_trans_RD7 = carla.Transform(carla.Location(z=1.762, x=(-0.2410), y=(0.6278)), carla.Rotation(yaw=111.0)) # distance=(0.6725)
                camera_init_trans_RD8 = carla.Transform(carla.Location(z=1.762, x=(-0.2777), y=(0.6237)), carla.Rotation(yaw=114.0)) # distance=(0.6828)
                camera_init_trans_RD9 = carla.Transform(carla.Location(z=1.762, x=(-0.3154), y=(0.6190)), carla.Rotation(yaw=117.0)) # distance=(0.6947)
                camera_init_trans_RD10 = carla.Transform(carla.Location(z=1.762, x=(-0.3541), y=(0.6133)), carla.Rotation(yaw=120.0)) # distance=(0.7082)
                camera_init_trans_RD11 = carla.Transform(carla.Location(z=1.762, x=(-0.3941), y=(0.6068)), carla.Rotation(yaw=123.0)) # distance=(0.7236)
                camera_init_trans_RD12 = carla.Transform(carla.Location(z=1.762, x=(-0.4354), y=(0.5993)), carla.Rotation(yaw=126.0)) # distance=(0.7407)
                camera_init_trans_RD13 = carla.Transform(carla.Location(z=1.762, x=(-0.4782), y=(0.5905)), carla.Rotation(yaw=129.0)) # distance=(0.7598)
                camera_init_trans_RD14 = carla.Transform(carla.Location(z=1.762, x=(-0.5226), y=(0.5804)), carla.Rotation(yaw=132.0)) # distance=(0.7810)
                camera_init_trans_RD15 = carla.Transform(carla.Location(z=1.762, x=(-0.5687), y=(0.5687)), carla.Rotation(yaw=135.0)) # distance=(0.8043)
                camera_init_trans_RD16 = carla.Transform(carla.Location(z=1.762, x=(-0.6167), y=(0.5552)), carla.Rotation(yaw=138.0)) # distance=(0.8298)
                camera_init_trans_RD17 = carla.Transform(carla.Location(z=1.762, x=(-0.6665), y=(0.5397)), carla.Rotation(yaw=141.0)) # distance=(0.8576)
                camera_init_trans_RD18 = carla.Transform(carla.Location(z=1.762, x=(-0.7181), y=(0.5217)), carla.Rotation(yaw=144.0)) # distance=(0.8877)
                camera_init_trans_RD19 = carla.Transform(carla.Location(z=1.762, x=(-0.7715), y=(0.5010)), carla.Rotation(yaw=147.0)) # distance=(0.9199)
                camera_init_trans_RD20 = carla.Transform(carla.Location(z=1.762, x=(-0.8264), y=(0.4771)), carla.Rotation(yaw=150.0)) # distance=(0.9543)
                camera_init_trans_RD21 = carla.Transform(carla.Location(z=1.762, x=(-0.8824), y=(0.4496)), carla.Rotation(yaw=153.0)) # distance=(0.9904)
                camera_init_trans_RD22 = carla.Transform(carla.Location(z=1.762, x=(-0.9389), y=(0.4180)), carla.Rotation(yaw=156.0)) # distance=(1.0278)
                camera_init_trans_RD23 = carla.Transform(carla.Location(z=1.762, x=(-0.9950), y=(0.3819)), carla.Rotation(yaw=159.0)) # distance=(1.0658)
                camera_init_trans_RD24 = carla.Transform(carla.Location(z=1.762, x=(-1.0494), y=(0.3410)), carla.Rotation(yaw=162.0)) # distance=(1.1034)
                camera_init_trans_RD25 = carla.Transform(carla.Location(z=1.762, x=(-1.1005), y=(0.2949)), carla.Rotation(yaw=165.0)) # distance=(1.1394)
                camera_init_trans_RD26 = carla.Transform(carla.Location(z=1.762, x=(-1.1466), y=(0.2437)), carla.Rotation(yaw=168.0)) # distance=(1.1722)
                camera_init_trans_RD27 = carla.Transform(carla.Location(z=1.762, x=(-1.1854), y=(0.1878)), carla.Rotation(yaw=171.0)) # distance=(1.2002)
                camera_init_trans_RD28 = carla.Transform(carla.Location(z=1.762, x=(-1.2151), y=(0.1277)), carla.Rotation(yaw=174.0)) # distance=(1.2218)
                camera_init_trans_RD29 = carla.Transform(carla.Location(z=1.762, x=(-1.2337), y=(0.0647)), carla.Rotation(yaw=177.0)) # distance=(1.2353)
                
            elif args.Camera_set_sh == 'Baseline_LD':
                print('====LD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=1.24), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-1.24), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.64), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.64), carla.Rotation(yaw=90))

                camera_init_trans_LD29 = carla.Transform(carla.Location(z=1.762, x=(-1.2337), y=(-0.0647)), carla.Rotation(yaw=-177.0)) # distance=(1.2353)
                camera_init_trans_LD28 = carla.Transform(carla.Location(z=1.762, x=(-1.2151), y=(-0.1277)), carla.Rotation(yaw=-174.0)) # distance=(1.2218)
                camera_init_trans_LD27 = carla.Transform(carla.Location(z=1.762, x=(-1.1854), y=(-0.1877)), carla.Rotation(yaw=-171.0)) # distance=(1.2002)
                camera_init_trans_LD26 = carla.Transform(carla.Location(z=1.762, x=(-1.1466), y=(-0.2437)), carla.Rotation(yaw=-168.0)) # distance=(1.1722)
                camera_init_trans_LD25 = carla.Transform(carla.Location(z=1.762, x=(-1.1005), y=(-0.2949)), carla.Rotation(yaw=-165.0)) # distance=(1.1394)
                camera_init_trans_LD24 = carla.Transform(carla.Location(z=1.762, x=(-1.0494), y=(-0.3410)), carla.Rotation(yaw=-162.0)) # distance=(1.1034)
                camera_init_trans_LD23 = carla.Transform(carla.Location(z=1.762, x=(-0.9950), y=(-0.3819)), carla.Rotation(yaw=-159.0)) # distance=(1.0658)
                camera_init_trans_LD22 = carla.Transform(carla.Location(z=1.762, x=(-0.9389), y=(-0.4180)), carla.Rotation(yaw=-156.0)) # distance=(1.0278)
                camera_init_trans_LD21 = carla.Transform(carla.Location(z=1.762, x=(-0.8825), y=(-0.4496)), carla.Rotation(yaw=-153.0)) # distance=(0.9904)
                camera_init_trans_LD20 = carla.Transform(carla.Location(z=1.762, x=(-0.8264), y=(-0.4771)), carla.Rotation(yaw=-150.0)) # distance=(0.9543)
                camera_init_trans_LD19 = carla.Transform(carla.Location(z=1.762, x=(-0.7715), y=(-0.5010)), carla.Rotation(yaw=-147.0)) # distance=(0.9199)
                camera_init_trans_LD18 = carla.Transform(carla.Location(z=1.762, x=(-0.7181), y=(-0.5217)), carla.Rotation(yaw=-144.0)) # distance=(0.8877)
                camera_init_trans_LD17 = carla.Transform(carla.Location(z=1.762, x=(-0.6665), y=(-0.5397)), carla.Rotation(yaw=-141.0)) # distance=(0.8576)
                camera_init_trans_LD16 = carla.Transform(carla.Location(z=1.762, x=(-0.6167), y=(-0.5552)), carla.Rotation(yaw=-138.0)) # distance=(0.8298)
                camera_init_trans_LD15 = carla.Transform(carla.Location(z=1.762, x=(-0.5687), y=(-0.5687)), carla.Rotation(yaw=-135.0)) # distance=(0.8043)
                camera_init_trans_LD14 = carla.Transform(carla.Location(z=1.762, x=(-0.5226), y=(-0.5804)), carla.Rotation(yaw=-132.0)) # distance=(0.7810)
                camera_init_trans_LD13 = carla.Transform(carla.Location(z=1.762, x=(-0.4782), y=(-0.5905)), carla.Rotation(yaw=-129.0)) # distance=(0.7598)
                camera_init_trans_LD12 = carla.Transform(carla.Location(z=1.762, x=(-0.4354), y=(-0.5993)), carla.Rotation(yaw=-126.0)) # distance=(0.7407)
                camera_init_trans_LD11 = carla.Transform(carla.Location(z=1.762, x=(-0.3941), y=(-0.6068)), carla.Rotation(yaw=-123.0)) # distance=(0.7236)
                camera_init_trans_LD10 = carla.Transform(carla.Location(z=1.762, x=(-0.3541), y=(-0.6133)), carla.Rotation(yaw=-120.0)) # distance=(0.7082)
                camera_init_trans_LD9 = carla.Transform(carla.Location(z=1.762, x=(-0.3154), y=(-0.6190)), carla.Rotation(yaw=-117.0)) # distance=(0.6947)
                camera_init_trans_LD8 = carla.Transform(carla.Location(z=1.762, x=(-0.2777), y=(-0.6237)), carla.Rotation(yaw=-114.0)) # distance=(0.6828)
                camera_init_trans_LD7 = carla.Transform(carla.Location(z=1.762, x=(-0.2410), y=(-0.6278)), carla.Rotation(yaw=-111.0)) # distance=(0.6725)
                camera_init_trans_LD6 = carla.Transform(carla.Location(z=1.762, x=(-0.2051), y=(-0.6312)), carla.Rotation(yaw=-108.0)) # distance=(0.6637)
                camera_init_trans_LD5 = carla.Transform(carla.Location(z=1.762, x=(-0.1699), y=(-0.6340)), carla.Rotation(yaw=-105.0)) # distance=(0.6563)
                camera_init_trans_LD4 = carla.Transform(carla.Location(z=1.762, x=(-0.1352), y=(-0.6362)), carla.Rotation(yaw=-102.0)) # distance=(0.6504)
                camera_init_trans_LD3 = carla.Transform(carla.Location(z=1.762, x=(-0.1010), y=(-0.6379)), carla.Rotation(yaw=-99.0)) # distance=(0.6458)
                camera_init_trans_LD2 = carla.Transform(carla.Location(z=1.762, x=(-0.0672), y=(-0.6391)), carla.Rotation(yaw=-96.0)) # distance=(0.6426)
                camera_init_trans_LD1 = carla.Transform(carla.Location(z=1.762, x=(-0.0335), y=(-0.6398)), carla.Rotation(yaw=-93.0)) # distance=(0.6406)
                
            elif args.Camera_set_sh == 'Pluto_plus_RU':
                print('====RU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=2.5, x=0.95), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=2.5, x=-0.95), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=2.5, x=0.0, y=-0.35), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=2.5, x=(0.0), y=0.35), carla.Rotation(yaw=90))

                camera_init_trans_RU1 = carla.Transform(carla.Location(z=2.5, x=(0.9405), y=(0.0493)), carla.Rotation(yaw=3.0000)) # distance=(0.9418)
                camera_init_trans_RU2 = carla.Transform(carla.Location(z=2.5, x=(0.9136), y=(0.0960)), carla.Rotation(yaw=6.0000)) # distance=(0.9186)
                camera_init_trans_RU3 = carla.Transform(carla.Location(z=2.5, x=(0.8728), y=(0.1382)), carla.Rotation(yaw=9.0000)) # distance=(0.8836)
                camera_init_trans_RU4 = carla.Transform(carla.Location(z=2.5, x=(0.8229), y=(0.1749)), carla.Rotation(yaw=12.0000)) # distance=(0.8413)
                camera_init_trans_RU5 = carla.Transform(carla.Location(z=2.5, x=(0.7683), y=(0.2059)), carla.Rotation(yaw=15.0000)) # distance=(0.7954)
                camera_init_trans_RU6 = carla.Transform(carla.Location(z=2.5, x=(0.7125), y=(0.2315)), carla.Rotation(yaw=18.0001)) # distance=(0.7492)
                camera_init_trans_RU7 = carla.Transform(carla.Location(z=2.5, x=(0.6578), y=(0.2525)), carla.Rotation(yaw=21.0001)) # distance=(0.7046)
                camera_init_trans_RU8 = carla.Transform(carla.Location(z=2.5, x=(0.6056), y=(0.2697)), carla.Rotation(yaw=24.0001)) # distance=(0.6630)
                camera_init_trans_RU9 = carla.Transform(carla.Location(z=2.5, x=(0.5566), y=(0.2836)), carla.Rotation(yaw=27.0001)) # distance=(0.6247)
                camera_init_trans_RU10 = carla.Transform(carla.Location(z=2.5, x=(0.5110), y=(0.2950)), carla.Rotation(yaw=30.0000)) # distance=(0.5901)
                camera_init_trans_RU11 = carla.Transform(carla.Location(z=2.5, x=(0.4688), y=(0.3044)), carla.Rotation(yaw=33.0003)) # distance=(0.5589)
                camera_init_trans_RU12 = carla.Transform(carla.Location(z=2.5, x=(0.4296), y=(0.3122)), carla.Rotation(yaw=36.0003)) # distance=(0.5311)
                camera_init_trans_RU13 = carla.Transform(carla.Location(z=2.5, x=(0.3934), y=(0.3186)), carla.Rotation(yaw=39.0004)) # distance=(0.5062)
                camera_init_trans_RU14 = carla.Transform(carla.Location(z=2.5, x=(0.3598), y=(0.3239)), carla.Rotation(yaw=42.0000)) # distance=(0.4841)
                camera_init_trans_RU15 = carla.Transform(carla.Location(z=2.5, x=(0.3284), y=(0.3284)), carla.Rotation(yaw=45.0001)) # distance=(0.4645)
                camera_init_trans_RU16 = carla.Transform(carla.Location(z=2.5, x=(0.2991), y=(0.3322)), carla.Rotation(yaw=48.0001)) # distance=(0.4470)
                camera_init_trans_RU17 = carla.Transform(carla.Location(z=2.5, x=(0.2716), y=(0.3354)), carla.Rotation(yaw=51.0003)) # distance=(0.4316)
                camera_init_trans_RU18 = carla.Transform(carla.Location(z=2.5, x=(0.2456), y=(0.3381)), carla.Rotation(yaw=54.0005)) # distance=(0.4179)
                camera_init_trans_RU19 = carla.Transform(carla.Location(z=2.5, x=(0.2211), y=(0.3404)), carla.Rotation(yaw=57.0001)) # distance=(0.4059)
                camera_init_trans_RU20 = carla.Transform(carla.Location(z=2.5, x=(0.1976), y=(0.3423)), carla.Rotation(yaw=60.0003)) # distance=(0.3953)
                camera_init_trans_RU21 = carla.Transform(carla.Location(z=2.5, x=(0.1753), y=(0.3440)), carla.Rotation(yaw=63.0007)) # distance=(0.3861)
                camera_init_trans_RU22 = carla.Transform(carla.Location(z=2.5, x=(0.1538), y=(0.3454)), carla.Rotation(yaw=66.0000)) # distance=(0.3781)
                camera_init_trans_RU23 = carla.Transform(carla.Location(z=2.5, x=(0.1330), y=(0.3466)), carla.Rotation(yaw=69.0000)) # distance=(0.3712)
                camera_init_trans_RU24 = carla.Transform(carla.Location(z=2.5, x=(0.1129), y=(0.3475)), carla.Rotation(yaw=72.0007)) # distance=(0.3654)
                camera_init_trans_RU25 = carla.Transform(carla.Location(z=2.5, x=(0.0933), y=(0.3483)), carla.Rotation(yaw=75.0007)) # distance=(0.3606)
                camera_init_trans_RU26 = carla.Transform(carla.Location(z=2.5, x=(0.0742), y=(0.3489)), carla.Rotation(yaw=78.0001)) # distance=(0.3567)
                camera_init_trans_RU27 = carla.Transform(carla.Location(z=2.5, x=(0.0553), y=(0.3494)), carla.Rotation(yaw=81.0005)) # distance=(0.3538)
                camera_init_trans_RU28 = carla.Transform(carla.Location(z=2.5, x=(0.0368), y=(0.3497)), carla.Rotation(yaw=84.0008)) # distance=(0.3517)
                camera_init_trans_RU29 = carla.Transform(carla.Location(z=2.5, x=(0.0183), y=(0.3499)), carla.Rotation(yaw=87.0006)) # distance=(0.3504)

            elif args.Camera_set_sh == 'Pluto_plus_LU':
                print('====LU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=2.5, x=0.95), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=2.5, x=-0.95), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=2.5, x=0.0, y=-0.35), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=2.5, x=(0.0), y=0.35), carla.Rotation(yaw=90))
                
                camera_init_trans_LU1 = carla.Transform(carla.Location(z=2.5, x=(0.9405), y=(-0.0493)), carla.Rotation(yaw=-3.0000)) # distance=(0.9418)
                camera_init_trans_LU2 = carla.Transform(carla.Location(z=2.5, x=(0.9136), y=(-0.0960)), carla.Rotation(yaw=-6.0000)) # distance=(0.9186)
                camera_init_trans_LU3 = carla.Transform(carla.Location(z=2.5, x=(0.8728), y=(-0.1382)), carla.Rotation(yaw=-9.0000)) # distance=(0.8836)
                camera_init_trans_LU4 = carla.Transform(carla.Location(z=2.5, x=(0.8229), y=(-0.1749)), carla.Rotation(yaw=-12.0000)) # distance=(0.8413)
                camera_init_trans_LU5 = carla.Transform(carla.Location(z=2.5, x=(0.7683), y=(-0.2059)), carla.Rotation(yaw=-15.0000)) # distance=(0.7954)
                camera_init_trans_LU6 = carla.Transform(carla.Location(z=2.5, x=(0.7125), y=(-0.2315)), carla.Rotation(yaw=-18.0001)) # distance=(0.7492)
                camera_init_trans_LU7 = carla.Transform(carla.Location(z=2.5, x=(0.6578), y=(-0.2525)), carla.Rotation(yaw=-21.0001)) # distance=(0.7046)
                camera_init_trans_LU8 = carla.Transform(carla.Location(z=2.5, x=(0.6056), y=(-0.2697)), carla.Rotation(yaw=-24.0001)) # distance=(0.6630)
                camera_init_trans_LU9 = carla.Transform(carla.Location(z=2.5, x=(0.5566), y=(-0.2836)), carla.Rotation(yaw=-27.0001)) # distance=(0.6247)
                camera_init_trans_LU10 = carla.Transform(carla.Location(z=2.5, x=(0.5110), y=(-0.2950)), carla.Rotation(yaw=-30.0000)) # distance=(0.5901)
                camera_init_trans_LU11 = carla.Transform(carla.Location(z=2.5, x=(0.4688), y=(-0.3044)), carla.Rotation(yaw=-33.0003)) # distance=(0.5589)
                camera_init_trans_LU12 = carla.Transform(carla.Location(z=2.5, x=(0.4296), y=(-0.3122)), carla.Rotation(yaw=-36.0003)) # distance=(0.5311)
                camera_init_trans_LU13 = carla.Transform(carla.Location(z=2.5, x=(0.3934), y=(-0.3186)), carla.Rotation(yaw=-39.0004)) # distance=(0.5062)
                camera_init_trans_LU14 = carla.Transform(carla.Location(z=2.5, x=(0.3598), y=(-0.3239)), carla.Rotation(yaw=-42.0000)) # distance=(0.4841)
                camera_init_trans_LU15 = carla.Transform(carla.Location(z=2.5, x=(0.3284), y=(-0.3284)), carla.Rotation(yaw=-45.0001)) # distance=(0.4645)
                camera_init_trans_LU16 = carla.Transform(carla.Location(z=2.5, x=(0.2991), y=(-0.3322)), carla.Rotation(yaw=-48.0001)) # distance=(0.4470)
                camera_init_trans_LU17 = carla.Transform(carla.Location(z=2.5, x=(0.2716), y=(-0.3354)), carla.Rotation(yaw=-51.0003)) # distance=(0.4316)
                camera_init_trans_LU18 = carla.Transform(carla.Location(z=2.5, x=(0.2456), y=(-0.3381)), carla.Rotation(yaw=-54.0005)) # distance=(0.4179)
                camera_init_trans_LU19 = carla.Transform(carla.Location(z=2.5, x=(0.2211), y=(-0.3404)), carla.Rotation(yaw=-57.0001)) # distance=(0.4059)
                camera_init_trans_LU20 = carla.Transform(carla.Location(z=2.5, x=(0.1976), y=(-0.3423)), carla.Rotation(yaw=-60.0003)) # distance=(0.3953)
                camera_init_trans_LU21 = carla.Transform(carla.Location(z=2.5, x=(0.1753), y=(-0.3440)), carla.Rotation(yaw=-63.0007)) # distance=(0.3861)
                camera_init_trans_LU22 = carla.Transform(carla.Location(z=2.5, x=(0.1538), y=(-0.3454)), carla.Rotation(yaw=-66.0000)) # distance=(0.3781)
                camera_init_trans_LU23 = carla.Transform(carla.Location(z=2.5, x=(0.1330), y=(-0.3466)), carla.Rotation(yaw=-69.0000)) # distance=(0.3712)
                camera_init_trans_LU24 = carla.Transform(carla.Location(z=2.5, x=(0.1129), y=(-0.3475)), carla.Rotation(yaw=-72.0007)) # distance=(0.3654)
                camera_init_trans_LU25 = carla.Transform(carla.Location(z=2.5, x=(0.0933), y=(-0.3483)), carla.Rotation(yaw=-75.0007)) # distance=(0.3606)
                camera_init_trans_LU26 = carla.Transform(carla.Location(z=2.5, x=(0.0742), y=(-0.3489)), carla.Rotation(yaw=-78.0001)) # distance=(0.3567)
                camera_init_trans_LU27 = carla.Transform(carla.Location(z=2.5, x=(0.0553), y=(-0.3494)), carla.Rotation(yaw=-81.0005)) # distance=(0.3538)
                camera_init_trans_LU28 = carla.Transform(carla.Location(z=2.5, x=(0.0368), y=(-0.3497)), carla.Rotation(yaw=-84.0008)) # distance=(0.3517)
                camera_init_trans_LU29 = carla.Transform(carla.Location(z=2.5, x=(0.0183), y=(-0.3499)), carla.Rotation(yaw=-87.0006)) # distance=(0.3504)
                
            elif args.Camera_set_sh == 'Pluto_plus_RD':
                print('====RD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=2.5, x=0.95), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=2.5, x=-0.95), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=2.5, x=0.0, y=-0.35), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=2.5, x=(0.0), y=0.35), carla.Rotation(yaw=90))
                
                camera_init_trans_RD1 = carla.Transform(carla.Location(z=2.5, x=(-0.0183), y=(0.3499)), carla.Rotation(yaw=93.0004)) # distance=(0.3504)
                camera_init_trans_RD2 = carla.Transform(carla.Location(z=2.5, x=(-0.0368), y=(0.3497)), carla.Rotation(yaw=96.0002)) # distance=(0.3517)
                camera_init_trans_RD3 = carla.Transform(carla.Location(z=2.5, x=(-0.0553), y=(0.3494)), carla.Rotation(yaw=99.0005)) # distance=(0.3538)
                camera_init_trans_RD4 = carla.Transform(carla.Location(z=2.5, x=(-0.0742), y=(0.3489)), carla.Rotation(yaw=102.0009)) # distance=(0.3567)
                camera_init_trans_RD5 = carla.Transform(carla.Location(z=2.5, x=(-0.0933), y=(0.3483)), carla.Rotation(yaw=105.0002)) # distance=(0.3606)
                camera_init_trans_RD6 = carla.Transform(carla.Location(z=2.5, x=(-0.1129), y=(0.3475)), carla.Rotation(yaw=108.0002)) # distance=(0.3654)
                camera_init_trans_RD7 = carla.Transform(carla.Location(z=2.5, x=(-0.1330), y=(0.3466)), carla.Rotation(yaw=111.0000)) # distance=(0.3712)
                camera_init_trans_RD8 = carla.Transform(carla.Location(z=2.5, x=(-0.1538), y=(0.3454)), carla.Rotation(yaw=114.0000)) # distance=(0.3781)
                camera_init_trans_RD9 = carla.Transform(carla.Location(z=2.5, x=(-0.1753), y=(0.3440)), carla.Rotation(yaw=117.0001)) # distance=(0.3861)
                camera_init_trans_RD10 = carla.Transform(carla.Location(z=2.5, x=(-0.1977), y=(0.3423)), carla.Rotation(yaw=120.0005)) # distance=(0.3953)
                camera_init_trans_RD11 = carla.Transform(carla.Location(z=2.5, x=(-0.2211), y=(0.3404)), carla.Rotation(yaw=123.0007)) # distance=(0.4059)
                camera_init_trans_RD12 = carla.Transform(carla.Location(z=2.5, x=(-0.2456), y=(0.3381)), carla.Rotation(yaw=126.0002)) # distance=(0.4179)
                camera_init_trans_RD13 = carla.Transform(carla.Location(z=2.5, x=(-0.2716), y=(0.3354)), carla.Rotation(yaw=129.0003)) # distance=(0.4316)
                camera_init_trans_RD14 = carla.Transform(carla.Location(z=2.5, x=(-0.2991), y=(0.3322)), carla.Rotation(yaw=132.0005)) # distance=(0.4470)
                camera_init_trans_RD15 = carla.Transform(carla.Location(z=2.5, x=(-0.3284), y=(0.3284)), carla.Rotation(yaw=135.0005)) # distance=(0.4645)
                camera_init_trans_RD16 = carla.Transform(carla.Location(z=2.5, x=(-0.3598), y=(0.3239)), carla.Rotation(yaw=138.0000)) # distance=(0.4841)
                camera_init_trans_RD17 = carla.Transform(carla.Location(z=2.5, x=(-0.3934), y=(0.3186)), carla.Rotation(yaw=141.0001)) # distance=(0.5062)
                camera_init_trans_RD18 = carla.Transform(carla.Location(z=2.5, x=(-0.4297), y=(0.3122)), carla.Rotation(yaw=144.0001)) # distance=(0.5311)
                camera_init_trans_RD19 = carla.Transform(carla.Location(z=2.5, x=(-0.4688), y=(0.3044)), carla.Rotation(yaw=147.0001)) # distance=(0.5589)
                camera_init_trans_RD20 = carla.Transform(carla.Location(z=2.5, x=(-0.5110), y=(0.2950)), carla.Rotation(yaw=150.0000)) # distance=(0.5901)
                camera_init_trans_RD21 = carla.Transform(carla.Location(z=2.5, x=(-0.5566), y=(0.2836)), carla.Rotation(yaw=153.0002)) # distance=(0.6247)
                camera_init_trans_RD22 = carla.Transform(carla.Location(z=2.5, x=(-0.6056), y=(0.2697)), carla.Rotation(yaw=156.0002)) # distance=(0.6630)
                camera_init_trans_RD23 = carla.Transform(carla.Location(z=2.5, x=(-0.6578), y=(0.2525)), carla.Rotation(yaw=159.0002)) # distance=(0.7046)
                camera_init_trans_RD24 = carla.Transform(carla.Location(z=2.5, x=(-0.7125), y=(0.2315)), carla.Rotation(yaw=162.0001)) # distance=(0.7492)
                camera_init_trans_RD25 = carla.Transform(carla.Location(z=2.5, x=(-0.7683), y=(0.2059)), carla.Rotation(yaw=165.0000)) # distance=(0.7954)
                camera_init_trans_RD26 = carla.Transform(carla.Location(z=2.5, x=(-0.8229), y=(0.1749)), carla.Rotation(yaw=168.0000)) # distance=(0.8413)
                camera_init_trans_RD27 = carla.Transform(carla.Location(z=2.5, x=(-0.8728), y=(0.1382)), carla.Rotation(yaw=171.0000)) # distance=(0.8836)
                camera_init_trans_RD28 = carla.Transform(carla.Location(z=2.5, x=(-0.9136), y=(0.0960)), carla.Rotation(yaw=174.0000)) # distance=(0.9186)
                camera_init_trans_RD29 = carla.Transform(carla.Location(z=2.5, x=(-0.9405), y=(0.0493)), carla.Rotation(yaw=177.0000)) # distance=(0.9418)
                
            elif args.Camera_set_sh == 'Pluto_plus_LD':
                print('====LD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=2.5, x=0.95), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=2.5, x=-0.95), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=2.5, x=0.0, y=-0.35), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=2.5, x=(0.0), y=0.35), carla.Rotation(yaw=90))
                
                camera_init_trans_LD1 = carla.Transform(carla.Location(z=2.5, x=(-0.0183), y=(-0.3499)), carla.Rotation(yaw=-93.0004)) # distance=(0.3504)
                camera_init_trans_LD2 = carla.Transform(carla.Location(z=2.5, x=(-0.0368), y=(-0.3497)), carla.Rotation(yaw=-96.0002)) # distance=(0.3517)
                camera_init_trans_LD3 = carla.Transform(carla.Location(z=2.5, x=(-0.0553), y=(-0.3494)), carla.Rotation(yaw=-99.0005)) # distance=(0.3538)
                camera_init_trans_LD4 = carla.Transform(carla.Location(z=2.5, x=(-0.0742), y=(-0.3489)), carla.Rotation(yaw=-102.0009)) # distance=(0.3567)
                camera_init_trans_LD5 = carla.Transform(carla.Location(z=2.5, x=(-0.0933), y=(-0.3483)), carla.Rotation(yaw=-105.0002)) # distance=(0.3606)
                camera_init_trans_LD6 = carla.Transform(carla.Location(z=2.5, x=(-0.1129), y=(-0.3475)), carla.Rotation(yaw=-108.0002)) # distance=(0.3654)
                camera_init_trans_LD7 = carla.Transform(carla.Location(z=2.5, x=(-0.1330), y=(-0.3466)), carla.Rotation(yaw=-111.0000)) # distance=(0.3712)
                camera_init_trans_LD8 = carla.Transform(carla.Location(z=2.5, x=(-0.1538), y=(-0.3454)), carla.Rotation(yaw=-114.0000)) # distance=(0.3781)
                camera_init_trans_LD9 = carla.Transform(carla.Location(z=2.5, x=(-0.1753), y=(-0.3440)), carla.Rotation(yaw=-117.0001)) # distance=(0.3861)
                camera_init_trans_LD10 = carla.Transform(carla.Location(z=2.5, x=(-0.1977), y=(-0.3423)), carla.Rotation(yaw=-120.0005)) # distance=(0.3953)
                camera_init_trans_LD11 = carla.Transform(carla.Location(z=2.5, x=(-0.2211), y=(-0.3404)), carla.Rotation(yaw=-123.0007)) # distance=(0.4059)
                camera_init_trans_LD12 = carla.Transform(carla.Location(z=2.5, x=(-0.2456), y=(-0.3381)), carla.Rotation(yaw=-126.0002)) # distance=(0.4179)
                camera_init_trans_LD13 = carla.Transform(carla.Location(z=2.5, x=(-0.2716), y=(-0.3354)), carla.Rotation(yaw=-129.0003)) # distance=(0.4316)
                camera_init_trans_LD14 = carla.Transform(carla.Location(z=2.5, x=(-0.2991), y=(-0.3322)), carla.Rotation(yaw=-132.0005)) # distance=(0.4470)
                camera_init_trans_LD15 = carla.Transform(carla.Location(z=2.5, x=(-0.3284), y=(-0.3284)), carla.Rotation(yaw=-135.0005)) # distance=(0.4645)
                camera_init_trans_LD16 = carla.Transform(carla.Location(z=2.5, x=(-0.3598), y=(-0.3239)), carla.Rotation(yaw=-138.0000)) # distance=(0.4841)
                camera_init_trans_LD17 = carla.Transform(carla.Location(z=2.5, x=(-0.3934), y=(-0.3186)), carla.Rotation(yaw=-141.0001)) # distance=(0.5062)
                camera_init_trans_LD18 = carla.Transform(carla.Location(z=2.5, x=(-0.4297), y=(-0.3122)), carla.Rotation(yaw=-144.0001)) # distance=(0.5311)
                camera_init_trans_LD19 = carla.Transform(carla.Location(z=2.5, x=(-0.4688), y=(-0.3044)), carla.Rotation(yaw=-147.0001)) # distance=(0.5589)
                camera_init_trans_LD20 = carla.Transform(carla.Location(z=2.5, x=(-0.5110), y=(-0.2950)), carla.Rotation(yaw=-150.0000)) # distance=(0.5901)
                camera_init_trans_LD21 = carla.Transform(carla.Location(z=2.5, x=(-0.5566), y=(-0.2836)), carla.Rotation(yaw=-153.0002)) # distance=(0.6247)
                camera_init_trans_LD22 = carla.Transform(carla.Location(z=2.5, x=(-0.6056), y=(-0.2697)), carla.Rotation(yaw=-156.0002)) # distance=(0.6630)
                camera_init_trans_LD23 = carla.Transform(carla.Location(z=2.5, x=(-0.6578), y=(-0.2525)), carla.Rotation(yaw=-159.0002)) # distance=(0.7046)
                camera_init_trans_LD24 = carla.Transform(carla.Location(z=2.5, x=(-0.7125), y=(-0.2315)), carla.Rotation(yaw=-162.0001)) # distance=(0.7492)
                camera_init_trans_LD25 = carla.Transform(carla.Location(z=2.5, x=(-0.7683), y=(-0.2059)), carla.Rotation(yaw=-165.0000)) # distance=(0.7954)
                camera_init_trans_LD26 = carla.Transform(carla.Location(z=2.5, x=(-0.8229), y=(-0.1749)), carla.Rotation(yaw=-168.0000)) # distance=(0.8413)
                camera_init_trans_LD27 = carla.Transform(carla.Location(z=2.5, x=(-0.8728), y=(-0.1382)), carla.Rotation(yaw=-171.0000)) # distance=(0.8836)
                camera_init_trans_LD28 = carla.Transform(carla.Location(z=2.5, x=(-0.9136), y=(-0.0960)), carla.Rotation(yaw=-174.0000)) # distance=(0.9186)
                camera_init_trans_LD29 = carla.Transform(carla.Location(z=2.5, x=(-0.9405), y=(-0.0493)), carla.Rotation(yaw=-177.0000)) # distance=(0.9418)

            elif args.Camera_set_sh == 'Indago3_RU':
                print('====RU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=0.55), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-0.55), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.56), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.56), carla.Rotation(yaw=90))

                camera_init_trans_RU1 = carla.Transform(carla.Location(z=1.762, x=(0.5493), y=(0.0288)), carla.Rotation(yaw=3.0001)) # distance=(0.5500)
                camera_init_trans_RU2 = carla.Transform(carla.Location(z=1.762, x=(0.5471), y=(0.0575)), carla.Rotation(yaw=6.0003)) # distance=(0.5501)
                camera_init_trans_RU3 = carla.Transform(carla.Location(z=1.762, x=(0.5435), y=(0.0861)), carla.Rotation(yaw=9.0001)) # distance=(0.5502)
                camera_init_trans_RU4 = carla.Transform(carla.Location(z=1.762, x=(0.5384), y=(0.1144)), carla.Rotation(yaw=12.0000)) # distance=(0.5504)
                camera_init_trans_RU5 = carla.Transform(carla.Location(z=1.762, x=(0.5319), y=(0.1425)), carla.Rotation(yaw=15.0003)) # distance=(0.5507)
                camera_init_trans_RU6 = carla.Transform(carla.Location(z=1.762, x=(0.5240), y=(0.1702)), carla.Rotation(yaw=18.0002)) # distance=(0.5509)
                camera_init_trans_RU7 = carla.Transform(carla.Location(z=1.762, x=(0.5146), y=(0.1976)), carla.Rotation(yaw=21.0003)) # distance=(0.5513)
                camera_init_trans_RU8 = carla.Transform(carla.Location(z=1.762, x=(0.5039), y=(0.2244)), carla.Rotation(yaw=24.0002)) # distance=(0.5516)
                camera_init_trans_RU9 = carla.Transform(carla.Location(z=1.762, x=(0.4918), y=(0.2506)), carla.Rotation(yaw=27.0003)) # distance=(0.5520)
                camera_init_trans_RU10 = carla.Transform(carla.Location(z=1.762, x=(0.4784), y=(0.2762)), carla.Rotation(yaw=30.0003)) # distance=(0.5524)
                camera_init_trans_RU11 = carla.Transform(carla.Location(z=1.762, x=(0.4637), y=(0.3011)), carla.Rotation(yaw=33.0001)) # distance=(0.5529)
                camera_init_trans_RU12 = carla.Transform(carla.Location(z=1.762, x=(0.4477), y=(0.3253)), carla.Rotation(yaw=36.0003)) # distance=(0.5534)
                camera_init_trans_RU13 = carla.Transform(carla.Location(z=1.762, x=(0.4305), y=(0.3486)), carla.Rotation(yaw=39.0002)) # distance=(0.5539)
                camera_init_trans_RU14 = carla.Transform(carla.Location(z=1.762, x=(0.4120), y=(0.3710)), carla.Rotation(yaw=42.0000)) # distance=(0.5544)
                camera_init_trans_RU15 = carla.Transform(carla.Location(z=1.762, x=(0.3924), y=(0.3924)), carla.Rotation(yaw=45.0003)) # distance=(0.5549)
                camera_init_trans_RU16 = carla.Transform(carla.Location(z=1.762, x=(0.3717), y=(0.4128)), carla.Rotation(yaw=48.0000)) # distance=(0.5555)
                camera_init_trans_RU17 = carla.Transform(carla.Location(z=1.762, x=(0.3499), y=(0.4321)), carla.Rotation(yaw=51.0002)) # distance=(0.5560)
                camera_init_trans_RU18 = carla.Transform(carla.Location(z=1.762, x=(0.3271), y=(0.4502)), carla.Rotation(yaw=54.0002)) # distance=(0.5565)
                camera_init_trans_RU19 = carla.Transform(carla.Location(z=1.762, x=(0.3034), y=(0.4671)), carla.Rotation(yaw=57.0001)) # distance=(0.5570)
                camera_init_trans_RU20 = carla.Transform(carla.Location(z=1.762, x=(0.2787), y=(0.4828)), carla.Rotation(yaw=60.0002)) # distance=(0.5574)
                camera_init_trans_RU21 = carla.Transform(carla.Location(z=1.762, x=(0.2533), y=(0.4971)), carla.Rotation(yaw=63.0001)) # distance=(0.5579)
                camera_init_trans_RU22 = carla.Transform(carla.Location(z=1.762, x=(0.2271), y=(0.5100)), carla.Rotation(yaw=66.0002)) # distance=(0.5583)
                camera_init_trans_RU23 = carla.Transform(carla.Location(z=1.762, x=(0.2002), y=(0.5216)), carla.Rotation(yaw=69.0001)) # distance=(0.5587)
                camera_init_trans_RU24 = carla.Transform(carla.Location(z=1.762, x=(0.1727), y=(0.5317)), carla.Rotation(yaw=72.0000)) # distance=(0.5590)
                camera_init_trans_RU25 = carla.Transform(carla.Location(z=1.762, x=(0.1448), y=(0.5403)), carla.Rotation(yaw=75.0001)) # distance=(0.5593)
                camera_init_trans_RU26 = carla.Transform(carla.Location(z=1.762, x=(0.1163), y=(0.5473)), carla.Rotation(yaw=78.0001)) # distance=(0.5596)
                camera_init_trans_RU27 = carla.Transform(carla.Location(z=1.762, x=(0.0876), y=(0.5529)), carla.Rotation(yaw=81.0000)) # distance=(0.5597)
                camera_init_trans_RU28 = carla.Transform(carla.Location(z=1.762, x=(0.0585), y=(0.5568)), carla.Rotation(yaw=84.0002)) # distance=(0.5599)
                camera_init_trans_RU29 = carla.Transform(carla.Location(z=1.762, x=(0.0293), y=(0.5592)), carla.Rotation(yaw=87.0002)) # distance=(0.5600)

            elif args.Camera_set_sh == 'Indago3_LU':
                print('====LU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=0.55), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-0.55), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.56), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.56), carla.Rotation(yaw=90))
                
                camera_init_trans_LU1 = carla.Transform(carla.Location(z=1.762, x=(0.5493), y=(-0.0288)), carla.Rotation(yaw=-3.0001)) # distance=(0.5500)
                camera_init_trans_LU2 = carla.Transform(carla.Location(z=1.762, x=(0.5471), y=(-0.0575)), carla.Rotation(yaw=-6.0003)) # distance=(0.5501)
                camera_init_trans_LU3 = carla.Transform(carla.Location(z=1.762, x=(0.5435), y=(-0.0861)), carla.Rotation(yaw=-9.0001)) # distance=(0.5502)
                camera_init_trans_LU4 = carla.Transform(carla.Location(z=1.762, x=(0.5384), y=(-0.1144)), carla.Rotation(yaw=-12.0000)) # distance=(0.5504)
                camera_init_trans_LU5 = carla.Transform(carla.Location(z=1.762, x=(0.5319), y=(-0.1425)), carla.Rotation(yaw=-15.0003)) # distance=(0.5507)
                camera_init_trans_LU6 = carla.Transform(carla.Location(z=1.762, x=(0.5240), y=(-0.1702)), carla.Rotation(yaw=-18.0002)) # distance=(0.5509)
                camera_init_trans_LU7 = carla.Transform(carla.Location(z=1.762, x=(0.5146), y=(-0.1976)), carla.Rotation(yaw=-21.0003)) # distance=(0.5513)
                camera_init_trans_LU8 = carla.Transform(carla.Location(z=1.762, x=(0.5039), y=(-0.2244)), carla.Rotation(yaw=-24.0002)) # distance=(0.5516)
                camera_init_trans_LU9 = carla.Transform(carla.Location(z=1.762, x=(0.4918), y=(-0.2506)), carla.Rotation(yaw=-27.0003)) # distance=(0.5520)
                camera_init_trans_LU10 = carla.Transform(carla.Location(z=1.762, x=(0.4784), y=(-0.2762)), carla.Rotation(yaw=-30.0003)) # distance=(0.5524)
                camera_init_trans_LU11 = carla.Transform(carla.Location(z=1.762, x=(0.4637), y=(-0.3011)), carla.Rotation(yaw=-33.0001)) # distance=(0.5529)
                camera_init_trans_LU12 = carla.Transform(carla.Location(z=1.762, x=(0.4477), y=(-0.3253)), carla.Rotation(yaw=-36.0003)) # distance=(0.5534)
                camera_init_trans_LU13 = carla.Transform(carla.Location(z=1.762, x=(0.4305), y=(-0.3486)), carla.Rotation(yaw=-39.0002)) # distance=(0.5539)
                camera_init_trans_LU14 = carla.Transform(carla.Location(z=1.762, x=(0.4120), y=(-0.3710)), carla.Rotation(yaw=-42.0000)) # distance=(0.5544)
                camera_init_trans_LU15 = carla.Transform(carla.Location(z=1.762, x=(0.3924), y=(-0.3924)), carla.Rotation(yaw=-45.0003)) # distance=(0.5549)
                camera_init_trans_LU16 = carla.Transform(carla.Location(z=1.762, x=(0.3717), y=(-0.4128)), carla.Rotation(yaw=-48.0000)) # distance=(0.5555)
                camera_init_trans_LU17 = carla.Transform(carla.Location(z=1.762, x=(0.3499), y=(-0.4321)), carla.Rotation(yaw=-51.0002)) # distance=(0.5560)
                camera_init_trans_LU18 = carla.Transform(carla.Location(z=1.762, x=(0.3271), y=(-0.4502)), carla.Rotation(yaw=-54.0002)) # distance=(0.5565)
                camera_init_trans_LU19 = carla.Transform(carla.Location(z=1.762, x=(0.3034), y=(-0.4671)), carla.Rotation(yaw=-57.0001)) # distance=(0.5570)
                camera_init_trans_LU20 = carla.Transform(carla.Location(z=1.762, x=(0.2787), y=(-0.4828)), carla.Rotation(yaw=-60.0002)) # distance=(0.5574)
                camera_init_trans_LU21 = carla.Transform(carla.Location(z=1.762, x=(0.2533), y=(-0.4971)), carla.Rotation(yaw=-63.0001)) # distance=(0.5579)
                camera_init_trans_LU22 = carla.Transform(carla.Location(z=1.762, x=(0.2271), y=(-0.5100)), carla.Rotation(yaw=-66.0002)) # distance=(0.5583)
                camera_init_trans_LU23 = carla.Transform(carla.Location(z=1.762, x=(0.2002), y=(-0.5216)), carla.Rotation(yaw=-69.0001)) # distance=(0.5587)
                camera_init_trans_LU24 = carla.Transform(carla.Location(z=1.762, x=(0.1727), y=(-0.5317)), carla.Rotation(yaw=-72.0000)) # distance=(0.5590)
                camera_init_trans_LU25 = carla.Transform(carla.Location(z=1.762, x=(0.1448), y=(-0.5403)), carla.Rotation(yaw=-75.0001)) # distance=(0.5593)
                camera_init_trans_LU26 = carla.Transform(carla.Location(z=1.762, x=(0.1163), y=(-0.5473)), carla.Rotation(yaw=-78.0001)) # distance=(0.5596)
                camera_init_trans_LU27 = carla.Transform(carla.Location(z=1.762, x=(0.0876), y=(-0.5529)), carla.Rotation(yaw=-81.0000)) # distance=(0.5597)
                camera_init_trans_LU28 = carla.Transform(carla.Location(z=1.762, x=(0.0585), y=(-0.5568)), carla.Rotation(yaw=-84.0002)) # distance=(0.5599)
                camera_init_trans_LU29 = carla.Transform(carla.Location(z=1.762, x=(0.0293), y=(-0.5592)), carla.Rotation(yaw=-87.0002)) # distance=(0.5600)

            elif args.Camera_set_sh == 'Indago3_RD':
                print('====RD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=0.55), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-0.55), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.56), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.56), carla.Rotation(yaw=90))
                
                camera_init_trans_RD1 = carla.Transform(carla.Location(z=1.762, x=(-0.0293), y=(0.5592)), carla.Rotation(yaw=93.0002)) # distance=(0.5600)
                camera_init_trans_RD2 = carla.Transform(carla.Location(z=1.762, x=(-0.0585), y=(0.5568)), carla.Rotation(yaw=96.0002)) # distance=(0.5599)
                camera_init_trans_RD3 = carla.Transform(carla.Location(z=1.762, x=(-0.0876), y=(0.5529)), carla.Rotation(yaw=99.0000)) # distance=(0.5597)
                camera_init_trans_RD4 = carla.Transform(carla.Location(z=1.762, x=(-0.1163), y=(0.5473)), carla.Rotation(yaw=102.0002)) # distance=(0.5596)
                camera_init_trans_RD5 = carla.Transform(carla.Location(z=1.762, x=(-0.1448), y=(0.5403)), carla.Rotation(yaw=105.0003)) # distance=(0.5593)
                camera_init_trans_RD6 = carla.Transform(carla.Location(z=1.762, x=(-0.1727), y=(0.5317)), carla.Rotation(yaw=108.0000)) # distance=(0.5590)
                camera_init_trans_RD7 = carla.Transform(carla.Location(z=1.762, x=(-0.2002), y=(0.5216)), carla.Rotation(yaw=111.0003)) # distance=(0.5587)
                camera_init_trans_RD8 = carla.Transform(carla.Location(z=1.762, x=(-0.2271), y=(0.5100)), carla.Rotation(yaw=114.0001)) # distance=(0.5583)
                camera_init_trans_RD9 = carla.Transform(carla.Location(z=1.762, x=(-0.2533), y=(0.4971)), carla.Rotation(yaw=117.0003)) # distance=(0.5579)
                camera_init_trans_RD10 = carla.Transform(carla.Location(z=1.762, x=(-0.2787), y=(0.4828)), carla.Rotation(yaw=120.0001)) # distance=(0.5574)
                camera_init_trans_RD11 = carla.Transform(carla.Location(z=1.762, x=(-0.3034), y=(0.4671)), carla.Rotation(yaw=123.0002)) # distance=(0.5570)
                camera_init_trans_RD12 = carla.Transform(carla.Location(z=1.762, x=(-0.3271), y=(0.4502)), carla.Rotation(yaw=126.0002)) # distance=(0.5565)
                camera_init_trans_RD13 = carla.Transform(carla.Location(z=1.762, x=(-0.3499), y=(0.4321)), carla.Rotation(yaw=129.0002)) # distance=(0.5560)
                camera_init_trans_RD14 = carla.Transform(carla.Location(z=1.762, x=(-0.3717), y=(0.4128)), carla.Rotation(yaw=132.0000)) # distance=(0.5555)
                camera_init_trans_RD15 = carla.Transform(carla.Location(z=1.762, x=(-0.3924), y=(0.3924)), carla.Rotation(yaw=135.0001)) # distance=(0.5549)
                camera_init_trans_RD16 = carla.Transform(carla.Location(z=1.762, x=(-0.4120), y=(0.3710)), carla.Rotation(yaw=138.0000)) # distance=(0.5544)
                camera_init_trans_RD17 = carla.Transform(carla.Location(z=1.762, x=(-0.4305), y=(0.3486)), carla.Rotation(yaw=141.0002)) # distance=(0.5539)
                camera_init_trans_RD18 = carla.Transform(carla.Location(z=1.762, x=(-0.4477), y=(0.3253)), carla.Rotation(yaw=144.0001)) # distance=(0.5534)
                camera_init_trans_RD19 = carla.Transform(carla.Location(z=1.762, x=(-0.4637), y=(0.3011)), carla.Rotation(yaw=147.0002)) # distance=(0.5529)
                camera_init_trans_RD20 = carla.Transform(carla.Location(z=1.762, x=(-0.4784), y=(0.2762)), carla.Rotation(yaw=150.0001)) # distance=(0.5524)
                camera_init_trans_RD21 = carla.Transform(carla.Location(z=1.762, x=(-0.4919), y=(0.2506)), carla.Rotation(yaw=153.0001)) # distance=(0.5520)
                camera_init_trans_RD22 = carla.Transform(carla.Location(z=1.762, x=(-0.5039), y=(0.2244)), carla.Rotation(yaw=156.0002)) # distance=(0.5516)
                camera_init_trans_RD23 = carla.Transform(carla.Location(z=1.762, x=(-0.5146), y=(0.1976)), carla.Rotation(yaw=159.0001)) # distance=(0.5513)
                camera_init_trans_RD24 = carla.Transform(carla.Location(z=1.762, x=(-0.5240), y=(0.1702)), carla.Rotation(yaw=162.0001)) # distance=(0.5509)
                camera_init_trans_RD25 = carla.Transform(carla.Location(z=1.762, x=(-0.5319), y=(0.1425)), carla.Rotation(yaw=165.0001)) # distance=(0.5507)
                camera_init_trans_RD26 = carla.Transform(carla.Location(z=1.762, x=(-0.5384), y=(0.1144)), carla.Rotation(yaw=168.0000)) # distance=(0.5504)
                camera_init_trans_RD27 = carla.Transform(carla.Location(z=1.762, x=(-0.5435), y=(0.0861)), carla.Rotation(yaw=171.0003)) # distance=(0.5502)
                camera_init_trans_RD28 = carla.Transform(carla.Location(z=1.762, x=(-0.5471), y=(0.0575)), carla.Rotation(yaw=174.0001)) # distance=(0.5501)
                camera_init_trans_RD29 = carla.Transform(carla.Location(z=1.762, x=(-0.5493), y=(0.0288)), carla.Rotation(yaw=177.0003)) # distance=(0.5500)

            elif args.Camera_set_sh == 'Indago3_LD':
                print('====LD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=0.55), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-0.55), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.56), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.56), carla.Rotation(yaw=90))
                
                camera_init_trans_LD1 = carla.Transform(carla.Location(z=1.762, x=(-0.0293), y=(-0.5592)), carla.Rotation(yaw=-93.0002)) # distance=(0.5600)
                camera_init_trans_LD2 = carla.Transform(carla.Location(z=1.762, x=(-0.0585), y=(-0.5568)), carla.Rotation(yaw=-96.0002)) # distance=(0.5599)
                camera_init_trans_LD3 = carla.Transform(carla.Location(z=1.762, x=(-0.0876), y=(-0.5529)), carla.Rotation(yaw=-99.0000)) # distance=(0.5597)
                camera_init_trans_LD4 = carla.Transform(carla.Location(z=1.762, x=(-0.1163), y=(-0.5473)), carla.Rotation(yaw=-102.0002)) # distance=(0.5596)
                camera_init_trans_LD5 = carla.Transform(carla.Location(z=1.762, x=(-0.1448), y=(-0.5403)), carla.Rotation(yaw=-105.0003)) # distance=(0.5593)
                camera_init_trans_LD6 = carla.Transform(carla.Location(z=1.762, x=(-0.1727), y=(-0.5317)), carla.Rotation(yaw=-108.0000)) # distance=(0.5590)
                camera_init_trans_LD7 = carla.Transform(carla.Location(z=1.762, x=(-0.2002), y=(-0.5216)), carla.Rotation(yaw=-111.0003)) # distance=(0.5587)
                camera_init_trans_LD8 = carla.Transform(carla.Location(z=1.762, x=(-0.2271), y=(-0.5100)), carla.Rotation(yaw=-114.0001)) # distance=(0.5583)
                camera_init_trans_LD9 = carla.Transform(carla.Location(z=1.762, x=(-0.2533), y=(-0.4971)), carla.Rotation(yaw=-117.0003)) # distance=(0.5579)
                camera_init_trans_LD10 = carla.Transform(carla.Location(z=1.762, x=(-0.2787), y=(-0.4828)), carla.Rotation(yaw=-120.0001)) # distance=(0.5574)
                camera_init_trans_LD11 = carla.Transform(carla.Location(z=1.762, x=(-0.3034), y=(-0.4671)), carla.Rotation(yaw=-123.0002)) # distance=(0.5570)
                camera_init_trans_LD12 = carla.Transform(carla.Location(z=1.762, x=(-0.3271), y=(-0.4502)), carla.Rotation(yaw=-126.0002)) # distance=(0.5565)
                camera_init_trans_LD13 = carla.Transform(carla.Location(z=1.762, x=(-0.3499), y=(-0.4321)), carla.Rotation(yaw=-129.0002)) # distance=(0.5560)
                camera_init_trans_LD14 = carla.Transform(carla.Location(z=1.762, x=(-0.3717), y=(-0.4128)), carla.Rotation(yaw=-132.0000)) # distance=(0.5555)
                camera_init_trans_LD15 = carla.Transform(carla.Location(z=1.762, x=(-0.3924), y=(-0.3924)), carla.Rotation(yaw=-135.0001)) # distance=(0.5549)
                camera_init_trans_LD16 = carla.Transform(carla.Location(z=1.762, x=(-0.4120), y=(-0.3710)), carla.Rotation(yaw=-138.0000)) # distance=(0.5544)
                camera_init_trans_LD17 = carla.Transform(carla.Location(z=1.762, x=(-0.4305), y=(-0.3486)), carla.Rotation(yaw=-141.0002)) # distance=(0.5539)
                camera_init_trans_LD18 = carla.Transform(carla.Location(z=1.762, x=(-0.4477), y=(-0.3253)), carla.Rotation(yaw=-144.0001)) # distance=(0.5534)
                camera_init_trans_LD19 = carla.Transform(carla.Location(z=1.762, x=(-0.4637), y=(-0.3011)), carla.Rotation(yaw=-147.0002)) # distance=(0.5529)
                camera_init_trans_LD20 = carla.Transform(carla.Location(z=1.762, x=(-0.4784), y=(-0.2762)), carla.Rotation(yaw=-150.0001)) # distance=(0.5524)
                camera_init_trans_LD21 = carla.Transform(carla.Location(z=1.762, x=(-0.4919), y=(-0.2506)), carla.Rotation(yaw=-153.0001)) # distance=(0.5520)
                camera_init_trans_LD22 = carla.Transform(carla.Location(z=1.762, x=(-0.5039), y=(-0.2244)), carla.Rotation(yaw=-156.0002)) # distance=(0.5516)
                camera_init_trans_LD23 = carla.Transform(carla.Location(z=1.762, x=(-0.5146), y=(-0.1976)), carla.Rotation(yaw=-159.0001)) # distance=(0.5513)
                camera_init_trans_LD24 = carla.Transform(carla.Location(z=1.762, x=(-0.5240), y=(-0.1702)), carla.Rotation(yaw=-162.0001)) # distance=(0.5509)
                camera_init_trans_LD25 = carla.Transform(carla.Location(z=1.762, x=(-0.5319), y=(-0.1425)), carla.Rotation(yaw=-165.0001)) # distance=(0.5507)
                camera_init_trans_LD26 = carla.Transform(carla.Location(z=1.762, x=(-0.5384), y=(-0.1144)), carla.Rotation(yaw=-168.0000)) # distance=(0.5504)
                camera_init_trans_LD27 = carla.Transform(carla.Location(z=1.762, x=(-0.5435), y=(-0.0861)), carla.Rotation(yaw=-171.0003)) # distance=(0.5502)
                camera_init_trans_LD28 = carla.Transform(carla.Location(z=1.762, x=(-0.5471), y=(-0.0575)), carla.Rotation(yaw=-174.0001)) # distance=(0.5501)
                camera_init_trans_LD29 = carla.Transform(carla.Location(z=1.762, x=(-0.5493), y=(-0.0288)), carla.Rotation(yaw=-177.0003)) # distance=(0.5500)

            elif args.Camera_set_sh == 'BaselineA_RU':
                print('====RU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=1.662, x=1.24), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=1.662, x=-1.24), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=1.662, x=(0.0), y=-0.64), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=1.662, x=(0.0), y=0.64), carla.Rotation(yaw=90))

                camera_init_trans_RU1 = carla.Transform(carla.Location(z=1.662, x=(1.2337), y=(0.0647)), carla.Rotation(yaw=3.0)) # distance=(1.2353)
                camera_init_trans_RU2 = carla.Transform(carla.Location(z=1.662, x=(1.2151), y=(0.1277)), carla.Rotation(yaw=6.0)) # distance=(1.2218)
                camera_init_trans_RU3 = carla.Transform(carla.Location(z=1.662, x=(1.1854), y=(0.1878)), carla.Rotation(yaw=9.0)) # distance=(1.2002)
                camera_init_trans_RU4 = carla.Transform(carla.Location(z=1.662, x=(1.1466), y=(0.2437)), carla.Rotation(yaw=12.0)) # distance=(1.1722)
                camera_init_trans_RU5 = carla.Transform(carla.Location(z=1.662, x=(1.1005), y=(0.2949)), carla.Rotation(yaw=15.0)) # distance=(1.1394)
                camera_init_trans_RU6 = carla.Transform(carla.Location(z=1.662, x=(1.0494), y=(0.3410)), carla.Rotation(yaw=18.0)) # distance=(1.1034)
                camera_init_trans_RU7 = carla.Transform(carla.Location(z=1.662, x=(0.9950), y=(0.3819)), carla.Rotation(yaw=21.0)) # distance=(1.0658)
                camera_init_trans_RU8 = carla.Transform(carla.Location(z=1.662, x=(0.9389), y=(0.4180)), carla.Rotation(yaw=24.0)) # distance=(1.0278)
                camera_init_trans_RU9 = carla.Transform(carla.Location(z=1.662, x=(0.8824), y=(0.4496)), carla.Rotation(yaw=27.0)) # distance=(0.9904)
                camera_init_trans_RU10 = carla.Transform(carla.Location(z=1.662, x=(0.8264), y=(0.4771)), carla.Rotation(yaw=30.0)) # distance=(0.9543)
                camera_init_trans_RU11 = carla.Transform(carla.Location(z=1.662, x=(0.7715), y=(0.5010)), carla.Rotation(yaw=33.0)) # distance=(0.9199)
                camera_init_trans_RU12 = carla.Transform(carla.Location(z=1.662, x=(0.7181), y=(0.5218)), carla.Rotation(yaw=36.0)) # distance=(0.8877)
                camera_init_trans_RU13 = carla.Transform(carla.Location(z=1.662, x=(0.6665), y=(0.5397)), carla.Rotation(yaw=39.0)) # distance=(0.8576)
                camera_init_trans_RU14 = carla.Transform(carla.Location(z=1.662, x=(0.6167), y=(0.5552)), carla.Rotation(yaw=42.0)) # distance=(0.8298)
                camera_init_trans_RU15 = carla.Transform(carla.Location(z=1.662, x=(0.5687), y=(0.5687)), carla.Rotation(yaw=45.0)) # distance=(0.8043)
                camera_init_trans_RU16 = carla.Transform(carla.Location(z=1.662, x=(0.5226), y=(0.5804)), carla.Rotation(yaw=48.0)) # distance=(0.7810)
                camera_init_trans_RU17 = carla.Transform(carla.Location(z=1.662, x=(0.4782), y=(0.5905)), carla.Rotation(yaw=51.0)) # distance=(0.7598)
                camera_init_trans_RU18 = carla.Transform(carla.Location(z=1.662, x=(0.4354), y=(0.5993)), carla.Rotation(yaw=54.0)) # distance=(0.7407)
                camera_init_trans_RU19 = carla.Transform(carla.Location(z=1.662, x=(0.3941), y=(0.6068)), carla.Rotation(yaw=57.0)) # distance=(0.7235)
                camera_init_trans_RU20 = carla.Transform(carla.Location(z=1.662, x=(0.3541), y=(0.6133)), carla.Rotation(yaw=60.0)) # distance=(0.7082)
                camera_init_trans_RU21 = carla.Transform(carla.Location(z=1.662, x=(0.3154), y=(0.6190)), carla.Rotation(yaw=63.0)) # distance=(0.6947)
                camera_init_trans_RU22 = carla.Transform(carla.Location(z=1.662, x=(0.2777), y=(0.6237)), carla.Rotation(yaw=66.0)) # distance=(0.6828)
                camera_init_trans_RU23 = carla.Transform(carla.Location(z=1.662, x=(0.2410), y=(0.6278)), carla.Rotation(yaw=69.0)) # distance=(0.6725)
                camera_init_trans_RU24 = carla.Transform(carla.Location(z=1.662, x=(0.2051), y=(0.6312)), carla.Rotation(yaw=72.0)) # distance=(0.6637)
                camera_init_trans_RU25 = carla.Transform(carla.Location(z=1.662, x=(0.1699), y=(0.6340)), carla.Rotation(yaw=75.0)) # distance=(0.6563)
                camera_init_trans_RU26 = carla.Transform(carla.Location(z=1.662, x=(0.1352), y=(0.6362)), carla.Rotation(yaw=78.0)) # distance=(0.6504)
                camera_init_trans_RU27 = carla.Transform(carla.Location(z=1.662, x=(0.1010), y=(0.6379)), carla.Rotation(yaw=81.0)) # distance=(0.6458)
                camera_init_trans_RU28 = carla.Transform(carla.Location(z=1.662, x=(0.0672), y=(0.6391)), carla.Rotation(yaw=84.0)) # distance=(0.6426)
                camera_init_trans_RU29 = carla.Transform(carla.Location(z=1.662, x=(0.0335), y=(0.6398)), carla.Rotation(yaw=87.0)) # distance=(0.6406)
                
            elif args.Camera_set_sh == 'BaselineA_LU':
                print('====LU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=1.662, x=1.24), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=1.662, x=-1.24), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=1.662, x=(0.0), y=-0.64), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=1.662, x=(0.0), y=0.64), carla.Rotation(yaw=90))

                camera_init_trans_LU29 = carla.Transform(carla.Location(z=1.662, x=(0.0335), y=(-0.6398)), carla.Rotation(yaw=-87.0)) # distance=(0.6406)
                camera_init_trans_LU28 = carla.Transform(carla.Location(z=1.662, x=(0.0672), y=(-0.6391)), carla.Rotation(yaw=-84.0)) # distance=(0.6426)
                camera_init_trans_LU27 = carla.Transform(carla.Location(z=1.662, x=(0.1010), y=(-0.6379)), carla.Rotation(yaw=-81.0)) # distance=(0.6458)
                camera_init_trans_LU26 = carla.Transform(carla.Location(z=1.662, x=(0.1352), y=(-0.6362)), carla.Rotation(yaw=-78.0)) # distance=(0.6504)
                camera_init_trans_LU25 = carla.Transform(carla.Location(z=1.662, x=(0.1699), y=(-0.6340)), carla.Rotation(yaw=-75.0)) # distance=(0.6563)
                camera_init_trans_LU24 = carla.Transform(carla.Location(z=1.662, x=(0.2051), y=(-0.6312)), carla.Rotation(yaw=-72.0)) # distance=(0.6637)
                camera_init_trans_LU23 = carla.Transform(carla.Location(z=1.662, x=(0.2410), y=(-0.6278)), carla.Rotation(yaw=-69.0)) # distance=(0.6725)
                camera_init_trans_LU22 = carla.Transform(carla.Location(z=1.662, x=(0.2777), y=(-0.6237)), carla.Rotation(yaw=-66.0)) # distance=(0.6828)
                camera_init_trans_LU21 = carla.Transform(carla.Location(z=1.662, x=(0.3154), y=(-0.6190)), carla.Rotation(yaw=-63.0)) # distance=(0.6947)
                camera_init_trans_LU20 = carla.Transform(carla.Location(z=1.662, x=(0.3541), y=(-0.6133)), carla.Rotation(yaw=-60.0)) # distance=(0.7082)
                camera_init_trans_LU19 = carla.Transform(carla.Location(z=1.662, x=(0.3941), y=(-0.6068)), carla.Rotation(yaw=-57.0)) # distance=(0.7235)
                camera_init_trans_LU18 = carla.Transform(carla.Location(z=1.662, x=(0.4354), y=(-0.5993)), carla.Rotation(yaw=-54.0)) # distance=(0.7407)
                camera_init_trans_LU17 = carla.Transform(carla.Location(z=1.662, x=(0.4782), y=(-0.5905)), carla.Rotation(yaw=-51.0)) # distance=(0.7598)
                camera_init_trans_LU16 = carla.Transform(carla.Location(z=1.662, x=(0.5226), y=(-0.5804)), carla.Rotation(yaw=-48.0)) # distance=(0.7810)
                camera_init_trans_LU15 = carla.Transform(carla.Location(z=1.662, x=(0.5687), y=(-0.5687)), carla.Rotation(yaw=-45.0)) # distance=(0.8043)
                camera_init_trans_LU14 = carla.Transform(carla.Location(z=1.662, x=(0.6167), y=(-0.5552)), carla.Rotation(yaw=-42.0)) # distance=(0.8298)
                camera_init_trans_LU13 = carla.Transform(carla.Location(z=1.662, x=(0.6665), y=(-0.5397)), carla.Rotation(yaw=-39.0)) # distance=(0.8576)
                camera_init_trans_LU12 = carla.Transform(carla.Location(z=1.662, x=(0.7181), y=(-0.5218)), carla.Rotation(yaw=-36.0)) # distance=(0.8877)
                camera_init_trans_LU11 = carla.Transform(carla.Location(z=1.662, x=(0.7715), y=(-0.5010)), carla.Rotation(yaw=-33.0)) # distance=(0.9199)
                camera_init_trans_LU10 = carla.Transform(carla.Location(z=1.662, x=(0.8264), y=(-0.4771)), carla.Rotation(yaw=-30.0)) # distance=(0.9543)
                camera_init_trans_LU9 = carla.Transform(carla.Location(z=1.662, x=(0.8824), y=(-0.4496)), carla.Rotation(yaw=-27.0)) # distance=(0.9904)
                camera_init_trans_LU8 = carla.Transform(carla.Location(z=1.662, x=(0.9389), y=(-0.4180)), carla.Rotation(yaw=-24.0)) # distance=(1.0278)
                camera_init_trans_LU7 = carla.Transform(carla.Location(z=1.662, x=(0.9950), y=(-0.3819)), carla.Rotation(yaw=-21.0)) # distance=(1.0658)
                camera_init_trans_LU6 = carla.Transform(carla.Location(z=1.662, x=(1.0494), y=(-0.3410)), carla.Rotation(yaw=-18.0)) # distance=(1.1034)
                camera_init_trans_LU5 = carla.Transform(carla.Location(z=1.662, x=(1.1005), y=(-0.2949)), carla.Rotation(yaw=-15.0)) # distance=(1.1394)
                camera_init_trans_LU4 = carla.Transform(carla.Location(z=1.662, x=(1.1466), y=(-0.2437)), carla.Rotation(yaw=-12.0)) # distance=(1.1722)
                camera_init_trans_LU3 = carla.Transform(carla.Location(z=1.662, x=(1.1854), y=(-0.1878)), carla.Rotation(yaw=-9.0)) # distance=(1.2002)
                camera_init_trans_LU2 = carla.Transform(carla.Location(z=1.662, x=(1.2151), y=(-0.1277)), carla.Rotation(yaw=-6.0)) # distance=(1.2218)
                camera_init_trans_LU1 = carla.Transform(carla.Location(z=1.662, x=(1.2337), y=(-0.0647)), carla.Rotation(yaw=-3.0)) # distance=(1.2353)
                
            elif args.Camera_set_sh == 'BaselineA_RD':
                print('====RD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=1.662, x=1.24), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=1.662, x=-1.24), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=1.662, x=(0.0), y=-0.64), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=1.662, x=(0.0), y=0.64), carla.Rotation(yaw=90))

                camera_init_trans_RD1 = carla.Transform(carla.Location(z=1.662, x=(-0.0335), y=(0.6398)), carla.Rotation(yaw=93.0)) # distance=(0.6406)
                camera_init_trans_RD2 = carla.Transform(carla.Location(z=1.662, x=(-0.0672), y=(0.6391)), carla.Rotation(yaw=96.0)) # distance=(0.6426)
                camera_init_trans_RD3 = carla.Transform(carla.Location(z=1.662, x=(-0.1010), y=(0.6379)), carla.Rotation(yaw=99.0)) # distance=(0.6458)
                camera_init_trans_RD4 = carla.Transform(carla.Location(z=1.662, x=(-0.1352), y=(0.6362)), carla.Rotation(yaw=102.0)) # distance=(0.6504)
                camera_init_trans_RD5 = carla.Transform(carla.Location(z=1.662, x=(-0.1699), y=(0.6340)), carla.Rotation(yaw=105.0)) # distance=(0.6563)
                camera_init_trans_RD6 = carla.Transform(carla.Location(z=1.662, x=(-0.2051), y=(0.6312)), carla.Rotation(yaw=108.0)) # distance=(0.6637)
                camera_init_trans_RD7 = carla.Transform(carla.Location(z=1.662, x=(-0.2410), y=(0.6278)), carla.Rotation(yaw=111.0)) # distance=(0.6725)
                camera_init_trans_RD8 = carla.Transform(carla.Location(z=1.662, x=(-0.2777), y=(0.6237)), carla.Rotation(yaw=114.0)) # distance=(0.6828)
                camera_init_trans_RD9 = carla.Transform(carla.Location(z=1.662, x=(-0.3154), y=(0.6190)), carla.Rotation(yaw=117.0)) # distance=(0.6947)
                camera_init_trans_RD10 = carla.Transform(carla.Location(z=1.662, x=(-0.3541), y=(0.6133)), carla.Rotation(yaw=120.0)) # distance=(0.7082)
                camera_init_trans_RD11 = carla.Transform(carla.Location(z=1.662, x=(-0.3941), y=(0.6068)), carla.Rotation(yaw=123.0)) # distance=(0.7236)
                camera_init_trans_RD12 = carla.Transform(carla.Location(z=1.662, x=(-0.4354), y=(0.5993)), carla.Rotation(yaw=126.0)) # distance=(0.7407)
                camera_init_trans_RD13 = carla.Transform(carla.Location(z=1.662, x=(-0.4782), y=(0.5905)), carla.Rotation(yaw=129.0)) # distance=(0.7598)
                camera_init_trans_RD14 = carla.Transform(carla.Location(z=1.662, x=(-0.5226), y=(0.5804)), carla.Rotation(yaw=132.0)) # distance=(0.7810)
                camera_init_trans_RD15 = carla.Transform(carla.Location(z=1.662, x=(-0.5687), y=(0.5687)), carla.Rotation(yaw=135.0)) # distance=(0.8043)
                camera_init_trans_RD16 = carla.Transform(carla.Location(z=1.662, x=(-0.6167), y=(0.5552)), carla.Rotation(yaw=138.0)) # distance=(0.8298)
                camera_init_trans_RD17 = carla.Transform(carla.Location(z=1.662, x=(-0.6665), y=(0.5397)), carla.Rotation(yaw=141.0)) # distance=(0.8576)
                camera_init_trans_RD18 = carla.Transform(carla.Location(z=1.662, x=(-0.7181), y=(0.5217)), carla.Rotation(yaw=144.0)) # distance=(0.8877)
                camera_init_trans_RD19 = carla.Transform(carla.Location(z=1.662, x=(-0.7715), y=(0.5010)), carla.Rotation(yaw=147.0)) # distance=(0.9199)
                camera_init_trans_RD20 = carla.Transform(carla.Location(z=1.662, x=(-0.8264), y=(0.4771)), carla.Rotation(yaw=150.0)) # distance=(0.9543)
                camera_init_trans_RD21 = carla.Transform(carla.Location(z=1.662, x=(-0.8824), y=(0.4496)), carla.Rotation(yaw=153.0)) # distance=(0.9904)
                camera_init_trans_RD22 = carla.Transform(carla.Location(z=1.662, x=(-0.9389), y=(0.4180)), carla.Rotation(yaw=156.0)) # distance=(1.0278)
                camera_init_trans_RD23 = carla.Transform(carla.Location(z=1.662, x=(-0.9950), y=(0.3819)), carla.Rotation(yaw=159.0)) # distance=(1.0658)
                camera_init_trans_RD24 = carla.Transform(carla.Location(z=1.662, x=(-1.0494), y=(0.3410)), carla.Rotation(yaw=162.0)) # distance=(1.1034)
                camera_init_trans_RD25 = carla.Transform(carla.Location(z=1.662, x=(-1.1005), y=(0.2949)), carla.Rotation(yaw=165.0)) # distance=(1.1394)
                camera_init_trans_RD26 = carla.Transform(carla.Location(z=1.662, x=(-1.1466), y=(0.2437)), carla.Rotation(yaw=168.0)) # distance=(1.1722)
                camera_init_trans_RD27 = carla.Transform(carla.Location(z=1.662, x=(-1.1854), y=(0.1878)), carla.Rotation(yaw=171.0)) # distance=(1.2002)
                camera_init_trans_RD28 = carla.Transform(carla.Location(z=1.662, x=(-1.2151), y=(0.1277)), carla.Rotation(yaw=174.0)) # distance=(1.2218)
                camera_init_trans_RD29 = carla.Transform(carla.Location(z=1.662, x=(-1.2337), y=(0.0647)), carla.Rotation(yaw=177.0)) # distance=(1.2353)
                
            elif args.Camera_set_sh == 'BaselineA_LD':
                print('====LD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=1.662, x=1.24), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=1.662, x=-1.24), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=1.662, x=(0.0), y=-0.64), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=1.662, x=(0.0), y=0.64), carla.Rotation(yaw=90))

                camera_init_trans_LD29 = carla.Transform(carla.Location(z=1.662, x=(-1.2337), y=(-0.0647)), carla.Rotation(yaw=-177.0)) # distance=(1.2353)
                camera_init_trans_LD28 = carla.Transform(carla.Location(z=1.662, x=(-1.2151), y=(-0.1277)), carla.Rotation(yaw=-174.0)) # distance=(1.2218)
                camera_init_trans_LD27 = carla.Transform(carla.Location(z=1.662, x=(-1.1854), y=(-0.1877)), carla.Rotation(yaw=-171.0)) # distance=(1.2002)
                camera_init_trans_LD26 = carla.Transform(carla.Location(z=1.662, x=(-1.1466), y=(-0.2437)), carla.Rotation(yaw=-168.0)) # distance=(1.1722)
                camera_init_trans_LD25 = carla.Transform(carla.Location(z=1.662, x=(-1.1005), y=(-0.2949)), carla.Rotation(yaw=-165.0)) # distance=(1.1394)
                camera_init_trans_LD24 = carla.Transform(carla.Location(z=1.662, x=(-1.0494), y=(-0.3410)), carla.Rotation(yaw=-162.0)) # distance=(1.1034)
                camera_init_trans_LD23 = carla.Transform(carla.Location(z=1.662, x=(-0.9950), y=(-0.3819)), carla.Rotation(yaw=-159.0)) # distance=(1.0658)
                camera_init_trans_LD22 = carla.Transform(carla.Location(z=1.662, x=(-0.9389), y=(-0.4180)), carla.Rotation(yaw=-156.0)) # distance=(1.0278)
                camera_init_trans_LD21 = carla.Transform(carla.Location(z=1.662, x=(-0.8825), y=(-0.4496)), carla.Rotation(yaw=-153.0)) # distance=(0.9904)
                camera_init_trans_LD20 = carla.Transform(carla.Location(z=1.662, x=(-0.8264), y=(-0.4771)), carla.Rotation(yaw=-150.0)) # distance=(0.9543)
                camera_init_trans_LD19 = carla.Transform(carla.Location(z=1.662, x=(-0.7715), y=(-0.5010)), carla.Rotation(yaw=-147.0)) # distance=(0.9199)
                camera_init_trans_LD18 = carla.Transform(carla.Location(z=1.662, x=(-0.7181), y=(-0.5217)), carla.Rotation(yaw=-144.0)) # distance=(0.8877)
                camera_init_trans_LD17 = carla.Transform(carla.Location(z=1.662, x=(-0.6665), y=(-0.5397)), carla.Rotation(yaw=-141.0)) # distance=(0.8576)
                camera_init_trans_LD16 = carla.Transform(carla.Location(z=1.662, x=(-0.6167), y=(-0.5552)), carla.Rotation(yaw=-138.0)) # distance=(0.8298)
                camera_init_trans_LD15 = carla.Transform(carla.Location(z=1.662, x=(-0.5687), y=(-0.5687)), carla.Rotation(yaw=-135.0)) # distance=(0.8043)
                camera_init_trans_LD14 = carla.Transform(carla.Location(z=1.662, x=(-0.5226), y=(-0.5804)), carla.Rotation(yaw=-132.0)) # distance=(0.7810)
                camera_init_trans_LD13 = carla.Transform(carla.Location(z=1.662, x=(-0.4782), y=(-0.5905)), carla.Rotation(yaw=-129.0)) # distance=(0.7598)
                camera_init_trans_LD12 = carla.Transform(carla.Location(z=1.662, x=(-0.4354), y=(-0.5993)), carla.Rotation(yaw=-126.0)) # distance=(0.7407)
                camera_init_trans_LD11 = carla.Transform(carla.Location(z=1.662, x=(-0.3941), y=(-0.6068)), carla.Rotation(yaw=-123.0)) # distance=(0.7236)
                camera_init_trans_LD10 = carla.Transform(carla.Location(z=1.662, x=(-0.3541), y=(-0.6133)), carla.Rotation(yaw=-120.0)) # distance=(0.7082)
                camera_init_trans_LD9 = carla.Transform(carla.Location(z=1.662, x=(-0.3154), y=(-0.6190)), carla.Rotation(yaw=-117.0)) # distance=(0.6947)
                camera_init_trans_LD8 = carla.Transform(carla.Location(z=1.662, x=(-0.2777), y=(-0.6237)), carla.Rotation(yaw=-114.0)) # distance=(0.6828)
                camera_init_trans_LD7 = carla.Transform(carla.Location(z=1.662, x=(-0.2410), y=(-0.6278)), carla.Rotation(yaw=-111.0)) # distance=(0.6725)
                camera_init_trans_LD6 = carla.Transform(carla.Location(z=1.662, x=(-0.2051), y=(-0.6312)), carla.Rotation(yaw=-108.0)) # distance=(0.6637)
                camera_init_trans_LD5 = carla.Transform(carla.Location(z=1.662, x=(-0.1699), y=(-0.6340)), carla.Rotation(yaw=-105.0)) # distance=(0.6563)
                camera_init_trans_LD4 = carla.Transform(carla.Location(z=1.662, x=(-0.1352), y=(-0.6362)), carla.Rotation(yaw=-102.0)) # distance=(0.6504)
                camera_init_trans_LD3 = carla.Transform(carla.Location(z=1.662, x=(-0.1010), y=(-0.6379)), carla.Rotation(yaw=-99.0)) # distance=(0.6458)
                camera_init_trans_LD2 = carla.Transform(carla.Location(z=1.662, x=(-0.0672), y=(-0.6391)), carla.Rotation(yaw=-96.0)) # distance=(0.6426)
                camera_init_trans_LD1 = carla.Transform(carla.Location(z=1.662, x=(-0.0335), y=(-0.6398)), carla.Rotation(yaw=-93.0)) # distance=(0.6406)
                
            elif args.Camera_set_sh == 'base_plus05_RU':
                print('====RU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=2.3, x=0.24), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=2.3, x=-0.24), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=2.3, x=0.0, y=-0.44), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=2.3, x=(0.0), y=0.44), carla.Rotation(yaw=90))

                camera_init_trans_RU1 = carla.Transform(carla.Location(z=2.3, x=(0.2399), y=(0.0126)), carla.Rotation(yaw=3.0004)) # distance=(0.2402)
                camera_init_trans_RU2 = carla.Transform(carla.Location(z=2.3, x=(0.2396), y=(0.0252)), carla.Rotation(yaw=6.0005)) # distance=(0.2409)
                camera_init_trans_RU3 = carla.Transform(carla.Location(z=2.3, x=(0.2391), y=(0.0379)), carla.Rotation(yaw=9.0003)) # distance=(0.2421)
                camera_init_trans_RU4 = carla.Transform(carla.Location(z=2.3, x=(0.2384), y=(0.0507)), carla.Rotation(yaw=12.0004)) # distance=(0.2437)
                camera_init_trans_RU5 = carla.Transform(carla.Location(z=2.3, x=(0.2375), y=(0.0636)), carla.Rotation(yaw=15.0002)) # distance=(0.2459)
                camera_init_trans_RU6 = carla.Transform(carla.Location(z=2.3, x=(0.2363), y=(0.0768)), carla.Rotation(yaw=18.0000)) # distance=(0.2485)
                camera_init_trans_RU7 = carla.Transform(carla.Location(z=2.3, x=(0.2349), y=(0.0902)), carla.Rotation(yaw=21.0004)) # distance=(0.2516)
                camera_init_trans_RU8 = carla.Transform(carla.Location(z=2.3, x=(0.2332), y=(0.1038)), carla.Rotation(yaw=24.0000)) # distance=(0.2553)
                camera_init_trans_RU9 = carla.Transform(carla.Location(z=2.3, x=(0.2312), y=(0.1178)), carla.Rotation(yaw=27.0000)) # distance=(0.2595)
                camera_init_trans_RU10 = carla.Transform(carla.Location(z=2.3, x=(0.2289), y=(0.1322)), carla.Rotation(yaw=30.0000)) # distance=(0.2643)
                camera_init_trans_RU11 = carla.Transform(carla.Location(z=2.3, x=(0.2262), y=(0.1469)), carla.Rotation(yaw=33.0003)) # distance=(0.2697)
                camera_init_trans_RU12 = carla.Transform(carla.Location(z=2.3, x=(0.2231), y=(0.1621)), carla.Rotation(yaw=36.0002)) # distance=(0.2758)
                camera_init_trans_RU13 = carla.Transform(carla.Location(z=2.3, x=(0.2195), y=(0.1778)), carla.Rotation(yaw=39.0003)) # distance=(0.2825)
                camera_init_trans_RU14 = carla.Transform(carla.Location(z=2.3, x=(0.2154), y=(0.1940)), carla.Rotation(yaw=42.0003)) # distance=(0.2899)
                camera_init_trans_RU15 = carla.Transform(carla.Location(z=2.3, x=(0.2107), y=(0.2107)), carla.Rotation(yaw=45.0002)) # distance=(0.2980)
                camera_init_trans_RU16 = carla.Transform(carla.Location(z=2.3, x=(0.2053), y=(0.2280)), carla.Rotation(yaw=48.0004)) # distance=(0.3068)
                camera_init_trans_RU17 = carla.Transform(carla.Location(z=2.3, x=(0.1991), y=(0.2458)), carla.Rotation(yaw=51.0001)) # distance=(0.3163)
                camera_init_trans_RU18 = carla.Transform(carla.Location(z=2.3, x=(0.1919), y=(0.2642)), carla.Rotation(yaw=54.0000)) # distance=(0.3265)
                camera_init_trans_RU19 = carla.Transform(carla.Location(z=2.3, x=(0.1838), y=(0.2830)), carla.Rotation(yaw=57.0002)) # distance=(0.3374)
                camera_init_trans_RU20 = carla.Transform(carla.Location(z=2.3, x=(0.1745), y=(0.3022)), carla.Rotation(yaw=60.0000)) # distance=(0.3489)
                camera_init_trans_RU21 = carla.Transform(carla.Location(z=2.3, x=(0.1638), y=(0.3215)), carla.Rotation(yaw=63.0002)) # distance=(0.3609)
                camera_init_trans_RU22 = carla.Transform(carla.Location(z=2.3, x=(0.1518), y=(0.3409)), carla.Rotation(yaw=66.0000)) # distance=(0.3731)
                camera_init_trans_RU23 = carla.Transform(carla.Location(z=2.3, x=(0.1381), y=(0.3598)), carla.Rotation(yaw=69.0000)) # distance=(0.3854)
                camera_init_trans_RU24 = carla.Transform(carla.Location(z=2.3, x=(0.1228), y=(0.3780)), carla.Rotation(yaw=72.0002)) # distance=(0.3975)
                camera_init_trans_RU25 = carla.Transform(carla.Location(z=2.3, x=(0.1058), y=(0.3949)), carla.Rotation(yaw=75.0001)) # distance=(0.4089)
                camera_init_trans_RU26 = carla.Transform(carla.Location(z=2.3, x=(0.0871), y=(0.4100)), carla.Rotation(yaw=78.0001)) # distance=(0.4191)
                camera_init_trans_RU27 = carla.Transform(carla.Location(z=2.3, x=(0.0669), y=(0.4225)), carla.Rotation(yaw=81.0000)) # distance=(0.4278)
                camera_init_trans_RU28 = carla.Transform(carla.Location(z=2.3, x=(0.0454), y=(0.4321)), carla.Rotation(yaw=84.0001)) # distance=(0.4344)
                camera_init_trans_RU29 = carla.Transform(carla.Location(z=2.3, x=(0.0230), y=(0.4380)), carla.Rotation(yaw=87.0000)) # distance=(0.4386)

            elif args.Camera_set_sh == 'base_plus05_LU':
                print('====LU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=2.3, x=0.24), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=2.3, x=-0.24), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=2.3, x=0.0, y=-0.44), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=2.3, x=(0.0), y=0.44), carla.Rotation(yaw=90))
                
                camera_init_trans_LU1 = carla.Transform(carla.Location(z=2.3, x=(0.2399), y=(-0.0126)), carla.Rotation(yaw=-3.0004)) # distance=(0.2402)
                camera_init_trans_LU2 = carla.Transform(carla.Location(z=2.3, x=(0.2396), y=(-0.0252)), carla.Rotation(yaw=-6.0005)) # distance=(0.2409)
                camera_init_trans_LU3 = carla.Transform(carla.Location(z=2.3, x=(0.2391), y=(-0.0379)), carla.Rotation(yaw=-9.0003)) # distance=(0.2421)
                camera_init_trans_LU4 = carla.Transform(carla.Location(z=2.3, x=(0.2384), y=(-0.0507)), carla.Rotation(yaw=-12.0004)) # distance=(0.2437)
                camera_init_trans_LU5 = carla.Transform(carla.Location(z=2.3, x=(0.2375), y=(-0.0636)), carla.Rotation(yaw=-15.0002)) # distance=(0.2459)
                camera_init_trans_LU6 = carla.Transform(carla.Location(z=2.3, x=(0.2363), y=(-0.0768)), carla.Rotation(yaw=-18.0000)) # distance=(0.2485)
                camera_init_trans_LU7 = carla.Transform(carla.Location(z=2.3, x=(0.2349), y=(-0.0902)), carla.Rotation(yaw=-21.0004)) # distance=(0.2516)
                camera_init_trans_LU8 = carla.Transform(carla.Location(z=2.3, x=(0.2332), y=(-0.1038)), carla.Rotation(yaw=-24.0000)) # distance=(0.2553)
                camera_init_trans_LU9 = carla.Transform(carla.Location(z=2.3, x=(0.2312), y=(-0.1178)), carla.Rotation(yaw=-27.0000)) # distance=(0.2595)
                camera_init_trans_LU10 = carla.Transform(carla.Location(z=2.3, x=(0.2289), y=(-0.1322)), carla.Rotation(yaw=-30.0000)) # distance=(0.2643)
                camera_init_trans_LU11 = carla.Transform(carla.Location(z=2.3, x=(0.2262), y=(-0.1469)), carla.Rotation(yaw=-33.0003)) # distance=(0.2697)
                camera_init_trans_LU12 = carla.Transform(carla.Location(z=2.3, x=(0.2231), y=(-0.1621)), carla.Rotation(yaw=-36.0002)) # distance=(0.2758)
                camera_init_trans_LU13 = carla.Transform(carla.Location(z=2.3, x=(0.2195), y=(-0.1778)), carla.Rotation(yaw=-39.0003)) # distance=(0.2825)
                camera_init_trans_LU14 = carla.Transform(carla.Location(z=2.3, x=(0.2154), y=(-0.1940)), carla.Rotation(yaw=-42.0003)) # distance=(0.2899)
                camera_init_trans_LU15 = carla.Transform(carla.Location(z=2.3, x=(0.2107), y=(-0.2107)), carla.Rotation(yaw=-45.0002)) # distance=(0.2980)
                camera_init_trans_LU16 = carla.Transform(carla.Location(z=2.3, x=(0.2053), y=(-0.2280)), carla.Rotation(yaw=-48.0004)) # distance=(0.3068)
                camera_init_trans_LU17 = carla.Transform(carla.Location(z=2.3, x=(0.1991), y=(-0.2458)), carla.Rotation(yaw=-51.0001)) # distance=(0.3163)
                camera_init_trans_LU18 = carla.Transform(carla.Location(z=2.3, x=(0.1919), y=(-0.2642)), carla.Rotation(yaw=-54.0000)) # distance=(0.3265)
                camera_init_trans_LU19 = carla.Transform(carla.Location(z=2.3, x=(0.1838), y=(-0.2830)), carla.Rotation(yaw=-57.0002)) # distance=(0.3374)
                camera_init_trans_LU20 = carla.Transform(carla.Location(z=2.3, x=(0.1745), y=(-0.3022)), carla.Rotation(yaw=-60.0000)) # distance=(0.3489)
                camera_init_trans_LU21 = carla.Transform(carla.Location(z=2.3, x=(0.1638), y=(-0.3215)), carla.Rotation(yaw=-63.0002)) # distance=(0.3609)
                camera_init_trans_LU22 = carla.Transform(carla.Location(z=2.3, x=(0.1518), y=(-0.3409)), carla.Rotation(yaw=-66.0000)) # distance=(0.3731)
                camera_init_trans_LU23 = carla.Transform(carla.Location(z=2.3, x=(0.1381), y=(-0.3598)), carla.Rotation(yaw=-69.0000)) # distance=(0.3854)
                camera_init_trans_LU24 = carla.Transform(carla.Location(z=2.3, x=(0.1228), y=(-0.3780)), carla.Rotation(yaw=-72.0002)) # distance=(0.3975)
                camera_init_trans_LU25 = carla.Transform(carla.Location(z=2.3, x=(0.1058), y=(-0.3949)), carla.Rotation(yaw=-75.0001)) # distance=(0.4089)
                camera_init_trans_LU26 = carla.Transform(carla.Location(z=2.3, x=(0.0871), y=(-0.4100)), carla.Rotation(yaw=-78.0001)) # distance=(0.4191)
                camera_init_trans_LU27 = carla.Transform(carla.Location(z=2.3, x=(0.0669), y=(-0.4225)), carla.Rotation(yaw=-81.0000)) # distance=(0.4278)
                camera_init_trans_LU28 = carla.Transform(carla.Location(z=2.3, x=(0.0454), y=(-0.4321)), carla.Rotation(yaw=-84.0001)) # distance=(0.4344)
                camera_init_trans_LU29 = carla.Transform(carla.Location(z=2.3, x=(0.0230), y=(-0.4380)), carla.Rotation(yaw=-87.0000)) # distance=(0.4386)

            elif args.Camera_set_sh == 'base_plus05_RD':
                print('====RD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=2.3, x=0.24), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=2.3, x=-0.24), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=2.3, x=0.0, y=-0.44), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=2.3, x=(0.0), y=0.44), carla.Rotation(yaw=90))
                
                camera_init_trans_RD1 = carla.Transform(carla.Location(z=2.3, x=(-0.0230), y=(0.4380)), carla.Rotation(yaw=93.0000)) # distance=(0.4386)
                camera_init_trans_RD2 = carla.Transform(carla.Location(z=2.3, x=(-0.0454), y=(0.4321)), carla.Rotation(yaw=96.0001)) # distance=(0.4344)
                camera_init_trans_RD3 = carla.Transform(carla.Location(z=2.3, x=(-0.0669), y=(0.4225)), carla.Rotation(yaw=99.0000)) # distance=(0.4278)
                camera_init_trans_RD4 = carla.Transform(carla.Location(z=2.3, x=(-0.0871), y=(0.4100)), carla.Rotation(yaw=102.0001)) # distance=(0.4191)
                camera_init_trans_RD5 = carla.Transform(carla.Location(z=2.3, x=(-0.1058), y=(0.3949)), carla.Rotation(yaw=105.0001)) # distance=(0.4089)
                camera_init_trans_RD6 = carla.Transform(carla.Location(z=2.3, x=(-0.1228), y=(0.3780)), carla.Rotation(yaw=108.0001)) # distance=(0.3975)
                camera_init_trans_RD7 = carla.Transform(carla.Location(z=2.3, x=(-0.1381), y=(0.3598)), carla.Rotation(yaw=111.0000)) # distance=(0.3854)
                camera_init_trans_RD8 = carla.Transform(carla.Location(z=2.3, x=(-0.1518), y=(0.3409)), carla.Rotation(yaw=114.0000)) # distance=(0.3731)
                camera_init_trans_RD9 = carla.Transform(carla.Location(z=2.3, x=(-0.1638), y=(0.3215)), carla.Rotation(yaw=117.0001)) # distance=(0.3609)
                camera_init_trans_RD10 = carla.Transform(carla.Location(z=2.3, x=(-0.1745), y=(0.3022)), carla.Rotation(yaw=120.0000)) # distance=(0.3489)
                camera_init_trans_RD11 = carla.Transform(carla.Location(z=2.3, x=(-0.1838), y=(0.2830)), carla.Rotation(yaw=123.0001)) # distance=(0.3374)
                camera_init_trans_RD12 = carla.Transform(carla.Location(z=2.3, x=(-0.1919), y=(0.2642)), carla.Rotation(yaw=126.0000)) # distance=(0.3265)
                camera_init_trans_RD13 = carla.Transform(carla.Location(z=2.3, x=(-0.1991), y=(0.2458)), carla.Rotation(yaw=129.0003)) # distance=(0.3163)
                camera_init_trans_RD14 = carla.Transform(carla.Location(z=2.3, x=(-0.2053), y=(0.2280)), carla.Rotation(yaw=132.0001)) # distance=(0.3068)
                camera_init_trans_RD15 = carla.Transform(carla.Location(z=2.3, x=(-0.2107), y=(0.2107)), carla.Rotation(yaw=135.0002)) # distance=(0.2980)
                camera_init_trans_RD16 = carla.Transform(carla.Location(z=2.3, x=(-0.2154), y=(0.1940)), carla.Rotation(yaw=138.0002)) # distance=(0.2899)
                camera_init_trans_RD17 = carla.Transform(carla.Location(z=2.3, x=(-0.2195), y=(0.1778)), carla.Rotation(yaw=141.0002)) # distance=(0.2825)
                camera_init_trans_RD18 = carla.Transform(carla.Location(z=2.3, x=(-0.2231), y=(0.1621)), carla.Rotation(yaw=144.0003)) # distance=(0.2758)
                camera_init_trans_RD19 = carla.Transform(carla.Location(z=2.3, x=(-0.2262), y=(0.1469)), carla.Rotation(yaw=147.0002)) # distance=(0.2697)
                camera_init_trans_RD20 = carla.Transform(carla.Location(z=2.3, x=(-0.2289), y=(0.1322)), carla.Rotation(yaw=150.0000)) # distance=(0.2643)
                camera_init_trans_RD21 = carla.Transform(carla.Location(z=2.3, x=(-0.2312), y=(0.1178)), carla.Rotation(yaw=153.0000)) # distance=(0.2595)
                camera_init_trans_RD22 = carla.Transform(carla.Location(z=2.3, x=(-0.2332), y=(0.1038)), carla.Rotation(yaw=156.0000)) # distance=(0.2553)
                camera_init_trans_RD23 = carla.Transform(carla.Location(z=2.3, x=(-0.2349), y=(0.0902)), carla.Rotation(yaw=159.0002)) # distance=(0.2516)
                camera_init_trans_RD24 = carla.Transform(carla.Location(z=2.3, x=(-0.2363), y=(0.0768)), carla.Rotation(yaw=162.0000)) # distance=(0.2485)
                camera_init_trans_RD25 = carla.Transform(carla.Location(z=2.3, x=(-0.2375), y=(0.0636)), carla.Rotation(yaw=165.0004)) # distance=(0.2459)
                camera_init_trans_RD26 = carla.Transform(carla.Location(z=2.3, x=(-0.2384), y=(0.0507)), carla.Rotation(yaw=168.0003)) # distance=(0.2437)
                camera_init_trans_RD27 = carla.Transform(carla.Location(z=2.3, x=(-0.2391), y=(0.0379)), carla.Rotation(yaw=171.0004)) # distance=(0.2421)
                camera_init_trans_RD28 = carla.Transform(carla.Location(z=2.3, x=(-0.2396), y=(0.0252)), carla.Rotation(yaw=174.0002)) # distance=(0.2409)
                camera_init_trans_RD29 = carla.Transform(carla.Location(z=2.3, x=(-0.2399), y=(0.0126)), carla.Rotation(yaw=177.0002)) # distance=(0.2402)
                
            elif args.Camera_set_sh == 'base_plus05_LD':
                print('====LD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=2.3, x=0.24), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=2.3, x=-0.24), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=2.3, x=0.0, y=-0.44), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=2.3, x=(0.0), y=0.44), carla.Rotation(yaw=90))
                   
                camera_init_trans_LD1 = carla.Transform(carla.Location(z=2.3, x=(-0.0230), y=(-0.4380)), carla.Rotation(yaw=-93.0000)) # distance=(0.4386)
                camera_init_trans_LD2 = carla.Transform(carla.Location(z=2.3, x=(-0.0454), y=(-0.4321)), carla.Rotation(yaw=-96.0001)) # distance=(0.4344)
                camera_init_trans_LD3 = carla.Transform(carla.Location(z=2.3, x=(-0.0669), y=(-0.4225)), carla.Rotation(yaw=-99.0000)) # distance=(0.4278)
                camera_init_trans_LD4 = carla.Transform(carla.Location(z=2.3, x=(-0.0871), y=(-0.4100)), carla.Rotation(yaw=-102.0001)) # distance=(0.4191)
                camera_init_trans_LD5 = carla.Transform(carla.Location(z=2.3, x=(-0.1058), y=(-0.3949)), carla.Rotation(yaw=-105.0001)) # distance=(0.4089)
                camera_init_trans_LD6 = carla.Transform(carla.Location(z=2.3, x=(-0.1228), y=(-0.3780)), carla.Rotation(yaw=-108.0001)) # distance=(0.3975)
                camera_init_trans_LD7 = carla.Transform(carla.Location(z=2.3, x=(-0.1381), y=(-0.3598)), carla.Rotation(yaw=-111.0000)) # distance=(0.3854)
                camera_init_trans_LD8 = carla.Transform(carla.Location(z=2.3, x=(-0.1518), y=(-0.3409)), carla.Rotation(yaw=-114.0000)) # distance=(0.3731)
                camera_init_trans_LD9 = carla.Transform(carla.Location(z=2.3, x=(-0.1638), y=(-0.3215)), carla.Rotation(yaw=-117.0001)) # distance=(0.3609)
                camera_init_trans_LD10 = carla.Transform(carla.Location(z=2.3, x=(-0.1745), y=(-0.3022)), carla.Rotation(yaw=-120.0000)) # distance=(0.3489)
                camera_init_trans_LD11 = carla.Transform(carla.Location(z=2.3, x=(-0.1838), y=(-0.2830)), carla.Rotation(yaw=-123.0001)) # distance=(0.3374)
                camera_init_trans_LD12 = carla.Transform(carla.Location(z=2.3, x=(-0.1919), y=(-0.2642)), carla.Rotation(yaw=-126.0000)) # distance=(0.3265)
                camera_init_trans_LD13 = carla.Transform(carla.Location(z=2.3, x=(-0.1991), y=(-0.2458)), carla.Rotation(yaw=-129.0003)) # distance=(0.3163)
                camera_init_trans_LD14 = carla.Transform(carla.Location(z=2.3, x=(-0.2053), y=(-0.2280)), carla.Rotation(yaw=-132.0001)) # distance=(0.3068)
                camera_init_trans_LD15 = carla.Transform(carla.Location(z=2.3, x=(-0.2107), y=(-0.2107)), carla.Rotation(yaw=-135.0002)) # distance=(0.2980)
                camera_init_trans_LD16 = carla.Transform(carla.Location(z=2.3, x=(-0.2154), y=(-0.1940)), carla.Rotation(yaw=-138.0002)) # distance=(0.2899)
                camera_init_trans_LD17 = carla.Transform(carla.Location(z=2.3, x=(-0.2195), y=(-0.1778)), carla.Rotation(yaw=-141.0002)) # distance=(0.2825)
                camera_init_trans_LD18 = carla.Transform(carla.Location(z=2.3, x=(-0.2231), y=(-0.1621)), carla.Rotation(yaw=-144.0003)) # distance=(0.2758)
                camera_init_trans_LD19 = carla.Transform(carla.Location(z=2.3, x=(-0.2262), y=(-0.1469)), carla.Rotation(yaw=-147.0002)) # distance=(0.2697)
                camera_init_trans_LD20 = carla.Transform(carla.Location(z=2.3, x=(-0.2289), y=(-0.1322)), carla.Rotation(yaw=-150.0000)) # distance=(0.2643)
                camera_init_trans_LD21 = carla.Transform(carla.Location(z=2.3, x=(-0.2312), y=(-0.1178)), carla.Rotation(yaw=-153.0000)) # distance=(0.2595)
                camera_init_trans_LD22 = carla.Transform(carla.Location(z=2.3, x=(-0.2332), y=(-0.1038)), carla.Rotation(yaw=-156.0000)) # distance=(0.2553)
                camera_init_trans_LD23 = carla.Transform(carla.Location(z=2.3, x=(-0.2349), y=(-0.0902)), carla.Rotation(yaw=-159.0002)) # distance=(0.2516)
                camera_init_trans_LD24 = carla.Transform(carla.Location(z=2.3, x=(-0.2363), y=(-0.0768)), carla.Rotation(yaw=-162.0000)) # distance=(0.2485)
                camera_init_trans_LD25 = carla.Transform(carla.Location(z=2.3, x=(-0.2375), y=(-0.0636)), carla.Rotation(yaw=-165.0004)) # distance=(0.2459)
                camera_init_trans_LD26 = carla.Transform(carla.Location(z=2.3, x=(-0.2384), y=(-0.0507)), carla.Rotation(yaw=-168.0003)) # distance=(0.2437)
                camera_init_trans_LD27 = carla.Transform(carla.Location(z=2.3, x=(-0.2391), y=(-0.0379)), carla.Rotation(yaw=-171.0004)) # distance=(0.2421)
                camera_init_trans_LD28 = carla.Transform(carla.Location(z=2.3, x=(-0.2396), y=(-0.0252)), carla.Rotation(yaw=-174.0002)) # distance=(0.2409)
                camera_init_trans_LD29 = carla.Transform(carla.Location(z=2.3, x=(-0.2399), y=(-0.0126)), carla.Rotation(yaw=-177.0002)) # distance=(0.2402)

            elif args.Camera_set_sh == 'test_00_RU': # 1.97 30
                print('====RU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=0.01), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=2.3, x=-0.44), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=2.3, x=(0.0), y=-0.25), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.01), carla.Rotation(yaw=90))

                camera_init_trans_RU1 = carla.Transform(carla.Location(z=1.762, x=(0.0100), y=(0.0005)), carla.Rotation(yaw=3.0002)) # distance=(0.0100)
                camera_init_trans_RU2 = carla.Transform(carla.Location(z=1.762, x=(0.0099), y=(0.0010)), carla.Rotation(yaw=6.0001)) # distance=(0.0100)
                camera_init_trans_RU3 = carla.Transform(carla.Location(z=1.762, x=(0.0099), y=(0.0016)), carla.Rotation(yaw=9.0000)) # distance=(0.0100)
                camera_init_trans_RU4 = carla.Transform(carla.Location(z=1.762, x=(0.0098), y=(0.0021)), carla.Rotation(yaw=12.0002)) # distance=(0.0100)
                camera_init_trans_RU5 = carla.Transform(carla.Location(z=1.762, x=(0.0097), y=(0.0026)), carla.Rotation(yaw=15.0001)) # distance=(0.0100)
                camera_init_trans_RU6 = carla.Transform(carla.Location(z=1.762, x=(0.0095), y=(0.0031)), carla.Rotation(yaw=18.0000)) # distance=(0.0100)
                camera_init_trans_RU7 = carla.Transform(carla.Location(z=1.762, x=(0.0093), y=(0.0036)), carla.Rotation(yaw=21.0002)) # distance=(0.0100)
                camera_init_trans_RU8 = carla.Transform(carla.Location(z=1.762, x=(0.0091), y=(0.0041)), carla.Rotation(yaw=24.0001)) # distance=(0.0100)
                camera_init_trans_RU9 = carla.Transform(carla.Location(z=1.762, x=(0.0089), y=(0.0045)), carla.Rotation(yaw=27.0000)) # distance=(0.0100)
                camera_init_trans_RU10 = carla.Transform(carla.Location(z=1.762, x=(0.0087), y=(0.0050)), carla.Rotation(yaw=30.0002)) # distance=(0.0100)
                camera_init_trans_RU11 = carla.Transform(carla.Location(z=1.762, x=(0.0084), y=(0.0054)), carla.Rotation(yaw=33.0001)) # distance=(0.0100)
                camera_init_trans_RU12 = carla.Transform(carla.Location(z=1.762, x=(0.0081), y=(0.0059)), carla.Rotation(yaw=36.0000)) # distance=(0.0100)
                camera_init_trans_RU13 = carla.Transform(carla.Location(z=1.762, x=(0.0078), y=(0.0063)), carla.Rotation(yaw=39.0002)) # distance=(0.0100)
                camera_init_trans_RU14 = carla.Transform(carla.Location(z=1.762, x=(0.0074), y=(0.0067)), carla.Rotation(yaw=42.0001)) # distance=(0.0100)
                camera_init_trans_RU15 = carla.Transform(carla.Location(z=1.762, x=(0.0071), y=(0.0071)), carla.Rotation(yaw=45.0000)) # distance=(0.0100)
                camera_init_trans_RU16 = carla.Transform(carla.Location(z=1.762, x=(0.0067), y=(0.0074)), carla.Rotation(yaw=48.0002)) # distance=(0.0100)
                camera_init_trans_RU17 = carla.Transform(carla.Location(z=1.762, x=(0.0063), y=(0.0078)), carla.Rotation(yaw=51.0001)) # distance=(0.0100)
                camera_init_trans_RU18 = carla.Transform(carla.Location(z=1.762, x=(0.0059), y=(0.0081)), carla.Rotation(yaw=54.0000)) # distance=(0.0100)
                camera_init_trans_RU19 = carla.Transform(carla.Location(z=1.762, x=(0.0054), y=(0.0084)), carla.Rotation(yaw=57.0002)) # distance=(0.0100)
                camera_init_trans_RU20 = carla.Transform(carla.Location(z=1.762, x=(0.0050), y=(0.0087)), carla.Rotation(yaw=60.0001)) # distance=(0.0100)
                camera_init_trans_RU21 = carla.Transform(carla.Location(z=1.762, x=(0.0045), y=(0.0089)), carla.Rotation(yaw=63.0000)) # distance=(0.0100)
                camera_init_trans_RU22 = carla.Transform(carla.Location(z=1.762, x=(0.0041), y=(0.0091)), carla.Rotation(yaw=66.0002)) # distance=(0.0100)
                camera_init_trans_RU23 = carla.Transform(carla.Location(z=1.762, x=(0.0036), y=(0.0093)), carla.Rotation(yaw=69.0001)) # distance=(0.0100)
                camera_init_trans_RU24 = carla.Transform(carla.Location(z=1.762, x=(0.0031), y=(0.0095)), carla.Rotation(yaw=72.0000)) # distance=(0.0100)
                camera_init_trans_RU25 = carla.Transform(carla.Location(z=1.762, x=(0.0026), y=(0.0097)), carla.Rotation(yaw=75.0002)) # distance=(0.0100)
                camera_init_trans_RU26 = carla.Transform(carla.Location(z=1.762, x=(0.0021), y=(0.0098)), carla.Rotation(yaw=78.0001)) # distance=(0.0100)
                camera_init_trans_RU27 = carla.Transform(carla.Location(z=1.762, x=(0.0016), y=(0.0099)), carla.Rotation(yaw=81.0000)) # distance=(0.0100)
                camera_init_trans_RU28 = carla.Transform(carla.Location(z=1.762, x=(0.0010), y=(0.0099)), carla.Rotation(yaw=84.0002)) # distance=(0.0100)
                camera_init_trans_RU29 = carla.Transform(carla.Location(z=1.762, x=(0.0005), y=(0.0100)), carla.Rotation(yaw=87.0001)) # distance=(0.0100)

            elif args.Camera_set_sh == 'test_00_LU':
                print('====LU Camera Set activate====') 
                # Set initial camera translation
                camera_init_trans_F = carla.Transform(carla.Location(z=1.762, x=0.01), carla.Rotation(yaw=0))
                # camera_init_trans_B = carla.Transform(carla.Location(z=2.3, x=-0.44), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.01), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=2.3, x=(0.0), y=0.25), carla.Rotation(yaw=90))
                
                camera_init_trans_LU1 = carla.Transform(carla.Location(z=1.762, x=(0.0100), y=(-0.0005)), carla.Rotation(yaw=-3.0002)) # distance=(0.0100)
                camera_init_trans_LU2 = carla.Transform(carla.Location(z=1.762, x=(0.0099), y=(-0.0010)), carla.Rotation(yaw=-6.0001)) # distance=(0.0100)
                camera_init_trans_LU3 = carla.Transform(carla.Location(z=1.762, x=(0.0099), y=(-0.0016)), carla.Rotation(yaw=-9.0000)) # distance=(0.0100)
                camera_init_trans_LU4 = carla.Transform(carla.Location(z=1.762, x=(0.0098), y=(-0.0021)), carla.Rotation(yaw=-12.0002)) # distance=(0.0100)
                camera_init_trans_LU5 = carla.Transform(carla.Location(z=1.762, x=(0.0097), y=(-0.0026)), carla.Rotation(yaw=-15.0001)) # distance=(0.0100)
                camera_init_trans_LU6 = carla.Transform(carla.Location(z=1.762, x=(0.0095), y=(-0.0031)), carla.Rotation(yaw=-18.0000)) # distance=(0.0100)
                camera_init_trans_LU7 = carla.Transform(carla.Location(z=1.762, x=(0.0093), y=(-0.0036)), carla.Rotation(yaw=-21.0002)) # distance=(0.0100)
                camera_init_trans_LU8 = carla.Transform(carla.Location(z=1.762, x=(0.0091), y=(-0.0041)), carla.Rotation(yaw=-24.0001)) # distance=(0.0100)
                camera_init_trans_LU9 = carla.Transform(carla.Location(z=1.762, x=(0.0089), y=(-0.0045)), carla.Rotation(yaw=-27.0000)) # distance=(0.0100)
                camera_init_trans_LU10 = carla.Transform(carla.Location(z=1.762, x=(0.0087), y=(-0.0050)), carla.Rotation(yaw=-30.0002)) # distance=(0.0100)
                camera_init_trans_LU11 = carla.Transform(carla.Location(z=1.762, x=(0.0084), y=(-0.0054)), carla.Rotation(yaw=-33.0001)) # distance=(0.0100)
                camera_init_trans_LU12 = carla.Transform(carla.Location(z=1.762, x=(0.0081), y=(-0.0059)), carla.Rotation(yaw=-36.0000)) # distance=(0.0100)
                camera_init_trans_LU13 = carla.Transform(carla.Location(z=1.762, x=(0.0078), y=(-0.0063)), carla.Rotation(yaw=-39.0002)) # distance=(0.0100)
                camera_init_trans_LU14 = carla.Transform(carla.Location(z=1.762, x=(0.0074), y=(-0.0067)), carla.Rotation(yaw=-42.0001)) # distance=(0.0100)
                camera_init_trans_LU15 = carla.Transform(carla.Location(z=1.762, x=(0.0071), y=(-0.0071)), carla.Rotation(yaw=-45.0000)) # distance=(0.0100)
                camera_init_trans_LU16 = carla.Transform(carla.Location(z=1.762, x=(0.0067), y=(-0.0074)), carla.Rotation(yaw=-48.0002)) # distance=(0.0100)
                camera_init_trans_LU17 = carla.Transform(carla.Location(z=1.762, x=(0.0063), y=(-0.0078)), carla.Rotation(yaw=-51.0001)) # distance=(0.0100)
                camera_init_trans_LU18 = carla.Transform(carla.Location(z=1.762, x=(0.0059), y=(-0.0081)), carla.Rotation(yaw=-54.0000)) # distance=(0.0100)
                camera_init_trans_LU19 = carla.Transform(carla.Location(z=1.762, x=(0.0054), y=(-0.0084)), carla.Rotation(yaw=-57.0002)) # distance=(0.0100)
                camera_init_trans_LU20 = carla.Transform(carla.Location(z=1.762, x=(0.0050), y=(-0.0087)), carla.Rotation(yaw=-60.0001)) # distance=(0.0100)
                camera_init_trans_LU21 = carla.Transform(carla.Location(z=1.762, x=(0.0045), y=(-0.0089)), carla.Rotation(yaw=-63.0000)) # distance=(0.0100)
                camera_init_trans_LU22 = carla.Transform(carla.Location(z=1.762, x=(0.0041), y=(-0.0091)), carla.Rotation(yaw=-66.0002)) # distance=(0.0100)
                camera_init_trans_LU23 = carla.Transform(carla.Location(z=1.762, x=(0.0036), y=(-0.0093)), carla.Rotation(yaw=-69.0001)) # distance=(0.0100)
                camera_init_trans_LU24 = carla.Transform(carla.Location(z=1.762, x=(0.0031), y=(-0.0095)), carla.Rotation(yaw=-72.0000)) # distance=(0.0100)
                camera_init_trans_LU25 = carla.Transform(carla.Location(z=1.762, x=(0.0026), y=(-0.0097)), carla.Rotation(yaw=-75.0002)) # distance=(0.0100)
                camera_init_trans_LU26 = carla.Transform(carla.Location(z=1.762, x=(0.0021), y=(-0.0098)), carla.Rotation(yaw=-78.0001)) # distance=(0.0100)
                camera_init_trans_LU27 = carla.Transform(carla.Location(z=1.762, x=(0.0016), y=(-0.0099)), carla.Rotation(yaw=-81.0000)) # distance=(0.0100)
                camera_init_trans_LU28 = carla.Transform(carla.Location(z=1.762, x=(0.0010), y=(-0.0099)), carla.Rotation(yaw=-84.0002)) # distance=(0.0100)
                camera_init_trans_LU29 = carla.Transform(carla.Location(z=1.762, x=(0.0005), y=(-0.0100)), carla.Rotation(yaw=-87.0001)) # distance=(0.0100)

            elif args.Camera_set_sh == 'test_00_RD':
                print('====RD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=2.3, x=0.44), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-0.01), carla.Rotation(yaw=180))
                # camera_init_trans_L = carla.Transform(carla.Location(z=2.3, x=(0.0), y=-0.25), carla.Rotation(yaw=-90))
                camera_init_trans_R = carla.Transform(carla.Location(z=1.762, x=(0.0), y=0.01), carla.Rotation(yaw=90))

                
                camera_init_trans_RD1 = carla.Transform(carla.Location(z=1.762, x=(-0.0005), y=(0.0100)), carla.Rotation(yaw=93.0002)) # distance=(0.0100)
                camera_init_trans_RD2 = carla.Transform(carla.Location(z=1.762, x=(-0.0010), y=(0.0099)), carla.Rotation(yaw=96.0001)) # distance=(0.0100)
                camera_init_trans_RD3 = carla.Transform(carla.Location(z=1.762, x=(-0.0016), y=(0.0099)), carla.Rotation(yaw=99.0000)) # distance=(0.0100)
                camera_init_trans_RD4 = carla.Transform(carla.Location(z=1.762, x=(-0.0021), y=(0.0098)), carla.Rotation(yaw=102.0002)) # distance=(0.0100)
                camera_init_trans_RD5 = carla.Transform(carla.Location(z=1.762, x=(-0.0026), y=(0.0097)), carla.Rotation(yaw=105.0001)) # distance=(0.0100)
                camera_init_trans_RD6 = carla.Transform(carla.Location(z=1.762, x=(-0.0031), y=(0.0095)), carla.Rotation(yaw=108.0000)) # distance=(0.0100)
                camera_init_trans_RD7 = carla.Transform(carla.Location(z=1.762, x=(-0.0036), y=(0.0093)), carla.Rotation(yaw=111.0002)) # distance=(0.0100)
                camera_init_trans_RD8 = carla.Transform(carla.Location(z=1.762, x=(-0.0041), y=(0.0091)), carla.Rotation(yaw=114.0001)) # distance=(0.0100)
                camera_init_trans_RD9 = carla.Transform(carla.Location(z=1.762, x=(-0.0045), y=(0.0089)), carla.Rotation(yaw=117.0000)) # distance=(0.0100)
                camera_init_trans_RD10 = carla.Transform(carla.Location(z=1.762, x=(-0.0050), y=(0.0087)), carla.Rotation(yaw=120.0002)) # distance=(0.0100)
                camera_init_trans_RD11 = carla.Transform(carla.Location(z=1.762, x=(-0.0054), y=(0.0084)), carla.Rotation(yaw=123.0001)) # distance=(0.0100)
                camera_init_trans_RD12 = carla.Transform(carla.Location(z=1.762, x=(-0.0059), y=(0.0081)), carla.Rotation(yaw=126.0000)) # distance=(0.0100)
                camera_init_trans_RD13 = carla.Transform(carla.Location(z=1.762, x=(-0.0063), y=(0.0078)), carla.Rotation(yaw=129.0002)) # distance=(0.0100)
                camera_init_trans_RD14 = carla.Transform(carla.Location(z=1.762, x=(-0.0067), y=(0.0074)), carla.Rotation(yaw=132.0001)) # distance=(0.0100)
                camera_init_trans_RD15 = carla.Transform(carla.Location(z=1.762, x=(-0.0071), y=(0.0071)), carla.Rotation(yaw=135.0000)) # distance=(0.0100)
                camera_init_trans_RD16 = carla.Transform(carla.Location(z=1.762, x=(-0.0074), y=(0.0067)), carla.Rotation(yaw=138.0002)) # distance=(0.0100)
                camera_init_trans_RD17 = carla.Transform(carla.Location(z=1.762, x=(-0.0078), y=(0.0063)), carla.Rotation(yaw=141.0001)) # distance=(0.0100)
                camera_init_trans_RD18 = carla.Transform(carla.Location(z=1.762, x=(-0.0081), y=(0.0059)), carla.Rotation(yaw=144.0000)) # distance=(0.0100)
                camera_init_trans_RD19 = carla.Transform(carla.Location(z=1.762, x=(-0.0084), y=(0.0054)), carla.Rotation(yaw=147.0002)) # distance=(0.0100)
                camera_init_trans_RD20 = carla.Transform(carla.Location(z=1.762, x=(-0.0087), y=(0.0050)), carla.Rotation(yaw=150.0001)) # distance=(0.0100)
                camera_init_trans_RD21 = carla.Transform(carla.Location(z=1.762, x=(-0.0089), y=(0.0045)), carla.Rotation(yaw=153.0000)) # distance=(0.0100)
                camera_init_trans_RD22 = carla.Transform(carla.Location(z=1.762, x=(-0.0091), y=(0.0041)), carla.Rotation(yaw=156.0002)) # distance=(0.0100)
                camera_init_trans_RD23 = carla.Transform(carla.Location(z=1.762, x=(-0.0093), y=(0.0036)), carla.Rotation(yaw=159.0001)) # distance=(0.0100)
                camera_init_trans_RD24 = carla.Transform(carla.Location(z=1.762, x=(-0.0095), y=(0.0031)), carla.Rotation(yaw=162.0000)) # distance=(0.0100)
                camera_init_trans_RD25 = carla.Transform(carla.Location(z=1.762, x=(-0.0097), y=(0.0026)), carla.Rotation(yaw=165.0002)) # distance=(0.0100)
                camera_init_trans_RD26 = carla.Transform(carla.Location(z=1.762, x=(-0.0098), y=(0.0021)), carla.Rotation(yaw=168.0001)) # distance=(0.0100)
                camera_init_trans_RD27 = carla.Transform(carla.Location(z=1.762, x=(-0.0099), y=(0.0016)), carla.Rotation(yaw=171.0000)) # distance=(0.0100)
                camera_init_trans_RD28 = carla.Transform(carla.Location(z=1.762, x=(-0.0099), y=(0.0010)), carla.Rotation(yaw=174.0002)) # distance=(0.0100)
                camera_init_trans_RD29 = carla.Transform(carla.Location(z=1.762, x=(-0.0100), y=(0.0005)), carla.Rotation(yaw=177.0001)) # distance=(0.0100)
            
            elif args.Camera_set_sh == 'test_00_LD':
                print('====LD Camera Set activate====') 
                # Set initial camera translation
                # camera_init_trans_F = carla.Transform(carla.Location(z=2.3, x=0.44), carla.Rotation(yaw=0))
                camera_init_trans_B = carla.Transform(carla.Location(z=1.762, x=-0.01), carla.Rotation(yaw=180))
                camera_init_trans_L = carla.Transform(carla.Location(z=1.762, x=(0.0), y=-0.01), carla.Rotation(yaw=-90))
                # camera_init_trans_R = carla.Transform(carla.Location(z=2.3, x=(0.0), y=0.25), carla.Rotation(yaw=90))
                   
                camera_init_trans_LD1 = carla.Transform(carla.Location(z=1.762, x=(-0.0005), y=(-0.0100)), carla.Rotation(yaw=-93.0002)) # distance=(0.0100)
                camera_init_trans_LD2 = carla.Transform(carla.Location(z=1.762, x=(-0.0010), y=(-0.0099)), carla.Rotation(yaw=-96.0001)) # distance=(0.0100)
                camera_init_trans_LD3 = carla.Transform(carla.Location(z=1.762, x=(-0.0016), y=(-0.0099)), carla.Rotation(yaw=-99.0000)) # distance=(0.0100)
                camera_init_trans_LD4 = carla.Transform(carla.Location(z=1.762, x=(-0.0021), y=(-0.0098)), carla.Rotation(yaw=-102.0002)) # distance=(0.0100)
                camera_init_trans_LD5 = carla.Transform(carla.Location(z=1.762, x=(-0.0026), y=(-0.0097)), carla.Rotation(yaw=-105.0001)) # distance=(0.0100)
                camera_init_trans_LD6 = carla.Transform(carla.Location(z=1.762, x=(-0.0031), y=(-0.0095)), carla.Rotation(yaw=-108.0000)) # distance=(0.0100)
                camera_init_trans_LD7 = carla.Transform(carla.Location(z=1.762, x=(-0.0036), y=(-0.0093)), carla.Rotation(yaw=-111.0002)) # distance=(0.0100)
                camera_init_trans_LD8 = carla.Transform(carla.Location(z=1.762, x=(-0.0041), y=(-0.0091)), carla.Rotation(yaw=-114.0001)) # distance=(0.0100)
                camera_init_trans_LD9 = carla.Transform(carla.Location(z=1.762, x=(-0.0045), y=(-0.0089)), carla.Rotation(yaw=-117.0000)) # distance=(0.0100)
                camera_init_trans_LD10 = carla.Transform(carla.Location(z=1.762, x=(-0.0050), y=(-0.0087)), carla.Rotation(yaw=-120.0002)) # distance=(0.0100)
                camera_init_trans_LD11 = carla.Transform(carla.Location(z=1.762, x=(-0.0054), y=(-0.0084)), carla.Rotation(yaw=-123.0001)) # distance=(0.0100)
                camera_init_trans_LD12 = carla.Transform(carla.Location(z=1.762, x=(-0.0059), y=(-0.0081)), carla.Rotation(yaw=-126.0000)) # distance=(0.0100)
                camera_init_trans_LD13 = carla.Transform(carla.Location(z=1.762, x=(-0.0063), y=(-0.0078)), carla.Rotation(yaw=-129.0002)) # distance=(0.0100)
                camera_init_trans_LD14 = carla.Transform(carla.Location(z=1.762, x=(-0.0067), y=(-0.0074)), carla.Rotation(yaw=-132.0001)) # distance=(0.0100)
                camera_init_trans_LD15 = carla.Transform(carla.Location(z=1.762, x=(-0.0071), y=(-0.0071)), carla.Rotation(yaw=-135.0000)) # distance=(0.0100)
                camera_init_trans_LD16 = carla.Transform(carla.Location(z=1.762, x=(-0.0074), y=(-0.0067)), carla.Rotation(yaw=-138.0002)) # distance=(0.0100)
                camera_init_trans_LD17 = carla.Transform(carla.Location(z=1.762, x=(-0.0078), y=(-0.0063)), carla.Rotation(yaw=-141.0001)) # distance=(0.0100)
                camera_init_trans_LD18 = carla.Transform(carla.Location(z=1.762, x=(-0.0081), y=(-0.0059)), carla.Rotation(yaw=-144.0000)) # distance=(0.0100)
                camera_init_trans_LD19 = carla.Transform(carla.Location(z=1.762, x=(-0.0084), y=(-0.0054)), carla.Rotation(yaw=-147.0002)) # distance=(0.0100)
                camera_init_trans_LD20 = carla.Transform(carla.Location(z=1.762, x=(-0.0087), y=(-0.0050)), carla.Rotation(yaw=-150.0001)) # distance=(0.0100)
                camera_init_trans_LD21 = carla.Transform(carla.Location(z=1.762, x=(-0.0089), y=(-0.0045)), carla.Rotation(yaw=-153.0000)) # distance=(0.0100)
                camera_init_trans_LD22 = carla.Transform(carla.Location(z=1.762, x=(-0.0091), y=(-0.0041)), carla.Rotation(yaw=-156.0002)) # distance=(0.0100)
                camera_init_trans_LD23 = carla.Transform(carla.Location(z=1.762, x=(-0.0093), y=(-0.0036)), carla.Rotation(yaw=-159.0001)) # distance=(0.0100)
                camera_init_trans_LD24 = carla.Transform(carla.Location(z=1.762, x=(-0.0095), y=(-0.0031)), carla.Rotation(yaw=-162.0000)) # distance=(0.0100)
                camera_init_trans_LD25 = carla.Transform(carla.Location(z=1.762, x=(-0.0097), y=(-0.0026)), carla.Rotation(yaw=-165.0002)) # distance=(0.0100)
                camera_init_trans_LD26 = carla.Transform(carla.Location(z=1.762, x=(-0.0098), y=(-0.0021)), carla.Rotation(yaw=-168.0001)) # distance=(0.0100)
                camera_init_trans_LD27 = carla.Transform(carla.Location(z=1.762, x=(-0.0099), y=(-0.0016)), carla.Rotation(yaw=-171.0000)) # distance=(0.0100)
                camera_init_trans_LD28 = carla.Transform(carla.Location(z=1.762, x=(-0.0099), y=(-0.0010)), carla.Rotation(yaw=-174.0002)) # distance=(0.0100)
                camera_init_trans_LD29 = carla.Transform(carla.Location(z=1.762, x=(-0.0100), y=(-0.0005)), carla.Rotation(yaw=-177.0001)) # distance=(0.0100)


            if ("RU" in args.Camera_set_sh):
                F_camera_bp = bp_lib.find('sensor.camera.fisheye')
                F_camera_bp = setup_camera(bp_lib, world, vehicle_sh, F_camera_bp, camera_init_trans_F, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                # B_camera_bp = bp_lib.find('sensor.camera.fisheye')
                # B_camera_bp = setup_camera(bp_lib, world, vehicle_sh, B_camera_bp, camera_init_trans_B, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                # L_camera_bp = bp_lib.find('sensor.camera.fisheye')
                # L_camera_bp = setup_camera(bp_lib, world, vehicle_sh, L_camera_bp, camera_init_trans_L, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                R_camera_bp = bp_lib.find('sensor.camera.fisheye')
                R_camera_bp = setup_camera(bp_lib, world, vehicle_sh, R_camera_bp, camera_init_trans_R, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU1_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU1_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU1_camera_bp, camera_init_trans_RU1, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU2_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU2_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU2_camera_bp, camera_init_trans_RU2, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU3_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU3_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU3_camera_bp, camera_init_trans_RU3, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU4_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU4_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU4_camera_bp, camera_init_trans_RU4, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU5_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU5_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU5_camera_bp, camera_init_trans_RU5, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU6_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU6_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU6_camera_bp, camera_init_trans_RU6, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU7_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU7_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU7_camera_bp, camera_init_trans_RU7, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU8_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU8_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU8_camera_bp, camera_init_trans_RU8, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU9_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU9_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU9_camera_bp, camera_init_trans_RU9, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU10_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU10_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU10_camera_bp, camera_init_trans_RU10, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU11_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU11_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU11_camera_bp, camera_init_trans_RU11, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU12_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU12_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU12_camera_bp, camera_init_trans_RU12, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU13_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU13_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU13_camera_bp, camera_init_trans_RU13, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU14_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU14_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU14_camera_bp, camera_init_trans_RU14, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU15_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU15_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU15_camera_bp, camera_init_trans_RU15, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU16_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU16_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU16_camera_bp, camera_init_trans_RU16, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU17_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU17_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU17_camera_bp, camera_init_trans_RU17, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU18_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU18_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU18_camera_bp, camera_init_trans_RU18, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU19_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU19_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU19_camera_bp, camera_init_trans_RU19, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU20_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU20_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU20_camera_bp, camera_init_trans_RU20, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU21_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU21_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU21_camera_bp, camera_init_trans_RU21, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU22_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU22_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU22_camera_bp, camera_init_trans_RU22, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU23_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU23_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU23_camera_bp, camera_init_trans_RU23, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU24_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU24_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU24_camera_bp, camera_init_trans_RU24, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU25_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU25_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU25_camera_bp, camera_init_trans_RU25, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU26_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU26_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU26_camera_bp, camera_init_trans_RU26, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU27_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU27_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU27_camera_bp, camera_init_trans_RU27, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU28_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU28_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU28_camera_bp, camera_init_trans_RU28, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RU29_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RU29_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RU29_camera_bp, camera_init_trans_RU29, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                
                
                print('====Fisheye camera Done====\n')
                
            elif ("LU" in args.Camera_set_sh):
                F_camera_bp = bp_lib.find('sensor.camera.fisheye')
                F_camera_bp = setup_camera(bp_lib, world, vehicle_sh, F_camera_bp, camera_init_trans_F, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                # B_camera_bp = bp_lib.find('sensor.camera.fisheye')
                # B_camera_bp = setup_camera(bp_lib, world, vehicle_sh, B_camera_bp, camera_init_trans_B, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                L_camera_bp = bp_lib.find('sensor.camera.fisheye')
                L_camera_bp = setup_camera(bp_lib, world, vehicle_sh, L_camera_bp, camera_init_trans_L, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                # R_camera_bp = bp_lib.find('sensor.camera.fisheye')
                # R_camera_bp = setup_camera(bp_lib, world, vehicle_sh, R_camera_bp, camera_init_trans_R, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU1_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU1_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU1_camera_bp, camera_init_trans_LU1, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU2_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU2_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU2_camera_bp, camera_init_trans_LU2, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU3_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU3_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU3_camera_bp, camera_init_trans_LU3, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU4_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU4_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU4_camera_bp, camera_init_trans_LU4, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU5_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU5_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU5_camera_bp, camera_init_trans_LU5, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU6_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU6_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU6_camera_bp, camera_init_trans_LU6, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU7_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU7_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU7_camera_bp, camera_init_trans_LU7, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU8_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU8_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU8_camera_bp, camera_init_trans_LU8, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU9_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU9_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU9_camera_bp, camera_init_trans_LU9, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU10_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU10_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU10_camera_bp, camera_init_trans_LU10, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU11_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU11_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU11_camera_bp, camera_init_trans_LU11, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU12_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU12_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU12_camera_bp, camera_init_trans_LU12, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU13_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU13_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU13_camera_bp, camera_init_trans_LU13, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU14_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU14_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU14_camera_bp, camera_init_trans_LU14, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU15_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU15_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU15_camera_bp, camera_init_trans_LU15, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU16_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU16_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU16_camera_bp, camera_init_trans_LU16, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU17_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU17_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU17_camera_bp, camera_init_trans_LU17, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU18_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU18_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU18_camera_bp, camera_init_trans_LU18, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU19_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU19_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU19_camera_bp, camera_init_trans_LU19, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU20_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU20_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU20_camera_bp, camera_init_trans_LU20, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU21_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU21_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU21_camera_bp, camera_init_trans_LU21, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU22_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU22_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU22_camera_bp, camera_init_trans_LU22, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU23_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU23_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU23_camera_bp, camera_init_trans_LU23, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU24_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU24_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU24_camera_bp, camera_init_trans_LU24, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU25_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU25_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU25_camera_bp, camera_init_trans_LU25, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU26_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU26_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU26_camera_bp, camera_init_trans_LU26, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU27_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU27_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU27_camera_bp, camera_init_trans_LU27, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU28_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU28_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU28_camera_bp, camera_init_trans_LU28, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LU29_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LU29_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LU29_camera_bp, camera_init_trans_LU29, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                
                
                print('====Fisheye camera Done====\n')
                
            elif ("RD" in args.Camera_set_sh):
                # F_camera_bp = bp_lib.find('sensor.camera.fisheye')
                # F_camera_bp = setup_camera(bp_lib, world, vehicle_sh, F_camera_bp, camera_init_trans_F, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                B_camera_bp = bp_lib.find('sensor.camera.fisheye')
                B_camera_bp = setup_camera(bp_lib, world, vehicle_sh, B_camera_bp, camera_init_trans_B, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                # L_camera_bp = bp_lib.find('sensor.camera.fisheye')
                # L_camera_bp = setup_camera(bp_lib, world, vehicle_sh, L_camera_bp, camera_init_trans_L, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                R_camera_bp = bp_lib.find('sensor.camera.fisheye')
                R_camera_bp = setup_camera(bp_lib, world, vehicle_sh, R_camera_bp, camera_init_trans_R, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD1_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD1_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD1_camera_bp, camera_init_trans_RD1, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD2_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD2_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD2_camera_bp, camera_init_trans_RD2, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD3_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD3_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD3_camera_bp, camera_init_trans_RD3, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD4_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD4_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD4_camera_bp, camera_init_trans_RD4, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD5_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD5_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD5_camera_bp, camera_init_trans_RD5, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD6_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD6_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD6_camera_bp, camera_init_trans_RD6, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD7_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD7_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD7_camera_bp, camera_init_trans_RD7, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD8_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD8_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD8_camera_bp, camera_init_trans_RD8, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD9_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD9_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD9_camera_bp, camera_init_trans_RD9, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD10_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD10_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD10_camera_bp, camera_init_trans_RD10, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD11_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD11_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD11_camera_bp, camera_init_trans_RD11, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD12_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD12_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD12_camera_bp, camera_init_trans_RD12, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD13_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD13_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD13_camera_bp, camera_init_trans_RD13, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD14_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD14_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD14_camera_bp, camera_init_trans_RD14, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD15_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD15_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD15_camera_bp, camera_init_trans_RD15, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD16_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD16_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD16_camera_bp, camera_init_trans_RD16, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD17_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD17_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD17_camera_bp, camera_init_trans_RD17, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD18_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD18_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD18_camera_bp, camera_init_trans_RD18, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD19_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD19_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD19_camera_bp, camera_init_trans_RD19, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD20_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD20_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD20_camera_bp, camera_init_trans_RD20, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD21_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD21_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD21_camera_bp, camera_init_trans_RD21, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD22_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD22_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD22_camera_bp, camera_init_trans_RD22, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD23_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD23_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD23_camera_bp, camera_init_trans_RD23, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD24_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD24_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD24_camera_bp, camera_init_trans_RD24, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD25_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD25_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD25_camera_bp, camera_init_trans_RD25, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD26_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD26_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD26_camera_bp, camera_init_trans_RD26, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD27_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD27_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD27_camera_bp, camera_init_trans_RD27, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD28_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD28_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD28_camera_bp, camera_init_trans_RD28, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                RD29_camera_bp = bp_lib.find('sensor.camera.fisheye')
                RD29_camera_bp = setup_camera(bp_lib, world, vehicle_sh, RD29_camera_bp, camera_init_trans_RD29, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                
                
                print('====Fisheye camera Done====\n')
                
            elif ("LD" in args.Camera_set_sh):
                # F_camera_bp = bp_lib.find('sensor.camera.fisheye')
                # F_camera_bp = setup_camera(bp_lib, world, vehicle_sh, F_camera_bp, camera_init_trans_F, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                B_camera_bp = bp_lib.find('sensor.camera.fisheye')
                B_camera_bp = setup_camera(bp_lib, world, vehicle_sh, B_camera_bp, camera_init_trans_B, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                L_camera_bp = bp_lib.find('sensor.camera.fisheye')
                L_camera_bp = setup_camera(bp_lib, world, vehicle_sh, L_camera_bp, camera_init_trans_L, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                # R_camera_bp = bp_lib.find('sensor.camera.fisheye')
                # R_camera_bp = setup_camera(bp_lib, world, vehicle_sh, R_camera_bp, camera_init_trans_R, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD1_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD1_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD1_camera_bp, camera_init_trans_LD1, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD2_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD2_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD2_camera_bp, camera_init_trans_LD2, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD3_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD3_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD3_camera_bp, camera_init_trans_LD3, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD4_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD4_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD4_camera_bp, camera_init_trans_LD4, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD5_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD5_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD5_camera_bp, camera_init_trans_LD5, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD6_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD6_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD6_camera_bp, camera_init_trans_LD6, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD7_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD7_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD7_camera_bp, camera_init_trans_LD7, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD8_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD8_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD8_camera_bp, camera_init_trans_LD8, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD9_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD9_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD9_camera_bp, camera_init_trans_LD9, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD10_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD10_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD10_camera_bp, camera_init_trans_LD10, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD11_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD11_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD11_camera_bp, camera_init_trans_LD11, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD12_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD12_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD12_camera_bp, camera_init_trans_LD12, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD13_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD13_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD13_camera_bp, camera_init_trans_LD13, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD14_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD14_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD14_camera_bp, camera_init_trans_LD14, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD15_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD15_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD15_camera_bp, camera_init_trans_LD15, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD16_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD16_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD16_camera_bp, camera_init_trans_LD16, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD17_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD17_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD17_camera_bp, camera_init_trans_LD17, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD18_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD18_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD18_camera_bp, camera_init_trans_LD18, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD19_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD19_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD19_camera_bp, camera_init_trans_LD19, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD20_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD20_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD20_camera_bp, camera_init_trans_LD20, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD21_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD21_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD21_camera_bp, camera_init_trans_LD21, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD22_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD22_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD22_camera_bp, camera_init_trans_LD22, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD23_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD23_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD23_camera_bp, camera_init_trans_LD23, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD24_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD24_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD24_camera_bp, camera_init_trans_LD24, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD25_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD25_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD25_camera_bp, camera_init_trans_LD25, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD26_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD26_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD26_camera_bp, camera_init_trans_LD26, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD27_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD27_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD27_camera_bp, camera_init_trans_LD27, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD28_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD28_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD28_camera_bp, camera_init_trans_LD28, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                LD29_camera_bp = bp_lib.find('sensor.camera.fisheye')
                LD29_camera_bp = setup_camera(bp_lib, world, vehicle_sh, LD29_camera_bp, camera_init_trans_LD29, image_w, image_h, fovset, focallengh_x, focallengh_y, optical_x, optical_y)
                
                
                print('====Fisheye camera Done====\n')
                
                
                
        if args.safe:
            # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            # blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        #spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
            args.number_of_vehicles = (number_of_spawn_points // float(args.Num_car_sh)) - int(args.index)
            # args.number_of_vehicles = (number_of_spawn_points // 5) - int(args.index)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            if transform == 155 and args.Map_sh == 'Town10HD': # check 1029
                continue
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if args.car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                
        if (args.number_of_walkers > 0):
            # -------------
            # Spawn Walkers
            # -------------
            # some settings
            if args.Map_sh == "Town05":
                args.number_of_walkers = 100
                print(f"Change walkers \n")
            percentagePedestriansRunning = 0.1      # how many pedestrians will run
            percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
            # 1. take all the random locations to spawn
            spawn_points = []
            for i in range(args.number_of_walkers):
                spawn_point = carla.Transform()
                loc = world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if (random.random() > percentagePedestriansRunning):
                        # walking
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    else:
                        # running
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = client.apply_batch_sync(batch, True) ## check
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put altogether the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = world.get_actors(all_id)

            # wait for a tick to ensure client receives the last transform of the walkers we have just created
            if not args.sync or not synchronous_master: ## check
                world.wait_for_tick()
            else:
                world.tick()

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
            # set how many pedestrians can cross the road
            world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(world.get_random_location_from_navigation())
                # max speed
                all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

            print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)
        
        waitingtime = ((int(args.seed) // 1123) * 15) + 10
        # number of cameras test isetta spawn=147 Num_car_sh=1.5
        # waitingtime = ((int(args.seed) // 1123) * 20) + 19
        
        for i in range(int(waitingtime)): # Suspension time 10
            world.tick()
            print("Waiting progress {:2.1%}".format((i+1)/(int(waitingtime))), end="\r")
        
        if record:
            if ("RU" in args.Camera_set_sh):
                print('====Record Start====\n')
                # Sensor synchronize
                sensor_list = []

                # Set sensors recording
                F_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "000"))
                sensor_list.append(F_camera_bp)
                # B_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "B"))
                # sensor_list.append(B_camera_bp)
                # L_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "L"))
                # sensor_list.append(L_camera_bp)
                R_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "030"))
                sensor_list.append(R_camera_bp)

                RU1_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "001"))
                sensor_list.append(RU1_camera_bp)
                RU2_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "002"))
                sensor_list.append(RU2_camera_bp)
                RU3_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "003"))
                sensor_list.append(RU3_camera_bp)
                RU4_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "004"))
                sensor_list.append(RU4_camera_bp)
                RU5_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "005"))
                sensor_list.append(RU5_camera_bp)
                RU6_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "006"))
                sensor_list.append(RU6_camera_bp)
                RU7_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "007"))
                sensor_list.append(RU7_camera_bp)
                RU8_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "008"))
                sensor_list.append(RU8_camera_bp)
                RU9_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "009"))
                sensor_list.append(RU9_camera_bp)
                RU10_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "010"))
                sensor_list.append(RU10_camera_bp)
                RU11_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "011"))
                sensor_list.append(RU11_camera_bp)
                RU12_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "012"))
                sensor_list.append(RU12_camera_bp)
                RU13_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "013"))
                sensor_list.append(RU13_camera_bp)
                RU14_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "014"))
                sensor_list.append(RU14_camera_bp)
                RU15_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "015"))
                sensor_list.append(RU15_camera_bp)
                RU16_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "016"))
                sensor_list.append(RU16_camera_bp)
                RU17_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "017"))
                sensor_list.append(RU17_camera_bp)
                RU18_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "018"))
                sensor_list.append(RU18_camera_bp)
                RU19_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "019"))
                sensor_list.append(RU19_camera_bp)
                RU20_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "020"))
                sensor_list.append(RU20_camera_bp)
                RU21_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "021"))
                sensor_list.append(RU21_camera_bp)
                RU22_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "022"))
                sensor_list.append(RU22_camera_bp)
                RU23_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "023"))
                sensor_list.append(RU23_camera_bp)
                RU24_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "024"))
                sensor_list.append(RU24_camera_bp)
                RU25_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "025"))
                sensor_list.append(RU25_camera_bp)
                RU26_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "026"))
                sensor_list.append(RU26_camera_bp)
                RU27_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "027"))
                sensor_list.append(RU27_camera_bp)
                RU28_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "028"))
                sensor_list.append(RU28_camera_bp)
                RU29_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "029"))
                sensor_list.append(RU29_camera_bp)
                
            elif ("LU" in args.Camera_set_sh):
                print('====Record Start====\n')
                # Sensor synchronize
                sensor_list = []

                # Set sensors recording
                F_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "120"))
                sensor_list.append(F_camera_bp)
                # B_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "B"))
                # sensor_list.append(B_camera_bp)
                L_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "090"))
                sensor_list.append(L_camera_bp)
                # R_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "R"))
                # sensor_list.append(R_camera_bp)

                LU1_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "119"))
                sensor_list.append(LU1_camera_bp)
                LU2_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "118"))
                sensor_list.append(LU2_camera_bp)
                LU3_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "117"))
                sensor_list.append(LU3_camera_bp)
                LU4_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "116"))
                sensor_list.append(LU4_camera_bp)
                LU5_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "115"))
                sensor_list.append(LU5_camera_bp)
                LU6_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "114"))
                sensor_list.append(LU6_camera_bp)
                LU7_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "113"))
                sensor_list.append(LU7_camera_bp)
                LU8_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "112"))
                sensor_list.append(LU8_camera_bp)
                LU9_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "111"))
                sensor_list.append(LU9_camera_bp)
                LU10_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "109"))
                sensor_list.append(LU10_camera_bp)
                LU11_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "108"))
                sensor_list.append(LU11_camera_bp)
                LU12_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "107"))
                sensor_list.append(LU12_camera_bp)
                LU13_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "106"))
                sensor_list.append(LU13_camera_bp)
                LU14_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "105"))
                sensor_list.append(LU14_camera_bp)
                LU15_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "104"))
                sensor_list.append(LU15_camera_bp)
                LU16_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "103"))
                sensor_list.append(LU16_camera_bp)
                LU17_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "102"))
                sensor_list.append(LU17_camera_bp)
                LU18_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "101"))
                sensor_list.append(LU18_camera_bp)
                LU19_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "100"))
                sensor_list.append(LU19_camera_bp)
                LU20_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "099"))
                sensor_list.append(LU20_camera_bp)
                LU21_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "098"))
                sensor_list.append(LU21_camera_bp)
                LU22_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "097"))
                sensor_list.append(LU22_camera_bp)
                LU23_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "096"))
                sensor_list.append(LU23_camera_bp)
                LU24_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "095"))
                sensor_list.append(LU24_camera_bp)
                LU25_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "094"))
                sensor_list.append(LU25_camera_bp)
                LU26_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "093"))
                sensor_list.append(LU26_camera_bp)
                LU27_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "092"))
                sensor_list.append(LU27_camera_bp)
                LU28_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "091"))
                sensor_list.append(LU28_camera_bp)
                LU29_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "090"))
                sensor_list.append(LU29_camera_bp)
                
            elif ("RD" in args.Camera_set_sh):
                print('====Record Start====\n')
                # Sensor synchronize
                sensor_list = []

                # Set sensors recording
                # F_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "F"))
                # sensor_list.append(F_camera_bp)
                B_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "060"))
                sensor_list.append(B_camera_bp)
                # L_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "L"))
                # sensor_list.append(L_camera_bp)
                R_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "030"))
                sensor_list.append(R_camera_bp)

                RD1_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "031"))
                sensor_list.append(RD1_camera_bp)
                RD2_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "032"))
                sensor_list.append(RD2_camera_bp)
                RD3_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "033"))
                sensor_list.append(RD3_camera_bp)
                RD4_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "034"))
                sensor_list.append(RD4_camera_bp)
                RD5_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "035"))
                sensor_list.append(RD5_camera_bp)
                RD6_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "036"))
                sensor_list.append(RD6_camera_bp)
                RD7_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "037"))
                sensor_list.append(RD7_camera_bp)
                RD8_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "038"))
                sensor_list.append(RD8_camera_bp)
                RD9_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "039"))
                sensor_list.append(RD9_camera_bp)
                RD10_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "040"))
                sensor_list.append(RD10_camera_bp)
                RD11_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "041"))
                sensor_list.append(RD11_camera_bp)
                RD12_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "042"))
                sensor_list.append(RD12_camera_bp)
                RD13_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "043"))
                sensor_list.append(RD13_camera_bp)
                RD14_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "044"))
                sensor_list.append(RD14_camera_bp)
                RD15_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "045"))
                sensor_list.append(RD15_camera_bp)
                RD16_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "046"))
                sensor_list.append(RD16_camera_bp)
                RD17_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "047"))
                sensor_list.append(RD17_camera_bp)
                RD18_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "048"))
                sensor_list.append(RD18_camera_bp)
                RD19_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "049"))
                sensor_list.append(RD19_camera_bp)
                RD20_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "050"))
                sensor_list.append(RD20_camera_bp)
                RD21_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "051"))
                sensor_list.append(RD21_camera_bp)
                RD22_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "052"))
                sensor_list.append(RD22_camera_bp)
                RD23_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "053"))
                sensor_list.append(RD23_camera_bp)
                RD24_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "054"))
                sensor_list.append(RD24_camera_bp)
                RD25_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "055"))
                sensor_list.append(RD25_camera_bp)
                RD26_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "056"))
                sensor_list.append(RD26_camera_bp)
                RD27_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "057"))
                sensor_list.append(RD27_camera_bp)
                RD28_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "058"))
                sensor_list.append(RD28_camera_bp)
                RD29_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "059"))
                sensor_list.append(RD29_camera_bp)
                
            elif ("LD" in args.Camera_set_sh):
                print('====Record Start====\n')
                # Sensor synchronize
                sensor_list = []

                # Set sensors recording
                # F_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "F"))
                # sensor_list.append(F_camera_bp)
                B_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "060"))
                sensor_list.append(B_camera_bp)
                L_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "090"))
                sensor_list.append(L_camera_bp)
                # R_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "R"))
                # sensor_list.append(R_camera_bp)

                LD1_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "089"))
                sensor_list.append(LD1_camera_bp)
                LD2_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "088"))
                sensor_list.append(LD2_camera_bp)
                LD3_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "087"))
                sensor_list.append(LD3_camera_bp)
                LD4_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "086"))
                sensor_list.append(LD4_camera_bp)
                LD5_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "085"))
                sensor_list.append(LD5_camera_bp)
                LD6_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "084"))
                sensor_list.append(LD6_camera_bp)
                LD7_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "083"))
                sensor_list.append(LD7_camera_bp)
                LD8_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "082"))
                sensor_list.append(LD8_camera_bp)
                LD9_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "081"))
                sensor_list.append(LD9_camera_bp)
                LD10_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "080"))
                sensor_list.append(LD10_camera_bp)
                LD11_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "079"))
                sensor_list.append(LD11_camera_bp)
                LD12_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "078"))
                sensor_list.append(LD12_camera_bp)
                LD13_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "077"))
                sensor_list.append(LD13_camera_bp)
                LD14_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "076"))
                sensor_list.append(LD14_camera_bp)
                LD15_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "075"))
                sensor_list.append(LD15_camera_bp)
                LD16_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "074"))
                sensor_list.append(LD16_camera_bp)
                LD17_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "073"))
                sensor_list.append(LD17_camera_bp)
                LD18_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "072"))
                sensor_list.append(LD18_camera_bp)
                LD19_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "071"))
                sensor_list.append(LD19_camera_bp)
                LD20_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "070"))
                sensor_list.append(LD20_camera_bp)
                LD21_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "069"))
                sensor_list.append(LD21_camera_bp)
                LD22_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "068"))
                sensor_list.append(LD22_camera_bp)
                LD23_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "067"))
                sensor_list.append(LD23_camera_bp)
                LD24_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "066"))
                sensor_list.append(LD24_camera_bp)
                LD25_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "065"))
                sensor_list.append(LD25_camera_bp)
                LD26_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "064"))
                sensor_list.append(LD26_camera_bp)
                LD27_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "063"))
                sensor_list.append(LD27_camera_bp)
                LD28_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "062"))
                sensor_list.append(LD28_camera_bp)
                LD29_camera_bp.listen(lambda data: sensor_callback(data, sensor_queue, "061"))
                sensor_list.append(LD29_camera_bp)
            
            framenum = 0
            frameend = 30
            while framenum < (frameend+1):
                if args.sync and synchronous_master:
                    world.tick()
                    time.sleep(0.2) # 0.2
                    print("Progress {:2.1%}".format(framenum / frameend), end="\r")
                    try:
                        for _ in range(len(sensor_list)):
                            s_frame = sensor_queue.get(True, 30.0)
                    except Empty:
                        print("Some of the sensor information is missed")
                        break
                        #sys.exit()
                        #os.system("shutdown -r -t 30")
                    framenum += 1
                else:
                    world.wait_for_tick()

    finally:
        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
            
        time.sleep(0.5)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        killcarla()
        time.sleep(3)
        print('\ndone.')
