import subprocess

Map_sh = ['Town10HD'] 
# 'Town10HD', 'Town03', 'Town05'
Weather_trainset = ['default', 'clear', 'night', 'overcast', 'overcast1', 'overcast2', 'rain', 'rain1', 'dawn']

SpawnP_trainset_m3 = [198, 66, 99, 132, 165, 33]
SpawnP_trainset_m5 = [159, 178, 210, 151, 293]
SpawnP_trainset_m10 = [118, 93, 147, 134, 99, 88]
SpawnP_trainset_test = [20, 40, 60, 80, 100, 120]

Vehicle_type_sh = ['vehicle.bmw.isetta']
# 'vehicle.bmw.isetta' 'vehicle.audi.etron' 'vehicle.volkswagen.t2'

Camera_set_sh = ['Baseline_RU', 'Baseline_LU','Baseline_RD', 'Baseline_LD']

# ['Baseline_RU', 'Baseline_LU','Baseline_RD', 'Baseline_LD']
# ['BaselineA_RU', 'BaselineA_LU','BaselineA_RD', 'BaselineA_LD']
# ['Pluto_plus_RU', 'Pluto_plus_LU','Pluto_plus_RD', 'Pluto_plus_LD']
# ['Indago3_RU', 'Indago3_LU','Indago3_RD', 'Indago3_LD']
# ['base_plus05_RU', 'base_plus05_LU','base_plus05_RD', 'base_plus05_LD']

if __name__ == '__main__':
    for M in Map_sh:
        for V in Vehicle_type_sh:  
            for index, SpawnP in enumerate(SpawnP_trainset_m10):
                for C in Camera_set_sh:
                    for Seed, W in enumerate(Weather_trainset):
                        python_command = f'python GV360_trainset.py --index={index} --seed={Seed * 1123} --SpawnP_sh={SpawnP} --Vehicle_type_sh={V} --Camera_set_sh={C} --Weather_sh={W} --Map_sh={M} --Num_car_sh=3'
                        print(python_command)
                        proc = subprocess.Popen(python_command, shell=True)
                        proc.wait()
