import subprocess

Map_sh = ['Town10HD', 'Town03', 'Town05']
Weather_testset = ['clear', 'rain1', 'dawn']
SpawnP_test_dataset = [25,75,125]

Vehicle_type_sh = ['vehicle.bmw.isetta']
Camera_set_sh = ['Baseline_RU', 'Baseline_LU','Baseline_RD', 'Baseline_LD']
# ['base_08_RU', 'base_08_LU','base_08_RD', 'base_08_LD']
# ['base_05_RU', 'base_05_LU','base_05_RD', 'base_05_LD']
# ['base_00_RU', 'base_00_LU','base_00_RD', 'base_00_LD']

if __name__ == '__main__':
    for M in Map_sh:
        for V in Vehicle_type_sh:  
            for index, SpawnP in enumerate(SpawnP_test_dataset):
                for C in Camera_set_sh:
                    for Seed, W in enumerate(Weather_testset):
                        python_command = f'python GV360_testset.py --index={index} --seed={Seed * 1123} --SpawnP_sh={SpawnP} --Vehicle_type_sh={V} --Camera_set_sh={C} --Weather_sh={W} --Map_sh={M} --Num_car_sh=3'
                        print(python_command)
                        proc = subprocess.Popen(python_command, shell=True)
                        proc.wait()
