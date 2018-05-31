
## deprecated


import os


scriptPath = 'D:\projects\RL\snake\hackathon\hackathon\slither.py'
python_3_path = r"C:\rafi\RL\code\AirSim\AirSim-1.1.7\PythonClient\venv\Scripts\python.exe"
folder = r'D:\projects\RL\snake\hackathon\rafi\models\saved_models'
folder_logs = r'D:\projects\RL\snake\hackathon\rafi\models\logs'
def compare_runs(files_by_time):
    for json_name in files_by_time.values():
        params="-P PolicyRafaelTest(model_path="+os.path.join(folder,json_name)+")"
        params += " -m 1 - bs(10, 10) - od 0 - pwt 0.2 - -random_food_prob 0.2 - -food_map(  # ,2,+1) --game_duration 1000  --policy_wait_time 0.1"
        params += "--log_file " +os.path.join(folder_logs,json_name,".log")
        # slither.run(params)
        command = python_3_path + " " + scriptPath + " " + params
        os.system(command)
        a=1
if __name__ == '__main__':

    files_names = os.listdir(folder)
    files_by_time = dict()
    for name in files_names:
        if 'json' in name:
            ts = int(name.split('.')[0][6:])
            files_by_time[ts] = os.path.splitext(name)[0]

    compare_runs(files_by_time)

    a=1