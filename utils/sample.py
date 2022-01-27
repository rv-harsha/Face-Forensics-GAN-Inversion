import os

def listdirs(rootdir):
    for it in os.scandir(rootdir):
        if it.is_dir():
            print(it.name)
            listdirs(it)

outdir = '/nas/vista-hdd01/users/raidurga/outputs/ai2ai_run_004/expts'
listdirs(outdir)