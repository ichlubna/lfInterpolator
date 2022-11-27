from evaluation import evaluator as eva
from evaluation import basher as bash
import tempfile
import shutil
import sys
import os
import traceback

binaryPath = "../build/lfInterpolator"
trajectory = "0.0,0.0,1.0,1.0"

def prepareDirs(temp):
    pathReference = os.path.join(workspace, "reference")
    os.mkdir(pathReference)
    pathResult = os.path.join(workspace, "result")
    os.mkdir(pathResult)
    return pathReference, pathResult

def makeCmd(ref, res, traj):
    command = binaryPath
    command += " -i "+sys.argv[1]
    command += " -t "+traj
    commandRef = command+" -o "+ref
    commandRes = command+" -o "+res
    #commandRes += " --tensor"
    return commandRef, commandRes

def runAndCheck(command):
    print("Running test: "+command)
    result = bash.run(command)
    if(result.returncode != 0):
        print(result.stderr)
        raise Exception("Command not executed.")
    times=[]
    firstSplit=result.stdout.split(": ")
    for frag in firstSplit:
        if " ms" in frag:
            times.append(float(frag.split(" ms")[0]))
    print("Average time of "+str(len(times))+" measurings: "+str(sum(times)/len(times)))

workspace = None
try:
    workspace = tempfile.mkdtemp()
    if len(sys.argv) == 1 or len(sys.argv) > 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        raise Exception("Call with one argument for the input folder as: python measure.py ./path/input")
    dirs = prepareDirs(workspace)
    commands = makeCmd(dirs[0], dirs[1], trajectory)
    runAndCheck(commands[0])
    runAndCheck(commands[1])
    evaluator = eva.Evaluator()
    print(evaluator.metrics(dirs[0], dirs[1]))

except Exception as e:
    print(e)
    print(traceback.format_exc())
finally:
    shutil.rmtree(workspace)
