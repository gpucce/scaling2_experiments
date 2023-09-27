"""
This is helper script to launch sbatch jobs and to handle two issues
we encountered:

- freezing/hanging
- limited maximum job time (24 hours in the best case, can be 6 hours when total compute budget is over)

The script automatically relaunch the sbatch script when the job either freezes
or is stopped/canceled.

How to use it?

## Step 1

install the clize package using:

`pip install clize`

##  Step 2

Since the script needs to be running indefinitely, we launch a screen:

`screen -S screen_name`

## Step 3

`python autorestart.py "sbatch <your_script.sh> <your_arguments>" --output-file-template="slurm-{job_id}.out" --check-interval-secs=900 --verbose`

It is necessary to replace the `output-file-template` with the one you use since it is the output
file which is used to figure out if the job is freezing or not.
`check-interval-secs` determines the interval by which the job is checked.

## Step 4

CTRL + A then D to leave the screen and keep the script running indefinitely.

"""
import os
import re
import time
from subprocess import call, check_output
from clize import run


def main(
    cmd,
    *,
    output_file_template="slurm-{job_id}_{array_task_id}.out",
    check_interval_secs=60 * 15,
    start_condition="",
    termination_str="",
    verbose=True,
    resume_job_id: int = None,
    resume_array_task_ids: int = None,
):
    cmd_check_job_in_queue = "squeue -r -j {job_id}"
    cmd_check_job_array_in_queue = "squeue -j {job_id}_{array_task_id}"
    cmd_check_job_running = "squeue -j {job_id}_{array_task_id} -t R"
    array_task_ids_to_restart = []
    array_task_ids_done = []
    while True:
        if start_condition:
            if verbose:
                print("Checking start condition...")
            if int(check_output(start_condition, shell=True)) == 0:
                if verbose:
                    print(
                        f"Start condition returned 0, not starting, retrying again in {check_interval_secs//60} mins."
                    )
                time.sleep(check_interval_secs)
                continue

        if verbose:
            print("Launch a new job")
            print(cmd)
        if resume_job_id is not None and resume_array_task_ids is not None:
            job_id = resume_job_id
            array_task_ids = resume_array_task_ids
            resume_job_id = None
            resume_array_task_ids = None
        else:
            # launch job on hold so to prevent immediate execution and get array ids
            output = check_output(
                cmd.replace("sbatch ", "sbatch --hold ")
                if "--hold" not in cmd
                else cmd,
                shell=True,
            ).decode()
            # get job id and all array task ids
            job_id = get_job_id(output).split("\n")[0].split("_")[0]
            array_task_ids = [
                i[len(job_id) + 1 :]
                for i in re.findall(
                    f"${job_id}_\d+", check_output(cmd_check_job_in_queue).decode()
                )
            ]
            array_task_ids = [i for i in array_task_ids if i not in array_task_ids_done]

            # actually start the job if it wasn't meant to be on hold
            if "--hold" not in cmd:
                call(f"scontrol release ${job_id}")

            if job_id is None:
                if verbose:
                    print("Cannot find job id in:")
                    print('"' + output + '"')
                    print(f"Retrying again in {check_interval_secs//60} mins...")
                time.sleep(check_interval_secs)
                continue

        if verbose:
            print("Current job ID:", job_id)
        while True:
            # Infinite-loop, check each `check_interval_secs` whether job is present
            # in the queue, then, if present in the queue check if it is still running
            # and not frozen. The job is relaunched when it is no longuer running or
            # frozen. Then the same process is repeated.

            for array_task_id in array_task_ids:
                try:
                    data = check_output(
                        cmd_check_job_array_in_queue.format(
                            job_id=job_id, array_task_id=array_task_id
                        ),
                        shell=True,
                    ).decode()
                except Exception as ex:
                    # Exception after checking, which means that the job id no longer exists.
                    # In this case, we wait and relaunch, except if termination string is found
                    if verbose:
                        print(ex)
                    if check_if_done(
                        output_file_template.format(
                            job_id=job_id, array_task_id=array_task_id
                        ),
                        termination_str,
                    ):
                        if verbose:
                            print("Termination string found, finishing")
                        array_task_ids_done.append(array_task_id)
                    continue

                # if job is not present in the queue, relaunch it directly, except if termination string is found
                if str(job_id) + "_" + str(array_task_id) not in data:
                    if check_if_done(
                        output_file_template.format(
                            job_id=job_id, array_task_id=array_task_id
                        ),
                        termination_str,
                    ):
                        if verbose:
                            print("Termination string found, finishing")
                        array_task_ids_done.append(array_task_id)
                        continue

                # Check first if job is specifically on a running state (to avoid the case where it is on pending state etc)
                data = check_output(
                    cmd_check_job_running.format(
                        job_id=job_id, array_task_id=array_task_id
                    ),
                    shell=True,
                ).decode()
                if str(job_id) + "_" + str(array_task_id) in data:
                    # job on running state
                    output_file = output_file_template.format(
                        job_id=job_id, array_task_id=array_task_id
                    )
                    if verbose:
                        print("Check if the job is freezing...")
                    # if job is on running state, check the output file
                    output_data_prev = get_file_content(output_file)
                    # wait few minutes
                    time.sleep(check_interval_secs)
                    # check again the output file
                    output_data = get_file_content(output_file)
                    # if the file did not change, then it is considered
                    # to be frozen
                    # (make sure there are is output before checking)
                    if (
                        output_data
                        and output_data_prev
                        and output_data == output_data_prev
                    ):
                        if verbose:
                            print("Job frozen, stopping the job then restarting it")
                        call(f"scancel {job_id}_{array_task_id}", shell=True)
                        continue

            array_task_ids = [i for i in array_task_ids if i not in array_task_ids_done]

            if not array_task_ids:
                break

            if verbose:
                print(f"Retrying again in {check_interval_secs//60} mins...")

            time.sleep(check_interval_secs)


def check_if_done(logfile, termination_str):
    return (
        os.path.exists(logfile)
        and (termination_str != "")
        and re.search(termination_str, open(logfile).read())
    )


def get_file_content(output_file):
    return open(output_file).read()


def get_job_id(s):
    try:
        return int(re.search("Submitted batch job ([0-9_]+)", s).group(1))
    except Exception:
        return None


if __name__ == "__main__":
    run(main)
