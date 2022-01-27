import pandas as pd
import os
import os.path as osp

job_status_csv_path = (
    "/nas/vista-hdd01/users/raidurga/outputs/ai2ai_run_004/expts/job-status.csv"
)
batch_info = "/nas/vista-hdd01/users/raidurga/workspace/aifi/dataset/celebA/batch-info"

def modify_value(x):
    return x.split("_")[1]

df = pd.read_csv(job_status_csv_path, header=None, index_col=False, error_bad_lines=False)
df = df[(df[2] == "COMPLETED")]
completed_jobs = df[1].apply(modify_value)
df1 = df.assign(file=pd.Series(completed_jobs).values)["file"]

all_jobs = []

for csvfile in os.listdir(batch_info):
    file = csvfile.rsplit("/")[-1].split(".")[0]
    all_jobs.append(file)

df2 = pd.DataFrame({"file": all_jobs})
left_join_df = pd.merge(df2, df1, on="file", how="left", indicator=True)
df = left_join_df[(left_join_df["_merge"] == "left_only")]["file"]
# df = df.sample(n=368)
save_to = osp.join(os.path.dirname(job_status_csv_path), "reprocess.csv")
df.to_csv(save_to, index=False, header=None)