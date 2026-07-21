from omegaconf import OmegaConf

# From M3ED Repositroy
cfg = OmegaConf.load("dataset_list.yaml") #https://github.com/daniilidis-group/m3ed/blob/main/dataset_list.yaml

filter_data = [
    elem
    for elem in cfg
    if not "calib" in elem.file
]
with open("down.sh", "w") as f:
    for i in range(len(filter_data)):
        f.write(
            f"mkdir -p /tmp/{filter_data[i]['file']}/\n"
        )
        f.write(
            f"aws s3 cp s3://m3ed-dist/processed/{filter_data[i]['file']}/{filter_data[i]['file']}_data.h5  /tmp/{filter_data[i]['file']}/ --no-sign-request& \n"
        )
        f.write(
           f"aws s3 cp s3://m3ed-dist/processed/{filter_data[i]['file']}/{filter_data[i]['file']}_depth_gt.h5  /tmp/{filter_data[i]['file']}/ --no-sign-request \n"
        )
        f.write(
            f"aws s3 cp s3://m3ed-dist/processed/{filter_data[i]['file']}/{filter_data[i]['file']}_pose_gt.h5  /tmp/{filter_data[i]['file']}/ --no-sign-request \n"
        )
