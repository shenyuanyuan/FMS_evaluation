# FMS_evaluation

## steps to run 
1. data used in this project can be downloaded from [here](https://doi.org/10.25452/figshare.plus.c.5774969)
1. use conda to build experiment environment
```shell
conda env create -f conda_env.yml
```
1. cd evaluation/st_gcn
1. you can run the following pythons scripts in that directory
    1. evaluation/st_gcn/front_view_all_joints.py # get data from front view and use all the joints 
    1. evaluation/st_gcn/front_view_part_joints.py # get data from front view and use important joints
    1. evaluation/st_gcn/front_view_part_joints_split_by_person.py # get data from front view, use important joints and split training/test set by persons
1. Directories CNN-Based and LSTM-Based follow the similar running methods as above
