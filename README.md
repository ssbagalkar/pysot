# PySOT Testing and Evaluation Tutorial

This implements testing of the trained model and evaluates its performance on different experiments and models. This repository will not deal with any training currently.

### Confirm presence of PySOT in your PYTHONPATH
Pysot path has already been added to the `bashrc` file. In case, you face issues like 
```
ModuleNotFoundError: No module named 'pysot'
```
do
```bash
export PYTHONPATH=export PYTHONPATH=/home/ubuntu/object_tracking/pysot/:$PYTHONPATH
```

### Testing
So, testing will run the model against a set of images, generally grouped together in separate folders based on its labels/class. 
This set of images also have its groundtruth ( labels, bounding box coordinates) known. We set a desired IOU threshold(eg:0.85) 
and when the tracker fails to track the object of interest, meaning goes below the specified IOU, we reset the tracker using the known groundtruth data for that frame

Note: We also set `burning-rate`- which tells the model how many frames to skip, before it starts to re-track again after a failure. This is essential
as sometimes, there is occlusion for a few frames and a good re-initization might be essential.

### Groundtruth Data Format

All associated groundtruth is in the ebs-volume `saurabh-OT-umich` --> `evaluation_data`
* Each folder name is organized according to the Dataset, for example, VOT2019


    | -- VOT_datasets
 
        | -- VOT2019
            | -- agility
                | -- camera_motion.tag
                | -- color
                    | -- 00000001.jpg
                    | -- 00000002.jpg
                    | -- ......
                    | -- 00000100.jpg
                | -- groundtruth.txt
                | -- illum_change.tag
                | -- occlusion.tag
                | -- size_change.tag
            | -- girl
                | -- ...
                | -- color
                    | -- .....
                | -- ...
            | -- ...
            | -- ...
            | -- VOT2019.json
   Note: the `.tag` files only make sense when using VOT20XX datasets. For custom models, we will ignore those. For example, umich/Sauron
   
    | -- Sauron-umich
        | -- Sauron
            | --  Cystotome
                | -- frame001671.png
                | -- framexxxx.png
                | -- ....
            | -- keratome
                | -- ...
            | -- ...
            | -- Sauron.json
  Note: The groudtruth data( bounding box coordinates) is converted from topo-chico to VOT-like dataset and placed in `Sauron.json` .
  
 
### Run OT Testing scripts
* `ssh` into the instance `OT-eval-tests`
* `source pytorch_p36` <-- this enables pytorch and the mandatory cuda environment
* mount the ebs volume if it is not already mounted to ec-2 instance. check with `lsblk` and then `sudo mount /dev/xvdf[check this location] /mountdir`
* `cd /home/ubuntu/object-tracking` and run
```
python pysot/tools/test.py 
--dataset VOT2019 --config pysot/experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml 
--snapshot /mountdir/14_videos_Nov_2020/results/generic_model_snapshot/model.pth 
--model_name SiamRPNpp --experiment_name generic --results_path /mountdir/evaluation_results 
--dataset_directory /mountdir/evaluation_data/VOT_datasets
```
* For running a custom trained model, like umich, you can do the following:
```
python pysot/tools/test.py 
--dataset Sauron --config pysot/experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml 
--snapshot /mountdir/14_videos_Nov_2020/results/snapshot/checkpoint_e9.pth --model_name SiamRPNpp 
--experiment_name checkpoint_e9_ec2 --results_path /mountdir/evaluation_results 
--dataset_directory /mountdir/evaluation_data/Sauron-umich
```

So both the above scripts will test your models against the test data and generate text files in the results directory
according to its class/labels in this location: `/mountdir/evaluation_results/Sauron[should match daatset name]` or `/mountdir/evaluation_results/VOT2019[should match dataset name]`

### Evaluation
The next step is to evaluate and get the metrics of how our model performed. At this time, we are only using VOT metrics like
**Accuracy**, **Precision**, **Lost Number** and **EAO( Expected Average Overlap)**. Read [this](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Kristan_The_Seventh_Visual_Object_Tracking_VOT2019_Challenge_Results_ICCVW_2019_paper.pdf) for their detailed description
Generally, Lost number should be as low as possible, while Accuracy, Precision and EAO should be high

### Test Results Data Format
* All test results are written here `/home/ubuntu/evaluation_results`


    | -- VOT2019
        | -- SiamRPNpp[or your architecture name]
            | -- generic [or your experiment name]
                | -- agility
                    | -- agility.txt
                | -- girl
                    | --girl.txt
                | -- ..
                    | -- ..
* So what is in this *.txt* files?
Let's look at one snippet
```
1
643.3181,515.2716,210.0846,565.4160
647.2288,517.6163,213.7098,568.8172
653.1119,510.3336,215.2389,572.5180
652.6388,510.9387,217.8031,575.4478
661.2344,512.7603,219.1808,575.4930
658.1307,526.9829,219.1412,568.9101
656.1899,539.7589,216.6842,557.0591
655.7529,546.2583,215.3035,545.8845
654.4932,552.3325,211.3256,530.9781
2
1
643.6036,672.2226,195.3636,417.2767
635.6833,678.4359,193.6209,412.7940
641.2213,705.8779,186.5106,399.3938
2
1
```
Here, `1` indicates, we initialize the tracker, the coordinates indicate that we are successfully tracking the object for
that many frames, in above case, 9 frames, with IOU of atleast 0.85 and `2` indicates, we lost the tracker. Then we again initialized it, so on and so forth.   

### Running OT Evaluation scripts
* From `/home/ubuntu/object-tracking` do:
 ```
python pysot/tools/eval.py 
--tracker_path /mountdir/evaluation_results --dataset Sauron 
--tracker_prefix SiamRPNpp 
--groundtruth_path /mountdir/evaluation_data/Sauron-umich
```

The comparison results will look something like this:
```
eval ar: 100%|████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 11.20it/s]
eval eao: 100%|███████████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.07it/s]


    |     Tracker Name     | Accuracy | Robustness | Lost Number |  EAO  |
    | SiamRPNpp_chk_e9_ec2 |  0.919   |   11.506   |    81.0     | 0.079 |
    |  SiamRPNpp_generic   |  0.897   |   12.784   |    90.0     | 0.066 |
```

Here- its safe to assume that `SiamRPNpp_chk_e9_ec2` is performing better than `SiamRPNpp_generic`.

