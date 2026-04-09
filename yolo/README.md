best.pt is the best performed yolo model we've trained.

Running `yolo.py` you can visualize the performance of the model

Running `yolo laser match.py` you can visualize how we assigned laser points with different meanings.


All scripts requires `ultralytics==8.4.34` and `rosbags`. No ros environment are required.
 
Training info and data sets are listed here:
https://drive.google.com/drive/folders/1-zE6DV8pEdiYcfC7sHqsTIyIuEYDFliM?usp=sharing

`extract pics.py` is the script we used to extract pictures from the ros bag to train the model.

We used `label studio` to label the pictures. We initially manually annotated 30 images for a feasibility study. Subsequently, we integrated the trained model into Label Studio for pre-labeling, followed by manual fine-tuning.