# Evaluator

Move to this directory
```
cd image_generation_t/evaluator
```



# Running the Code
To evaluate FID Score on the FFHQ dataset, run the following command:
```
python3 FFHQ_evaluate.py -s "path/to/FFHQ_Dataset" -g "path/to/generated_image_dataset"
```


To evaluate FID Score & Inception Score in ImageNet dataset, run the following command:
```
python3 ImageNet_evaluate.py -s "path/to/ImageNet_Dataset" -g "path/to/generated_image_dataset"
```
(Both the ImageNet dataset and the generated image dataset directories must contain 1,000 subfolders, each corresponding to one of the 1,000 classes.)