# pic-classify
Use pretrained resnet-152 for transfer learning on multilabel classification task.


# data format

- Your_Image_Folder/
- - img_0001.jpg
- - img_0002.jpg

example of label.csv:

filename,label_1,label_2,label_3,...,label_100\
img_00001.jpg,1,0,0,...,1\
img_00002.jpg,0,1,0,...,0\
...

# environments

Python 3.11.13

numpy 2.0.2

torch 2.6.0+cu124

pandas 2.2.2

sklearn 1.6.1

skmultilearn 0.2.0
