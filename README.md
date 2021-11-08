# Attention_OCR
CNN+Attention+Seq2Seq

-  The model and its tensor transformation are shown in the figure below

-  It is necessary ch_ train and ch_ test the picture address format of test text to its own file path format

-  There is a missing data picture in the data set originally given in the test set, and there is an empty picture in the picture data set


![](https://user-images.githubusercontent.com/60562159/140689855-ed7dc5ed-a18d-4b05-8259-f8efdf8d3a9e.JPG)


![](https://user-images.githubusercontent.com/60562159/140690037-dcb44c69-7d3c-4c1a-85d7-e35183b6e6ea.PNG)

#### The path in the text is as follows

```
/mnt/disk2/std2021/hejiabang-data/OCR/attention_img/AttentionData/59041171_106970752.jpg 项链付出了十年的苦役
/mnt/disk2/std2021/hejiabang-data/OCR/attention_img/AttentionData/38115031_1485663711.jpg 。直到台“国防部长”
/mnt/disk2/std2021/hejiabang-data/OCR/attention_img/AttentionData/22905328_1196841476.jpg 有惊无险地以21比1
/mnt/disk2/std2021/hejiabang-data/OCR/attention_img/AttentionData/41681796_2460379288.jpg 尼在门前两米处上演“
....
```

#### The training results are as follows

![](https://user-images.githubusercontent.com/60562159/140690357-ccf42e6c-7272-44f1-9ded-5a4d4efb22a6.png)


![](https://user-images.githubusercontent.com/60562159/140690324-3e41be68-627b-439b-9250-3c30ba380a5b.png)

![](https://user-images.githubusercontent.com/60562159/140690379-70a32ca8-1a1b-4eb5-b5c7-51eaf4072bec.png)
