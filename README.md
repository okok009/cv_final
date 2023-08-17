# CV_Final
## 整體架構
整體架構來自於MMLab的RTMDet，在某些內容中做了更改，使得結果能在特定使用環境所需，整個網路是經由Chengqi Lyu 等人所發表的RTMDet 模型進行影像的分割，


能夠辨識多達80 種類別的物件。第一次發表github ，若有著作權問題，願意馬上下架。
## 簡介
現今的Instance Segmentation 的輸出多半複雜難懂，若單純將輸出交給電腦去做應用或許沒有問題，

但為了能讓人類更方便的使用分割出來的輸出，

在Google Meeting 中若使用了背景替換功能，只能將人的畫面保留下來，

但若需要向對方展示手上的物件時，物件將被視為背景而變成草原或書架(視替換背景而異)。

為了解決這些問題，我在這次的作業中加入了以下功能:

1. 在分割後只將辨識出種類的物件的真實樣貌留下，並將背景轉黑，若需要也可替換背景。

2. 由於RTMDet 能夠分辨80 個種類的物件，我讓程式在Demo 的過程中可以隨意切換自己想留下的物件種類，詳細如"Demo 結果"所示。
## Demo 結果
使用一般鏡頭的結果呈現:

![image](https://github.com/okok009/cv_final/blob/master/WebCam_Demo.gif)

搭配Intel 的RealSense D435 的結果:

![image](https://github.com/okok009/cv_final/blob/master/RealSense_Demo.gif)
## 使用方法
需要先將mmdet, mmengine 資料夾與一開始install 在sit-packages 中的檔案取代。

再去mmdet 的github 下載RTMDet 的權重檔，這邊我是用:rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth，再將其放入checkpoints 中。

最後開啟webcam_demo.py 將檔案最上面註解掉的一行在cmd 中執行即可。

## 尚未完成的功能
對於這項結果我還有很多想法，之後有時間做的話再來更新。
