# UNET-for-Deblurring (Image Deblurring Using Concatenate UNet)
Practicing code to remove blurring artifacts from images using the modified UNET model.

## 簡介：
>>　　UNet實現影像去模糊化，基於這種像U型的堆疊方式除了能實現去模糊化，也可以做到物件切割，多虧了卷積神經網路保留空間特徵的特性。而此實驗是想延伸UNet的可能不單單只有同一層接再一起，而是把所有特徵(圖1)1、2、3、4層的特徵都同時傳入up sampling層是否可以提升效果，並嘗試加入注意力機制(attention)像是Squeeze-and-Excitation Networks當中的SEblock取得channel attention的資訊期望能更進一步提升準確度，過程中透過計算heat map視覺化每一層關注的區域了解特徵取樣傾向，最後計算FLOPs與參數量分析每一層layer期望未來能改進模型以輕量化為目標。

<div align="center">
    <img width="80%" alt="Concatenate UNET" src="https://imgur.com/78i0ie9.jpg"/><br>
    圖1. Concatenate UNET
</div>

## 資料集：
>>　　此實驗使用GoPro dataset當作去模糊化資料集(圖2)，是由首爾大學公開的資料，透過GoPro4錄製影片每次取7~13 frames的平均當作模糊影像，然後拿最中間的影像當作label，總共有2013 pairs訓練資料，1111 pairs 測試資料。

<div align="center">
    <img width="80%" alt="GoPro dataset" src="https://imgur.com/nTrg44F.jpg"/><br>
    圖2. GoPro dataset
</div>

## 評估指標：
>>　　損失函數使用SSIM loss評估(公式1)，考慮了亮度 (luminance)、對比度 (contrast) 和結構 (structure)指標。同時使用SSIM和PSNR(公式2)兩項去模糊化評估標準。

<table align="center">
  <tr align="center">
    <td>
      <img width="100%" alt="SSIM" src="https://imgur.com/8Bjf3nw.jpg"/>
    </td>
    <td>
      <img width="100%" alt="PSNR" src="https://imgur.com/8tEUtyb.jpg"/>
    </td>
  </tr>
  <tr align="center">
    <td>公式1. SSIM</td>
    <td>公式2. PSNR</td>
  </tr>
</table>

## 實驗結果：
>>　　本次使用Nvidia 1080ti GPU訓練，因為記憶體容量的限制，UNet每層的通道數相比原始論文都會砍半，batch size設4為基準修改模型。
圖3.為準備去模糊化的圖片經評估ssim: 0.9619、psnr: 28.0096，如果單純使用原始UNet架構預測結果為ssim: 0.9723、psnr: 29.1185，確實有達到去模糊的目的，其中以大物件效果最為顯著，如圖5與圖7差異。

<table align="center">
  <tr align="center">
    <td>
      <img width="100%" alt="test image" src="https://imgur.com/9z0QPAk.jpg"/>
    </td>
    <td>
      <img width="100%" alt="label image" src="https://imgur.com/QS4LYmo.jpg"/>
    </td>
  </tr>
  <tr align="center">
    <td>圖3. Test Image</td>
    <td>圖4. Label Image</td>
  </tr>
</table>

>>　　完成基礎架構的訓練之後，將(圖1)1、2、3、4層相接，這邊分成兩部分實驗，分成有無壓縮卷積層通道預測結果。圖8 ssim: 0.9716、psnr: 29.4348。圖9加入channel attention，ssim: 0.9734、psnr: 29.6071。圖10是壓縮通道減少參數量，ssim: 0.9295、psnr: 25.2366。為什麼圖10準度會下降? 因為我額外加入6層layers 為了有效壓縮通道超過GPU memory的限制不得不讓訓練圖像從原本640*480 pixel size縮小到240*240 pixel size，經過雙線性插值法拉升成原本的尺寸得出的結果並不理想，我相信如果在同樣的輸入尺寸訓練效果應該會提升。

<table align="center">
  <tr align="center">
    <td>
      <img width="80%" alt="fig.5" src="https://imgur.com/kniB1Jz.jpg"/>
    </td>
    <td>
      <img width="70%" alt="fig.6" src="https://imgur.com/U9hvCdT.jpg"/>
    </td>
    <td>
      <img width="80%" alt="fig.7" src="https://imgur.com/sIsqtdE.jpg"/>
    </td>
  </tr>
  <tr align="center">
    <td>圖5. Test Image</br>(enlarge)</td>
    <td>圖6. Label Image</br>(enlarge)</td>
    <td>圖7. Output of Original UNET</td>
  </tr>
  <tr align="center">
    <td>
      <img width="80%" alt="fig.8" src="https://imgur.com/ImLhaIj.jpg"/>
    </td>
    <td>
      <img width="70%" alt="fig.9" src="https://imgur.com/6Rbgq7x.jpg"/>
    </td>
    <td>
      <img width="85%" alt="fig.10" src="https://imgur.com/bgJ3tzz.jpg"/>
    </td>
  </tr>
  <tr align="center">
    <td>圖8. Output of Our Concatenate UNET</br>(without reduce channel)</td>
    <td>圖9. Output of Our Concatenate UNET + SElayer</br>(without reduce channel)</td>
    <td>圖10. Output of Concatenate UNET</br>(reduce channel)</td>
  </tr>
</table>

## Attention 視覺化 (Heatmap(GradCAM))
>>　　為了了解每層模型關注的區域，我把它視覺化。使用圖1編號標註每一層的順序，發現在第10層也就是準備輸出結果時幾乎每一點都是重點關注的區域。

<table align="center">
  <tr align="center">
    <td>
      <img width="100%" alt="fig.11" src="https://imgur.com/ky8FDLz.jpg"/>
    </td>
    <td>
      <img width="100%" alt="fig.12" src="https://imgur.com/OZSMqPA.jpg"/>
    </td>
    <td>
      <img width="100%" alt="fig.13" src="https://imgur.com/fdptxp9.jpg"/>
    </td>
    <td>
      <img width="100%" alt="fig.14" src="https://imgur.com/XuOLDdt.jpg"/>
    </td>
    <td>
      <img width="100%" alt="fig.15" src="https://imgur.com/IX6YyoD.jpg"/>
    </td>
  </tr>
  <tr align="center">
    <td>
      <img width="100%" alt="fig.16" src="https://imgur.com/uoVH5ko.jpg"/>
    </td>
    <td>
      <img width="100%" alt="fig.17" src="https://imgur.com/8oRLj36.jpg"/>
    </td>
    <td>
      <img width="100%" alt="fig.18" src="https://imgur.com/8ptzPQY.jpg"/>
    </td>
    <td>
      <img width="100%" alt="fig.19" src="https://imgur.com/rfqTECf.jpg"/>
    </td>
    <td>
      <img width="100%" alt="fig.20" src="https://imgur.com/BNqtUZt.jpg"/>
    </td>
  </tr>
</table>

<div align="center">
    <b>表1.Heatmap(GradCAM)</b><br>(1 represent Output of Layer1 in 圖1)
</div>

## Flops & Params (Concatenate UNet):
- MACs = 7.757008896 G
- FLOPs = 15.514017792 G
- Params = 4.318467 M

<div align="center">
  <img width="50%" alt="each layer's Flops and Params" src="https://imgur.com/0LT9LiC.jpg"/><br>
  圖11. Each Layer's Flops and Params
</div>

>>　　圖11 列出每一層的參數量得知在down4與up1的地方參數量占比最多，未來在輕量化時可以以那兩層為目標改進。

## 結論：
>>　　此次透過深度學習進行影像去模糊化處理真的有很多收穫，包括評估標準不單單只考慮像素對像素的差異，而是把像素周圍的關聯性一併納入計算，包括亮度、對比度和結構，也切身體會到模型獲取有用的特徵越多效果越好的事實。問題是如何處理這些特徵，不單單只是把它串再一起就沒事，我認為此作業的精華就是使用圖12的串接方式處理特徵通道。

<div align="center">
  <img width="80%" alt="Efficient Concatenate Different Channels" src="https://imgur.com/Csu4bD6.jpg"/><br>
  圖12. Efficient Concatenate Different Channels
</div>
