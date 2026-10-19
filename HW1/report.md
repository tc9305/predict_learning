# 預測學習作業一：風險最小化與模型選擇

本報告依據 [英文正式報告](Homework1_Report_EN.pdf) 整理題目 2.7、2.8、2.11 與 2.12 的實驗設定、主要數值及解釋。

## 摘要

| 題目 | 方法 | 主要結果 |
| --- | --- | --- |
| 2.7 | 2004 年訓練、2000 年測試；KNN + LOOCV | $k=7$；測試正確率 49.02% |
| 2.8 | 2000 年訓練、2004 年測試；KNN + LOOCV | $k=7$；測試正確率 56.86% |
| 2.11 | Schwartz 準則選迴歸模型階數 | cosine $m=0$；代數 $m=1$ |
| 2.12 | 五折交叉驗證比較預測誤差 | 平均 MSE：cosine 0.20392；代數 188.69045 |

## 資料與實驗設定

選舉資料各有 51 筆，涵蓋 50 州與華盛頓特區。唯一輸入特徵是肥胖率（百分比數值），標籤是資料提供的 D 或 R。兩年的州代碼相同，並保留原始列順序與標籤，包括 2004 年 Hawaii 標為 R。本文評估的是提供的資料標籤。相同州在兩年都出現，因此跨年測試並非對新州的獨立抽樣。

| 年份 | 筆數 | D | R | 平均肥胖率 |
| --- | ---: | ---: | ---: | ---: |
| 2000 | 51 | 21 | 30 | 19.10% |
| 2004 | 51 | 19 | 32 | 22.39% |

迴歸資料是另行產生的 10 筆樣本：使用 `np.random.default_rng(42)` 先取 $x\sim U(0,1)$，再取獨立的 $\epsilon\sim N(0,0.5^2)$，令 $y=x^2+0.1x+\epsilon$。兩種模型使用同一組樣本。原始教科書範例的十筆觀測未提供，因此不宣稱重現其個別數值。[產生的資料](data/generated_example_2_6.csv)包含生成值及參考函數；模型估計與選擇只使用 $x,y$。

## 題目 2.7 與 2.8：跨年 KNN 分類

### 方法與選參

以單一特徵計算歐氏距離 $|x-x_i|$，最近的 $k$ 筆以等權重投票。對每個 $k=1,\ldots,50$，只在**訓練年份**做留一交叉驗證（LOOCV）：每次用 50 筆訓練並預測剩下的一筆，共取得 51 個留出預測。以 0–1 損失計算錯誤率：

$$
\widehat R_{\mathrm{LOO}}(k)=\frac{1}{51}\sum_{i=1}^{51}\mathbf 1\{\widehat y_{-i,k}(x_i)\ne y_i\}.
$$

選擇 LOOCV 錯誤率最低的 $k$；若數個 $k$ 同分，取最小值。接著用該年全部 51 筆資料重新訓練，再預測另一年。測試年份不參與選參。實作用 scikit-learn 的 brute-force、Euclidean、uniform vote 與單執行緒設定。D/R 得票相同時，分類器按排序後的類別順序選 D；第 $k$ 與第 $k+1$ 個鄰居等距時，選入哪一筆可能受資料順序與套件版本影響。

### 題目 2.7：2004 → 2000

最佳 $k$ 同分候選為 7、20、22，依規則選 **$k=7$**。LOOCV 錯誤率 **14/51 = 27.45%**；訓練錯誤率 **12/51 = 23.53%**。跨年測試錯誤率 **0.50980392（26/51 = 50.98%）**，正確率 **49.02%**。

| 實際標籤 | 預測 D | 預測 R |
| --- | ---: | ---: |
| D | 17 | 4 |
| R | 22 | 8 |

混淆矩陣的列為實際 D/R、欄為預測 D/R。22 筆實際 R 被判為 D，是本題錯誤的主要來源。

![題目 2.7 的 LOOCV 錯誤率](results/figures/problem_2_7_loocv.png)

[測試預測逐筆資料](results/tables/problem_2_7_test_predictions.csv) · [分類錯誤州清單](results/tables/problem_2_7_misclassified_states.csv) · [完整 LOOCV 分數](results/tables/problem_2_7_loocv.csv)

### 題目 2.8：2000 → 2004

交換訓練與測試年份後，重新只在 2000 年資料選 $k$。同分候選為 7、9、16、24、25、26、27、28，依規則仍選 **$k=7$**。LOOCV 錯誤率 **19/51 = 37.25%**；訓練錯誤率 **14/51 = 27.45%**。跨年測試錯誤率 **0.43137255（22/51 = 43.14%）**，正確率 **56.86%**。

| 實際標籤 | 預測 D | 預測 R |
| --- | ---: | ---: |
| D | 2 | 17 |
| R | 5 | 27 |

![題目 2.8 的 LOOCV 錯誤率](results/figures/problem_2_8_loocv.png)

[測試預測逐筆資料](results/tables/problem_2_8_test_predictions.csv) · [分類錯誤州清單](results/tables/problem_2_8_misclassified_states.csv) · [完整 LOOCV 分數](results/tables/problem_2_8_loocv.csv)

### 跨年結果解讀

題目 2.7 的跨年測試錯誤率比 LOOCV 高 23.53 個百分點。LOOCV 評估的是 2004 年樣本內留出的觀測，也被用來選 $k$；它不能保證轉移至 2000 年仍有相同表現。作為參考，按 2004 年訓練集的多數標籤 R 固定預測，對 2000 年資料可達 30/51 = 58.82% 正確率，高於 KNN 的 49.02%。這項基準僅用於事後解讀，不參與選 $k$。

兩年的肥胖率分布、標籤比例與鄰居組成不同，可能造成方向不對稱。只用肥胖率無法充分區分資料中的 D/R；此統計關係也不能推論肥胖率造成選舉結果。[跨年比較表](results/tables/knn_comparison.csv)保留完整數值。

## 題目 2.11：迴歸模型與 Schwartz 選階

### 模型與準則

cosine 類只含餘弦項，**沒有 sine 項**；代數類使用冪次基底：

$$
f_m^{\mathrm{cos}}(x)=w_0+\sum_{j=1}^{m}w_j\cos(2\pi jx),\qquad
f_m^{\mathrm{alg}}(x)=\sum_{j=0}^{m}w_jx^j.
$$

兩類模型都有 $m+1$ 個係數。使用 `numpy.linalg.lstsq` 求最小平方解，不加入 ridge 或 lasso。對完整的 $n=10$ 筆資料，以自由度 $m+1<n$ 限制候選階數為 $m=0,\ldots,8$。令訓練 MSE 為 $R_{\mathrm{emp}}$、$p=(m+1)/n$，依課堂的乘法型 Schwartz 準則選最小分數：

$$
R_{\mathrm{SC}}=\left(1+\frac{p\ln n}{1-p}\right)R_{\mathrm{emp}}.
$$

完全同分時取較小的 $m$。此公式是本課使用的形式，與常見的對數型 BIC 不同。

### 全資料結果

| 模型 | 最佳 $m$ | 自由度 | 訓練 MSE | Schwartz 分數 |
| --- | ---: | ---: | ---: | ---: |
| cosine | 0 | 1 | 0.10764 | 0.13518 |
| 代數 | 1 | 2 | 0.08084 | 0.12738 |

選中的模型為 $\widehat f^{\mathrm{cos}}(x)=0.63373$ 與 $\widehat f^{\mathrm{alg}}(x)=0.29587+0.56646x$。代數 $m=2$ 雖略微降低訓練 MSE（0.07822 對 0.08084），增加的複雜度使 Schwartz 分數升至 0.15541，因此選出 $m=1$。cosine 模型滿足 $f(x)=f(1-x)$，而生成函數並不具備此對稱性。

全資料的 Schwartz 分數偏向代數類，但分數仍是由擬合樣本算出的選模指標，不能單獨證明其樣本外預測較佳。

![兩類模型的 Schwartz 分數](results/figures/problem_2_11_criterion.png)

![選出的代數模型](results/figures/problem_2_11_algebraic_fit.png)

![選出的 cosine 模型](results/figures/problem_2_11_trigonometric_fit.png)

[代數候選模型分數](results/tables/problem_2_11_algebraic_selection.csv) · [cosine 候選模型分數](results/tables/problem_2_11_trigonometric_selection.csv)

## 題目 2.12：五折驗證比較預測誤差

沿用題目 2.11 的同一組 10 筆資料，使用 `KFold(n_splits=5, shuffle=True, random_state=42)` 產生兩類模型共用的五折。每折 8 筆訓練、2 筆驗證。**每一折、每一類模型各自**在 8 筆訓練資料上用 Schwartz 準則選 $m=0,\ldots,6$，再計算保留兩筆的驗證 MSE；題目 2.11 在全部十筆上選出的階數不直接套用。每筆樣本恰有一次樣本外預測；因各折驗證筆數相同，平均折 MSE 等於十筆 out-of-fold 預測的整體 MSE。

### 各折結果

| 折 | cosine $m$ | cosine MSE | 代數 $m$ | 代數 MSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 0.01793 | 1 | 0.02860 |
| 2 | 0 | 0.14729 | 6 | 942.63846 |
| 3 | 5 | 0.40218 | 3 | 0.33297 |
| 4 | 0 | 0.24554 | 0 | 0.24554 |
| 5 | 0 | 0.20666 | 0 | 0.20666 |

| 模型 | 平均驗證 MSE | 各折 MSE 樣本標準差 |
| --- | ---: | ---: |
| cosine | 0.20392 | 0.14040 |
| 代數 | 188.69045 | 421.46977 |

此固定切分下，cosine 的平均驗證 MSE 較低。代數模型在第 **2 折**僅以 8 筆訓練資料選到 $m=6$，該折 MSE 達 **942.63846**，主導整體平均。其中一筆驗證輸入 $x=0.97562$，觀測 $y=0.61975$，預測卻為 **44.03538**。此輸入超出該折訓練輸入約 0.09418 至 0.85860 的範圍，顯示高階代數模型的外插風險。設計矩陣條件數約 $5.31\times10^5$；數值診斷及 SVD 解法也無法保證預測穩定。圖表保留這筆極端誤差，沒有截斷或重新調參。

![各折驗證 MSE](results/figures/problem_2_12_folds.png)

![樣本外預測與實際值](results/figures/problem_2_12_oof.png)

[五折明細](results/tables/problem_2_12_folds.csv) · [每筆樣本外預測](results/tables/problem_2_12_oof_predictions.csv) · [各折選階分數](results/tables/problem_2_12_fold_selection.csv)

## 結論與限制

題目 2.11 的全資料 Schwartz 分數偏向代數模型；題目 2.12 的五折預測誤差偏向 cosine 程序。兩者評估的量不同，因此不能只憑全資料選模分數斷言哪類模型在未知資料上一定較好。十筆含噪資料、每折僅兩筆驗證值及一次固定切分，使結果對折分配與極端預測敏感。表中的標準差描述五個折分數的分散程度，並非信賴區間。KNN 結果也只描述提供資料的跨年表現，不能推論因果關係。

## 重現方式

在 `HW1` 目錄安裝 [requirements.txt](requirements.txt) 後執行 `python main.py`。程式重新產生本報告、[表格](results/tables/)及[圖表](results/figures/)；[README.md](README.md)列有完整指令與檔案導覽。固定亂數種子、資料順序與套件版本有助重現。
