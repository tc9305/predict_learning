# 預測學習作業一

本專案重現風險最小化與模型選擇的四道習題。完整分析見 [中文報告](report.md)；[英文正式報告](Homework1_Report_EN.pdf)保留原始撰寫版本，[作業題目](Homework1.pdf)提供題意。

## 題目與結果速覽

| 題目 | 實驗 | 主要結果 |
| --- | --- | --- |
| 2.7 | 2004 年資料訓練 KNN，2000 年資料測試 | LOOCV 選 $k=7$；測試正確率 49.02% |
| 2.8 | 2000 年資料訓練 KNN，2004 年資料測試 | LOOCV 選 $k=7$；測試正確率 56.86% |
| 2.11 | 比較 cosine 與代數迴歸，以 Schwartz 準則選階 | cosine $m=0$；代數 $m=1$ |
| 2.12 | 五折驗證比較兩種迴歸程序 | 平均 MSE：cosine 0.20392；代數 188.69045 |

選舉分類只使用肥胖率作為特徵；迴歸則使用固定種子產生的十筆資料。報告列出選參流程、混淆矩陣、各折誤差，以及第 2 折代數模型的極端外插結果。

## 安裝與執行

建議使用 Python 3.11。從 `HW1` 資料夾執行：

```powershell
cd HW1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

若已在此工作區根目錄建立 `.venv`，可直接執行：

```powershell
.\.venv\Scripts\python.exe HW1\main.py
.\.venv\Scripts\python.exe -m pytest HW1\tests -q
```

`main.py` 可從任何工作目錄呼叫。若上層資料夾有選舉 CSV，程式會先複製到 `HW1/data/`；否則使用該資料夾內的 CSV。執行後會重建迴歸樣本、結果表格、圖表與 [中文報告](report.md)。

## 檔案導覽

| 位置 | 用途 |
| --- | --- |
| [main.py](main.py) | 執行四個實驗並輸出結果 |
| [src/](src/) | 資料檢查、KNN、迴歸、繪圖及報告發佈程式 |
| [data/](data/) | 兩年選舉資料及產生的迴歸樣本 |
| [results/tables/](results/tables/) | LOOCV、模型選擇、逐筆預測、混淆矩陣與五折結果 |
| [results/figures/](results/figures/) | 15 張實驗圖 |
| [tests/](tests/) | 資料、選參、驗證切分與產物一致性的自動測試 |
| [report.md](report.md) | 可閱讀的中文實驗報告 |

## 重現與閱讀提醒

- KNN 對 $k=1,\ldots,50$ 做訓練年 LOOCV；同分取最小 $k$，然後以全部訓練資料重新擬合。另一年的標籤不參與選參。
- 迴歸的 Schwartz 分數使用課堂指定的乘法型公式。五折驗證在每折的訓練資料內重新選階，避免用完整資料的最佳階數評估留出樣本。
- 2004 年資料將 Hawaii 標為 R；程式保留提供的標籤。跨年測試包含相同州的不同年份觀測，結果不能解讀為因果效應。
- 使用 [requirements.txt](requirements.txt) 固定的套件版本與原始資料順序，可重現圖表與數值。修改演算法或輸入資料時，也應重新檢查 [中文報告](report.md)中的文字敘述。
