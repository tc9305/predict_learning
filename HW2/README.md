# 預測學習作業二：VC-theory and its Applications

本作業包含兩題：以三角多項式迴歸比較模型選擇方法（Problem 1），以及用分類器擇時交易國際共同基金 FIGRX（Problem 2）。完整分析見 [英文報告](report/main.pdf)，題目見 [HW2.pdf](Instructions/HW2.pdf)。

## 題目與結果速覽

### Problem 1：模型選擇（4 分）

依課本比較模型選擇方法的實驗流程，比較課本使用的四種方法：fpe、gcv、VC 懲罰與留一法交叉驗證（cv），記錄它們選出的 DoF 與預測風險。模型為課程講義定義的三角多項式：

$$f_m(x) = w_0 + \sum_{i=1}^{m} \bigl[v_i \sin(ix) + w_i \cos(ix)\bigr], \qquad \text{DoF} = 2m + 1$$

資料為純雜訊（最佳 DoF = 1）與 sine-squared，n = 30 與 100 各重複 300 次。在 $x \in [0, 1]$ 上，這組基底無法精確表示 $\sin^2(2\pi x)$，所以 sine-squared 沒有單一正確 DoF，改比較 DoF 分布與風險。

| 預測風險第 95 百分位 | fpe | gcv | vc | cv |
| --- | --- | --- | --- | --- |
| 純雜訊，n = 30 | 169.1 | 2.82 | **0.144** | 0.492 |
| sine-squared，n = 30 | 413.3 | 3.41 | **0.232** | 0.390 |
| 純雜訊，n = 100 | 0.155 | 0.130 | **0.0381** | 0.152 |
| sine-squared，n = 100 | 0.0192 | 0.0179 | **0.0108** | 0.0195 |

VC 懲罰在四種設定的第 95 百分位風險都最低。純雜訊時，vc 選中常數模型的比例為 99.0%（n = 30）與 100%（n = 100），fpe 只有 61.3% 與 78.3%。

### Problem 2：FIGRX 擇時交易（6 分）

輸入為當日 S&P 500 與歐元匯率的漲跌（%），輸出為 FIGRX 隔日漲跌。2004 年資料訓練（243 筆），2005 年資料測試（244 筆）。

| 策略 | 訓練準確率 | 測試準確率 | 2005 GAIN | EXPOSURE |
| --- | --- | --- | --- | --- |
| 線性最小平方分類器 | 51.7% | 57.1% | 21.81% | 76.6% |
| 二次決策邊界分類器 | 57.2% | 55.8% | 15.19% | 82.0% |
| Buy-and-Hold | — | — | 14.92% | 100.0% |

兩個分類器的準確率都沒有明顯高於「永遠猜漲」（2005 年 55.0%）。線性策略的報酬較高，但只有一年測試資料；隨機挑選相同天數持有的策略中，有 3.1% 能達到同樣報酬；若每次切換扣 0.1% 成本，報酬降為 11.43%，低於 Buy-and-Hold。依分類的 VC 界（Eq. 4.11，信心水準約 74%），兩個模型的 n/h 為 78.7 與 39.3，遠高於課本小樣本的門檻 15，支持只擬合固定的低容量模型；但測試錯誤率上限為 0.781 與 0.813，高於隨機猜測的 0.5。

## 執行方式

在儲存庫根目錄建立虛擬環境並安裝 [requirements.txt](requirements.txt) 的套件（步驟見[根目錄 README](../README.md)），再從 `HW2` 資料夾執行：

```powershell
cd HW2
..\.venv\Scripts\python.exe practice\p1_mine.py
..\.venv\Scripts\python.exe -m practice.p2_mine
..\.venv\Scripts\python.exe -m practice.p1_checks
..\.venv\Scripts\python.exe -m practice.p2_checks
```

- `practice\p1_mine.py`：跑 Problem 1 的 300 次實驗，輸出選對比例與風險，並產生 `p1_n30.png`、`p1_n100.png`（約 30 秒）。
- `practice.p2_mine`：印出兩個分類器的係數、準確率、交易結果、隨機策略比較與分類的 VC 界，並產生 `p2_boundaries.png`。這個檔案需用 `-m` 以模組方式執行，才能載入 `src/data_loader.py`。
- `practice.p1_checks`：Problem 1 的數值驗證（設計矩陣的秩、留一法公式與逐次重擬合、逐次比對 cv 的選擇、三種 LAPACK 求解器），以及報告引用的其他數字。輸出存成 `p1_checks_output.txt`（約 2 分鐘）。
- `practice.p2_checks`：Problem 2 報告引用、但 `p2_mine` 沒有印出的數字（對齊天數、橢圓判定、交易成本、抽樣誤差等）。輸出存成 `p2_checks_output.txt`。

報告附錄列出各項驗證的設定（亂數種子、秩的判定門檻、求解器名稱、套件版本）。

報告以 XeLaTeX 編譯，需編譯兩次以產生圖表編號：

```powershell
cd report
xelatex main.tex
xelatex main.tex
```

## 資料處理

- 價格資料：S&P 500（`^GSPC`）與 FIGRX 取自 Yahoo Finance，歐元匯率（每 1 美元兌換的歐元）取自 UBC Pacific Exchange Rate Service。
- 依 [前處理說明](Instructions/Data%20Preprocessing%20for%20Homework%202.pdf)，只保留三份資料都有的日期：2004 年 243 天、2005 年 245 天。
- 每日漲跌以前一個對齊交易日為基準。程式把各年價格接成連續序列再計算，所以 2004 年最後一天的隔日漲跌取自 2005 年第一個交易日。
- 2004 年第一個交易日（1 月 6 日）需要 2003 年 12 月 31 日的價格，存放在 [data/extra_2003-12-31.csv](data/extra_2003-12-31.csv)。
- FIGRX 漲跌為 0 的 7 天：訓練二次分類器與計算準確率時排除；交易模擬保留。

## 檔案導覽

| 位置 | 用途 |
| --- | --- |
| [HW2.pdf](Instructions/HW2.pdf) | 作業題目 |
| [section2_4.pdf](Instructions/section2_4.pdf)、[section4_4.pdf](Instructions/section4_4.pdf) | 課本章節：懲罰因子、VC 界與實驗流程 |
| [Data Preprocessing for Homework 2.pdf](Instructions/Data%20Preprocessing%20for%20Homework%202.pdf) | 資料前處理說明 |
| [data/](data/) | 2004–2005 年三份價格資料與 2003/12/31 補充資料 |
| [src/data_loader.py](src/data_loader.py) | 讀取並依日期對齊三份價格資料 |
| [practice/p1_mine.py](practice/p1_mine.py) | Problem 1 程式 |
| [practice/p2_mine.py](practice/p2_mine.py) | Problem 2 程式 |
| [practice/p1_checks.py](practice/p1_checks.py)、[p1_checks_output.txt](practice/p1_checks_output.txt) | Problem 1 數值驗證與輸出紀錄 |
| [practice/p2_checks.py](practice/p2_checks.py)、[p2_checks_output.txt](practice/p2_checks_output.txt) | Problem 2 補充數字與輸出紀錄 |
| [practice/](practice/) | 報告使用的三張圖 |
| [report/](report/) | 報告原始檔（`main.tex`、`problem1.tex`、`problem2.tex`、`appendix.tex`）與 [main.pdf](report/main.pdf) |
| [requirements.txt](requirements.txt) | 固定版本的 Python 套件 |
