# 預測學習（Predictive Learning）

國立臺灣師範大學「預測學習」課程的作業與專題，教材為 V. Cherkassky, *Predictive Learning*（2013）。

| 項目 | 主題 | 內容 |
|---|---|---|
| [HW1](HW1/) | Risk Minimization and Model Selection | 跨年 KNN 分類、迴歸模型選擇、五折交叉驗證 |
| [HW2](HW2/) | VC-theory and its Applications | 三角多項式迴歸的模型選擇比較；FIGRX 基金擇時交易分類器 |
| [midterm_project](midterm_project/) | 首爾共享單車租借量預測 | 期中研究規劃與資料 |

## 專案架構

```text
.
├─ README.md
├─ .gitignore
├─ course/                      # 課程大綱與講義 PDF
├─ HW1/
│  ├─ README.md                # 作業說明與結果速覽
│  ├─ Homework1.pdf            # 作業題目
│  ├─ Homework1_Report_EN.pdf  # 英文報告
│  ├─ report.md                # 中文報告
│  ├─ main.py                  # 四題實驗的執行入口
│  ├─ requirements.txt
│  ├─ data/  src/  results/  tests/
├─ HW2/
│  ├─ README.md                # 作業說明與結果速覽
│  ├─ HW2.pdf                  # 作業題目
│  ├─ section2_4.pdf, section4_4.pdf, Data Preprocessing for Homework 2.pdf
│  ├─ requirements.txt
│  ├─ data/                    # GSPC、FIGRX、EURUSD 2004–2005 收盤價，及 2003/12/31 補充資料
│  ├─ src/data_loader.py       # 讀取並對齊三份價格資料
│  ├─ practice/
│  │  ├─ p1_mine.py            # Problem 1：模型選擇實驗與箱型圖
│  │  ├─ p2_mine.py            # Problem 2：分類器、決策邊界圖、交易模擬
│  │  └─ *.png                 # 報告使用的圖
│  └─ report/
│     ├─ main.tex              # 報告主檔（XeLaTeX）
│     ├─ problem1.tex, problem2.tex
│     └─ main.pdf              # 英文報告
└─ midterm_project/
   ├─ README.md                # 研究規劃
   └─ csv_data/
```

## HW1：Risk Minimization and Model Selection

[作業說明](HW1/README.md)介紹四題的重點，[中文報告](HW1/report.md)記錄實驗方法、數值與限制，[英文報告](HW1/Homework1_Report_EN.pdf)為正式版本。

## HW2：VC-theory and its Applications

[作業說明](HW2/README.md)整理兩題的結果與資料處理方式，[英文報告](HW2/report/main.pdf)為正式版本。

- **Problem 1：模型選擇。**依課本 Section 4.4 的實驗流程，以三角多項式迴歸比較 fpe、sc、gcv、VC 懲罰與留一法交叉驗證。資料為純雜訊與 sine-squared 兩組，樣本數 n = 30 與 100，各重複 300 次，以仿 Fig. 4.10–4.11 的箱型圖呈現選出的 DoF 與預測風險。
- **Problem 2：國際基金擇時交易。**以當日 S&P 500 與歐元匯率的漲跌預測 FIGRX 隔日漲跌。用 2004 年資料訓練線性最小平方與二次決策邊界兩種分類器，在 2005 年模擬交易，並與 Buy-and-Hold 比較 GAIN 與 EXPOSURE，最後以 VC 理論討論結果。

## 執行方式

建議使用 **Python 3.11**，在儲存庫根目錄建立虛擬環境。以下為 Windows PowerShell 指令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r HW1\requirements.txt
```

已安裝 [uv](https://docs.astral.sh/uv/) 時，也可以改用：

```powershell
uv venv --python 3.11 .venv
uv pip install --python .\.venv\Scripts\python.exe -r HW1\requirements.txt
```

HW1 與 HW2 的套件版本相同，裝一次即可。macOS 或 Linux 的執行檔路徑改為 `.venv/bin/python`。

**HW1**（在儲存庫根目錄）：

```powershell
.\.venv\Scripts\python.exe HW1\main.py
.\.venv\Scripts\python.exe -m pytest HW1\tests -q
```

`HW1/main.py` 會重建[結果表格](HW1/results/tables/)、[圖表](HW1/results/figures/)與[中文報告](HW1/report.md)。

**HW2**（在 `HW2` 資料夾內）：

```powershell
cd HW2
..\.venv\Scripts\python.exe practice\p1_mine.py
..\.venv\Scripts\python.exe -m practice.p2_mine
```

第一個指令產生 Problem 1 的兩張箱型圖；第二個指令印出分類器係數、準確率與交易結果，並產生決策邊界圖。圖檔存放在 `HW2/practice/`。

**HW2 報告**以 XeLaTeX 編譯（例如 MiKTeX 或 TeX Live）：

```powershell
cd HW2\report
xelatex main.tex
xelatex main.tex
```

編譯兩次才能正確產生圖表編號。

## 注意事項

- [course/](course/) 存放課程大綱與講義；各作業的題目 PDF 放在各自的資料夾。
- 原始資料與實驗輸出保留在版本庫，方便直接閱讀與比對結果。
- HW2 的 `HW2/data/extra_2003-12-31.csv` 是 2003 年最後一個交易日的三個收盤價（Yahoo Finance、UBC Pacific Exchange Rate Service），用來計算 2004 年第一筆樣本的漲跌。
- `.venv/`、`__pycache__/`、`.pytest_cache/` 與 LaTeX 編譯暫存檔由 [.gitignore](.gitignore) 排除。
- 套件版本或執行環境不同時，浮點數輸出可能略有差異；報告數值對應目前的資料與設定。
