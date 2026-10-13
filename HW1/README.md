# Predictive Learning Homework 1

四題共用一個重現入口。完整英文解答在 [report.md](report.md)，數值與預測明細在 `results/tables/`，15 張圖在 `results/figures/`。

## 執行

需要 Python 3.11 或更新版本。本工作區已建立 `.venv`，從工作區根目錄執行：

```powershell
.\.venv\Scripts\python.exe HW1\main.py
.\.venv\Scripts\python.exe -m pytest HW1\tests -q
```

另一台電腦可攜帶完整 HW1 資料夾後執行：

```powershell
cd HW1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m pytest -q
```

啟用環境後即可 `python main.py`。`hw1.py` 是相同流程的 IDE 入口；請將 IDE interpreter 設為工作區 `.venv/Scripts/python.exe`。所有資料路徑依程式位置解析，不依賴目前工作目錄。

父資料夾有原始選舉 CSV 時，每次複製到 `data/` 後使用；父資料夾無原檔時改讀 `data/`，方便搬移。原始 CSV 和 PDF 不變。重跑會重建回歸樣本、结果和報告；個人修改的報告請另存。

## 程式與輸出

| 檔案 | 用途 |
| --- | --- |
| `src/data_loader.py` | 無 header CSV 驗證、描述統計、固定種子資料生成 |
| `src/knn_experiment.py` | KNN、完整 k 搜尋、實際 LOOCV、最終測試、圖表 |
| `src/problem_2_7.py` / `problem_2_8.py` | 指定兩種訓練／測試年份方向 |
| `src/polynomial_models.py` | 精確 cosine / monomial basis、SVD 最小平方法、合法階数搜尋 |
| `src/schwartz_criterion.py` | 課堂 Schwartz 公式與 DoF<n 檢查 |
| `src/problem_2_11.py` | 完整十筆資料選模與預測曲線 |
| `src/problem_2_12.py` | 共用 folds、fold 內重新選階、OOF 評估 |
| `src/reporting.py` | 根據本次實驗生成英文報告 |
| `tests/test_homework1.py` | 數學、資料隔離、流程、重現與報告一致性測試 |

選舉：摘要、比較、每個 k 的 LOOCV 分數、每次 LOO 預測、訓練及測試預測、錯誤州、confusion matrix。每題四張圖：LOO 曲線、訓練分布、測試真實／預測、confusion matrix。

回歸：新生成十筆資料、兩種完整選模表、最佳係數及摘要、每折 train/validation 索引、全部 fold 選模表、OOF 預測、CV 摘要。2.11 四張圖：兩種最佳曲線、criterion 比較、訓練 MSE 比較；2.12 三張圖：各折 MSE、平均 MSE、OOF versus observed y。

## 方法與教材頁碼

- **2.7 / 2.8**：分別以 2004／2000 訓練，在各自訓練年搜尋 k=1…50。KNN uniform vote 與 Euclidean distance：scikit-learn 教材第 20–22 頁；0–1 loss：Lecture Set 2 第 19 頁；LOOCV：Lecture Set 3 第 42 頁。
- **2.11**：精確 cosine-only 與標準代數 basis，以 OLS 擬合。基底展開：scikit-learn 教材第 10 頁、Lecture Set 3 第 12–13 頁；精確基底以 Homework1.pdf 第 2 頁為準。
- **2.12**：每折八筆訓練各自重新選階，再預測兩筆保留資料。方法評估與選模分離：Lecture Set 3 第 36–43、47–48 頁。
- 前處理：Lecture Set 1 第 39–40 頁；bias–variance：Lecture Set 3 第 25–32 頁。

## 數值意義

- **Optimal k**：LOOCV 錯誤最小的鄰居數；越小越局部。分數同分取最小 k，類別票數平手取 D。等距鄰居使用固定版本 sklearn brute backend 與原始順序；更換版本或順序可能影響結果。
- **LOOCV Error**：每次保留一筆，以另 50 筆訓練，共 51 次；錯誤數／51。這是選模分數，不是跨年度測試誤差。所有 error 欄位都是比例。
- **Optimal m / DoF**：m 是代數最高次方或 cosine 最大頻率指標；兩者参数數為 m+1。完整資料搜尋 m=0…8，每折搜尋 m=0…6。
- **Schwartz Criterion**：`R_emp * (1+p*ln(n)/(1-p))`，`p=DoF/n`，`R_emp` 是訓練 MSE；以增加的複雜度代價平衡擬合。採 Lecture Set 3 第 34 頁公式，不是常見 BIC。較小表示此準則偏好的模型，不保證真實預測誤差最小。
- **5-Fold CV MSE**：比較整個「擬合與選階」流程的保留資料平方誤差。五折等大，所以平均 fold MSE 等於整體 OOF MSE。fold 標準差用 ddof=1；不是信賴區間。

## 請自行確認

回歸樣本是依題目方式生成的新十筆，不是課本原始數值；請確認老師接受這項假設。2004 CSV 中 Hawaii 是 R，程式保留原標籤，請向老師核對。Lecture Set 2 的例子沒有 0.1x，實作依作業加入此項。

閱讀錯誤州、選模表和 fold 明細，理解等距及票數平手規則。十筆資料 CV 可能高度不穩定，不能推論某模型普遍最佳。檢查 numerical rank、condition number 及 >1e8 診斷標記，未以額外正則化隱藏不穩定性。

請理解公式、圖表與限制，並根據自己的分析改寫討論後依課堂規範提交。報告是可重現的學習與分析輔助。
