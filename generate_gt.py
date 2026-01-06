import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat

# ================= 參數設定 =================
# 1. 原始水位數據路徑
MAT_FILE = "iHARP-ML-Challenge-2/NEUSTG_19502020_12stations.mat"

# 2. 官方閾值檔案路徑 (這是修正重點，必須使用此檔案定義洪水)
THRESH_MAT_FILE = "iHARP-ML-Challenge-2/Seed_Coastal_Stations_Thresholds.mat"

# 3. 測試索引檔案 (定義了哪些時段需要預測)
TEST_INDEX_FILE = "data/test_index.csv"

# 4. 輸出檔案路徑
OUTPUT_FILE = "output/ground_truth.csv"
# ===========================================

def load_mat_thresholds(mat_path):
    """
    從 .mat 檔案讀取官方定義的測站閾值。
    """
    print(f"正在讀取閾值檔案: {mat_path}")
    if not Path(mat_path).exists():
        raise FileNotFoundError(f"找不到閾值檔案: {mat_path}")

    try:
        mat = loadmat(mat_path)
    except Exception as e:
        raise RuntimeError(f"無法讀取 .mat 檔案: {e}")

    # 過濾系統變數
    valid_keys = [k for k in mat.keys() if not k.startswith('__')]
    
    # 自動偵測變數名稱
    # 通常是 'sname' (測站名) 和 'thminor_stnd' (小洪水閾值) 或類似名稱
    key_stn = next((k for k in valid_keys if 'station' in k.lower() or 'name' in k.lower() or 'sname' in k.lower()), None)
    key_thr = next((k for k in valid_keys if 'threshold' in k.lower() or 'value' in k.lower() or 'thminor' in k.lower()), None)
    
    if not key_stn: key_stn = valid_keys[0]
    if not key_thr: key_thr = valid_keys[1] if valid_keys[1] != key_stn else valid_keys[0]

    print(f"使用變數 '{key_stn}' 作為測站名稱, '{key_thr}' 作為閾值。")

    raw_stations = mat[key_stn].reshape(-1)
    raw_thresholds = mat[key_thr].reshape(-1)

    clean_stations = []
    for s in raw_stations:
        if isinstance(s, np.ndarray):
            s = s.item() if s.size > 0 else ""
        if isinstance(s, (bytes, np.bytes_)):
            s = s.decode('utf-8', errors='ignore')
        clean_stations.append(str(s).strip())

    df_thr = pd.DataFrame({
        "station_name": clean_stations,
        "flood_threshold": raw_thresholds
    })
    return df_thr

def generate_ground_truth():
    # --- 1. 讀取與處理水位資料 ---
    if not Path(MAT_FILE).exists():
        raise FileNotFoundError(f"找不到 {MAT_FILE}")
    
    print(f"讀取水位資料: {MAT_FILE} ...")
    d = loadmat(MAT_FILE)
    sea_level = d['sltg']
    station_names = [s[0].strip() for s in d['sname'].flatten()]
    time = d['t'].flatten()
    
    # 時間轉換 (MATLAB datenum 719529 = 1970-01-01)
    time_dt = pd.to_datetime(time - 719529, unit='D')
    
    T, S = sea_level.shape
    
    # 建立 Hourly DataFrame
    df_hourly = pd.DataFrame({
        "time": np.tile(time_dt, S),
        "station_name": np.repeat(station_names, T),
        "sea_level": sea_level.reshape(-1, order="F")
    })
    
    # 轉為 Daily Max (只需 Max 即可判斷是否淹水)
    print("聚合每日最高水位...")
    df_hourly["date"] = df_hourly["time"].dt.floor("D")
    daily = (df_hourly.groupby(["station_name", "date"])
             .agg(sea_level_max=("sea_level", "max"))
             .reset_index())

    # --- 2. 讀取並合併官方閾值 ---
    df_thresholds = load_mat_thresholds(THRESH_MAT_FILE)
    
    daily = daily.merge(df_thresholds, on="station_name", how="left")
    
    # 檢查是否有測站缺少閾值
    if daily["flood_threshold"].isna().any():
        missing = daily[daily["flood_threshold"].isna()]["station_name"].unique()
        print(f"警告: 測站 {missing} 缺少閾值定義，將默認為無洪水 (0)。")
    
    # --- 3. 標記每日洪水狀態 (Ground Truth Logic) ---
    # 定義: 當日最高水位 > 官方閾值 = 1 (淹水)
    daily["is_flood_day"] = (daily["sea_level_max"] > daily["flood_threshold"]).fillna(0).astype(int)

    # 加速查詢：設定索引
    daily_indexed = daily.set_index(["station_name", "date"]).sort_index()

    # --- 4. 讀取 Test Index 並生成答案 ---
    if not Path(TEST_INDEX_FILE).exists():
        raise FileNotFoundError(f"找不到 {TEST_INDEX_FILE}")
    
    print(f"讀取 {TEST_INDEX_FILE} 並生成標籤...")
    test_index = pd.read_csv(TEST_INDEX_FILE)
    
    test_index["future_start"] = pd.to_datetime(test_index["future_start"])
    test_index["future_end"] = pd.to_datetime(test_index["future_end"])
    
    ground_truth_rows = []
    total_rows = len(test_index)
    
    for idx, row in test_index.iterrows():
        stn = row["station_name"]
        f_start = row["future_start"]
        f_end = row["future_end"]
        # 使用 csv 中的 id，若無則用 index
        row_id = row["id"] if "id" in row else idx
        
        try:
            # 查詢該測站、該預測區間 [future_start, future_end] 的資料
            subset = daily_indexed.loc[(stn, f_start):(stn, f_end)]
            
            # 邏輯: 只要區間內有一天淹水 (is_flood_day=1)，則該樣本標籤為 1
            is_flood = subset["is_flood_day"].max()
            
            if np.isnan(is_flood):
                is_flood = 0
                
        except KeyError:
            # 若無資料
            is_flood = 0
            
        ground_truth_rows.append({
            "id": row_id,
            "y_true": int(is_flood)  # scoring.py 預設讀取 "y_true"
        })
        
        if (idx + 1) % 20000 == 0:
            print(f"已處理 {idx + 1}/{total_rows}...")

    # --- 5. 輸出結果 ---
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    gt_df = pd.DataFrame(ground_truth_rows)
    gt_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Ground Truth 生成完畢: {OUTPUT_FILE}")
    print(f"資料筆數: {len(gt_df)}")
    print(f"正樣本 (淹水) 比例: {gt_df['y_true'].mean():.2%}")
    print(gt_df.head())

if __name__ == "__main__":
    generate_ground_truth()