import os
import json
import pandas as pd
from datetime import datetime, timedelta

# Centralized ingestion folder
STORAGE_DIR = os.path.join("data", "raw")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Metric Configuration
METRIC_CONFIG = {
    "resting_heart_rate": {"label": "Resting HR", "unit": "BPM", "color": "#f59e0b"},
    "sleep_analysis": {"label": "Sleep Duration", "unit": "Hrs", "color": "#00f0ff"},
    "heart_rate": {"label": "Avg Heart Rate", "unit": "BPM", "color": "#ef4444"},
    "step_count": {"label": "Steps", "unit": "count", "color": "#3b82f6"},
    "active_energy": {"label": "Active Energy", "unit": "kJ", "color": "#f97316"},
    "heart_rate_variability": {"label": "HRV", "unit": "ms", "color": "#8b5cf6"}
}

def run_janitor_purge(directory, days_threshold=30):
    """
    Scans the data directory and deletes files older than the threshold days
    based on the timestamp encoded in their filenames.
    """
    now = datetime.now()
    cutoff_date = now - timedelta(days=days_threshold)
    purged_count = 0
    if not os.path.exists(directory):
        return purged_count
    
    for file_name in os.listdir(directory):
        if file_name.startswith("apple_health_") and file_name.endswith(".json"):
            try:
                date_str = file_name.replace("apple_health_", "").replace(".json", "")
                file_time = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                if file_time < cutoff_date:
                    os.remove(os.path.join(directory, file_name))
                    purged_count += 1
            except (ValueError, IndexError):
                continue
    return purged_count

def load_historical_data(directory):
    """
    Reads all valid JSON files within the vault and compiles them
    into a clean Pandas DataFrame for visualization and AI analysis.
    """
    daily_records = {}
    if not os.path.exists(directory):
        return pd.DataFrame()
        
    all_files = sorted(os.listdir(directory))
    
    for file_name in all_files:
        if file_name.endswith(".json"):
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, "r") as f:
                    payload = json.load(f)
                    
                data_obj = payload.get("data", {}) if isinstance(payload, dict) else {}
                metrics_list = data_obj.get("metrics", [])
                workouts_list = data_obj.get("workouts", [])
                
                # 1. Process Metrics dynamically
                if metrics_list:
                    for metric in metrics_list:
                        m_name = metric.get("name", "unknown")
                        m_data = metric.get("data", [])
                        
                        # Use predefined label for UI plots, otherwise dynamically format the name
                        is_ui_metric = m_name in METRIC_CONFIG
                        label = METRIC_CONFIG[m_name]["label"] if is_ui_metric else m_name.replace("_", " ").title()
                        
                        for entry in m_data:
                            raw_date = entry.get("date", "")
                            date_key = raw_date[:10] if raw_date else file_name[13:21]
                            
                            if date_key not in daily_records:
                                daily_records[date_key] = {"Date": date_key}
                                
                            val = entry.get("totalSleep") or entry.get("Avg") or entry.get("qty")
                            
                            if val is not None:
                                daily_records[date_key][label] = round(float(val), 2)
                                
                            # Detail Extraction: Extract specific sleep stages
                            if m_name == "sleep_analysis":
                                for stage in ["rem", "deep", "core", "awake", "inBed", "asleep"]:
                                    if stage in entry and entry[stage] is not None:
                                        stage_label = f"Sleep {stage.capitalize()}"
                                        daily_records[date_key][stage_label] = round(float(entry[stage]), 2)
                                        
                # 2. Process Workouts
                if workouts_list:
                    for workout in workouts_list:
                        start_time = workout.get("start", "")
                        date_key = start_time[:10] if start_time else file_name[13:21]
                        
                        if date_key not in daily_records:
                            daily_records[date_key] = {"Date": date_key}
                            
                        w_name = workout.get("name", "Workout")
                        w_duration_sec = workout.get("duration", 0)
                        w_duration = round(w_duration_sec / 60, 1) if w_duration_sec else 0
                        
                        # Safely parse nested units
                        w_energy = round(workout.get("activeEnergyBurned", {}).get("qty", 0), 1) if isinstance(workout.get("activeEnergyBurned"), dict) else 0
                        w_distance = round(workout.get("distance", {}).get("qty", 0), 2) if isinstance(workout.get("distance"), dict) else 0
                        w_avg_hr = round(workout.get("avgHeartRate", {}).get("qty", 0), 1) if isinstance(workout.get("avgHeartRate"), dict) else 0
                        
                        workouts_str = f"{w_name} ({w_duration} min, {w_energy} kJ, {w_distance} km, Avg HR {w_avg_hr})"
                        
                        # Append if there are multiple workouts in a single day
                        if "Workouts" in daily_records[date_key]:
                            daily_records[date_key]["Workouts"] += f" | {workouts_str}"
                        else:
                            daily_records[date_key]["Workouts"] = workouts_str
                                
                elif isinstance(payload, dict) and "resting_hr" in payload:
                    date_key = payload.get("date", file_name[13:21])
                    if date_key not in daily_records:
                        daily_records[date_key] = {"Date": date_key}
                    daily_records[date_key]["Resting HR"] = float(payload.get("resting_hr", 60))
                    daily_records[date_key]["Sleep Duration"] = float(payload.get("sleep_hours", 7.0))
                    
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
                
    if not daily_records:
        return pd.DataFrame()
        
    combined_records = list(daily_records.values())
    df = pd.DataFrame(combined_records)
    df = df.sort_values("Date").reset_index(drop=True)
    return df
