export interface ScenarioMeta {
  scenario: string;
  label: string;
  runs: number[];
  avg_duration_h: number;
  avg_points: number;
}

export interface DataPoint {
  index: number;
  total_points: number;
  progress: number;

  t_s: number;
  t_h: number;
  T_h_in_K: number;
  T_h_out_K: number;
  T_c_in_K: number;
  T_c_out_K: number;
  delta_P_Pa: number;

  dP_error_predicted: number | null;
  dP_total_predicted: number | null;
  model_available: boolean;

  alarm_predicted: boolean;
  alarm_actual: boolean;
  alarm_threshold_pa: number;

  remaining_runtime_h: number | null;
  rul_lower: number | null;
  rul_upper: number | null;
  dp_error_slope_pa_per_h: number | null;
  defrost_date: string | null;

  dP_error_actual: number;
  delta_f_mean_m: number;
  delta_f_max_m: number;
  U_mean_W_m2K: number;
}

export interface InterpretabilityData {
  model_name: string;
  run_id: string;
  feature_importance: Record<string, number>;
  forecast: Record<string, string | null>;
  figures: { name: string; filename: string }[];
}

export interface ModelInfo {
  model_name: string;
  run_id: string;
  is_best: boolean;
  importance: Record<string, number>;
  forecast: Record<string, string | null>;
}
