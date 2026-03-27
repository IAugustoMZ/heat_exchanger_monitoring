const API_BASE = '/api';

export async function fetchScenarios() {
  const res = await fetch(`${API_BASE}/scenarios`);
  if (!res.ok) throw new Error(`Failed to fetch scenarios: ${res.statusText}`);
  return res.json();
}

export async function fetchInterpretability() {
  const res = await fetch(`${API_BASE}/interpretability`);
  if (!res.ok) throw new Error(`Failed to fetch interpretability: ${res.statusText}`);
  return res.json();
}

export async function fetchAllModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  return res.json();
}

export function getFigureUrl(filename: string): string {
  return `${API_BASE}/interpretability/figures/${filename}`;
}

export function createStreamUrl(scenario: string, runId: number = 1, speed: number = 1.0): string {
  return `${API_BASE}/stream/${scenario}?run_id=${runId}&speed=${speed}`;
}
