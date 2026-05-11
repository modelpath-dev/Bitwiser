export function humanizeBytes(n: number): string {
  if (!isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 2)} ${units[i]}`;
}

export function ratioPct(input: number, output: number): string {
  if (!input) return "—";
  return `${((1 - output / input) * 100).toFixed(1)}%`;
}

export function inferFamily(model: { family: string | null; architecture: string | null; name: string }): string {
  const haystack = `${model.architecture ?? ""} ${model.family ?? ""} ${model.name}`.toLowerCase();
  for (const f of ["gemma", "llama", "mistral", "phi", "qwen"]) {
    if (haystack.includes(f)) return f;
  }
  return "model";
}
