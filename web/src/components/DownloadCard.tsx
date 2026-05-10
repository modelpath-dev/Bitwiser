import type { Model } from "../lib/api";
import { humanizeBytes } from "../lib/format";
import type { CompressedArtifact } from "./ShrinkPanel";

type Props = {
  model: Model;
  artifact: CompressedArtifact;
};

function basename(path: string): string {
  return path.split("/").pop() || path;
}

function friendlyName(model: Model, quant: string): string {
  // "library/mistral:7b" → "mistral_7b"
  return (
    model.name.replace(/^library\//, "").replace(/[:/\\]/g, "_") +
    "." +
    quant +
    ".gguf"
  );
}

export function DownloadCard({ model, artifact }: Props) {
  const file = basename(artifact.outputPath);
  const friendly = friendlyName(model, artifact.quant);
  const href = `/api/download/${encodeURIComponent(file)}?as_name=${encodeURIComponent(friendly)}`;

  return (
    <div className="glass relative overflow-hidden p-5">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 opacity-50"
        style={{
          background:
            "radial-gradient(80% 100% at 100% 0%, rgba(34,211,238,0.14), transparent 70%)",
        }}
      />
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <div className="text-[11px] uppercase tracking-[0.2em] text-[--color-ink-400]">
            Quantized model
          </div>
          <div className="mt-2 flex flex-wrap items-baseline gap-2">
            <span className="font-mono text-base text-[--color-ink-50]">
              {friendly}
            </span>
            <span className="pill pill-cyan">{artifact.quant}</span>
            <span className="pill">{humanizeBytes(artifact.outputBytes)}</span>
          </div>
          <div className="mt-1 truncate font-mono text-[11px] text-[--color-ink-400]">
            {artifact.outputPath}
          </div>
        </div>
        <a
          href={href}
          download={friendly}
          className="btn-primary shrink-0"
        >
          Download
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path
              d="M8 2v9m0 0l-3.5-3.5M8 11l3.5-3.5M2 14h12"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </a>
      </div>
    </div>
  );
}
