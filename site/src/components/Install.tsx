import { useState } from "react";

export function Install() {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);

  const copy = () => {
    setCopyError(false);
    navigator.clipboard.writeText("python -m pip install -e .").then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1800);
      },
      () => setCopyError(true),
    );
  };

  return (
    <section id="install" className="relative py-28 sm:py-36">
      <div className="container-x">
        <div className="relative mx-auto max-w-4xl overflow-hidden rounded-3xl border border-white/[0.08] bg-gradient-to-b from-ink-900/80 to-ink-950/80 p-10 sm:p-16">
          <div className="pointer-events-none absolute inset-0 -z-10 bg-grid-fade opacity-60" />
          <div className="pointer-events-none absolute -inset-x-20 -top-20 -z-10 h-[300px] bg-gradient-to-b from-brand-500/30 via-accent-500/20 to-transparent blur-3xl" />

          <div className="mx-auto max-w-2xl text-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
              <span className="h-1 w-1 rounded-full bg-emerald-400" />
              Get started
            </div>
            <h2 className="mt-6 font-display text-[34px] font-semibold leading-[1.05] tracking-tight text-white sm:text-[48px] text-balance">
              One command.{" "}
              <span className="bg-gradient-to-br from-white via-ink-200 to-ink-400 bg-clip-text text-transparent">
                A reproducible starting point.
              </span>
            </h2>
            <p className="mt-5 text-[17px] leading-relaxed text-ink-300 text-pretty">
              regulo is not publishing a PyPI release yet. Clone the repository
              and install it in editable mode so the source and experiments stay
              together.
            </p>

            <div className="mt-10 flex items-center justify-center">
              <button
                onClick={copy}
                aria-label="Copy source install command"
                aria-live="polite"
                className="group inline-flex items-center gap-3 rounded-full border border-white/10 bg-ink-950/80 px-5 py-3 font-mono text-[14px] text-ink-100 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.6)] transition-colors hover:border-white/20"
              >
                <span className="text-ink-500">$</span>
                <span>pip install -e .</span>
                <span className="ml-1 inline-flex h-5 w-5 items-center justify-center rounded-md border border-white/10 text-ink-300 transition-colors group-hover:text-white">
                  {copied ? (
                    <svg viewBox="0 0 16 16" className="h-3 w-3" fill="currentColor">
                      <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z" />
                    </svg>
                  ) : (
                    <svg
                      viewBox="0 0 16 16"
                      className="h-3 w-3"
                      fill="currentColor"
                    >
                      <path d="M5 2h6a1 1 0 0 1 1 1v1H8.5a1.5 1.5 0 0 0-1.5 1.5V9H4a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1h1Zm-3 6h4v1.5A1.5 1.5 0 0 0 7.5 11H9v3a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V8Zm5 0a1 1 0 0 1 1-1h3.5a1.5 1.5 0 0 1 1.5 1.5V11a1 1 0 0 1-1 1H10a1 1 0 0 1-1-1V8Z" />
                    </svg>
                  )}
                </span>
              </button>
              <span className="sr-only" role="status">
                {copyError ? "Copy failed" : copied ? "Copied" : ""}
              </span>
            </div>

            <div className="mt-10 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <a
                href="https://github.com/sachncs/regulo"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2.5 text-[13px] text-white transition-colors hover:bg-white/[0.06]"
              >
                Source
                <svg
                  viewBox="0 0 16 16"
                  className="h-3 w-3"
                  fill="currentColor"
                >
                  <path d="M3.75 8a.75.75 0 0 1 .75-.75h6.5a.75.75 0 0 1 0 1.5h-6.5A.75.75 0 0 1 3.75 8Z" />
                  <path d="M8.22 4.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 8 8.22 4.28a.75.75 0 0 1 0-1.06Z" />
                </svg>
              </a>
              <a
                href="#/docs/reproduce"
                className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2.5 text-[13px] text-white transition-colors hover:bg-white/[0.06]"
              >
                Reproduction
                <svg
                  viewBox="0 0 16 16"
                  className="h-3 w-3"
                  fill="currentColor"
                >
                  <path d="M3.75 8a.75.75 0 0 1 .75-.75h6.5a.75.75 0 0 1 0 1.5h-6.5A.75.75 0 0 1 3.75 8Z" />
                  <path d="M8.22 4.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 8 8.22 4.28a.75.75 0 0 1 0-1.06Z" />
                </svg>
              </a>
              <a
                href="#/docs/changelog"
                className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2.5 text-[13px] text-white transition-colors hover:bg-white/[0.06]"
              >
                Changelog
                <svg
                  viewBox="0 0 16 16"
                  className="h-3 w-3"
                  fill="currentColor"
                >
                  <path d="M3.75 8a.75.75 0 0 1 .75-.75h6.5a.75.75 0 0 1 0 1.5h-6.5A.75.75 0 0 1 3.75 8Z" />
                  <path d="M8.22 4.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 8 8.22 4.28a.75.75 0 0 1 0-1.06Z" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
