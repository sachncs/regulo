import { motion } from "framer-motion";

export function Hero() {
  return (
    <section id="top" className="relative isolate overflow-hidden pt-36 sm:pt-44">
      <div className="absolute inset-x-0 top-0 -z-10 h-[640px] grid-bg opacity-60" />
      <div className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[640px] bg-gradient-to-b from-transparent via-transparent to-ink-950" />
      <div className="pointer-events-none absolute left-1/2 top-[-200px] -z-10 h-[640px] w-[1100px] -translate-x-1/2 rounded-full bg-grid-fade opacity-80 blur-3xl" />

      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto max-w-3xl text-center"
        >
          <a
            href="https://github.com/sachncs/regulo"
            target="_blank"
            rel="noreferrer"
            className="group inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[12px] text-ink-200 backdrop-blur-md transition-colors hover:bg-white/[0.07]"
          >
            <span className="relative inline-flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
            </span>
            Reproducible with a single integer seed
            <svg
              viewBox="0 0 16 16"
              className="h-3 w-3 text-ink-400 transition-transform group-hover:translate-x-0.5"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M6.22 4.22a.75.75 0 0 1 1.06 0l3.25 3.25a.75.75 0 0 1 0 1.06l-3.25 3.25a.75.75 0 0 1-1.06-1.06L8.94 8 6.22 5.28a.75.75 0 0 1 0-1.06Z" />
              <path d="M2 8a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5A.75.75 0 0 1 2 8Z" />
            </svg>
          </a>

          <h1 className="mt-8 font-display text-[44px] font-semibold leading-[1.05] tracking-tightest text-white sm:text-[64px] lg:text-[80px] text-balance">
            Regularization,{" "}
            <span className="relative inline-block">
              <span className="bg-gradient-to-br from-white via-white to-ink-300 bg-clip-text text-transparent">
                made auditable.
              </span>
              <span className="absolute -bottom-1 left-0 right-0 h-[6px] rounded-full bg-gradient-to-r from-transparent via-brand-400 to-transparent opacity-60 blur-md" />
            </span>
          </h1>

          <p className="mx-auto mt-7 max-w-2xl text-[18px] leading-relaxed text-ink-300 sm:text-[20px] text-pretty">
            <span className="text-white">regulo</span> is a pure-Python,
            NumPy and SciPy reference implementation of covariance-aware
            regularizers for small MLPs. Every penalty is auditable. Every
            smooth gradient is analytical; non-smooth penalties expose
            explicit subgradients. Every run is seedable.
          </p>

          <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <a
              href="#install"
              className="group inline-flex h-11 items-center gap-2 rounded-full bg-white px-5 text-[14px] font-medium text-ink-950 shadow-[0_8px_24px_-6px_rgba(255,255,255,0.4)] transition-transform hover:scale-[1.02]"
            >
              <svg
                viewBox="0 0 16 16"
                className="h-4 w-4"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M4.72 5.97a.75.75 0 0 1 1.06 0l2.47 2.47 2.47-2.47a.75.75 0 1 1 1.06 1.06L9.31 9.5l2.47 2.47a.75.75 0 1 1-1.06 1.06l-2.47-2.47-2.47 2.47a.75.75 0 0 1-1.06-1.06L7.69 9.5 5.22 7.03a.75.75 0 0 1 0-1.06Z" />
                <path d="M2.75 4a.75.75 0 0 0-.75.75v6.5c0 .414.336.75.75.75h1a.75.75 0 0 0 0-1.5h-.25V5.5h.25a.75.75 0 0 0 0-1.5h-1Z" />
                <path d="M13.25 4a.75.75 0 0 1 .75.75v6.5a.75.75 0 0 1-.75.75h-1a.75.75 0 0 1 0-1.5h.25V5.5h-.25a.75.75 0 0 1 0-1.5h1Z" />
              </svg>
              <span className="font-mono">pip install regulo</span>
              <svg
                viewBox="0 0 16 16"
                className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M8.22 4.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 8 8.22 4.28a.75.75 0 0 1 0-1.06Z" />
                <path d="M3.75 8a.75.75 0 0 1 .75-.75h8.5a.75.75 0 0 1 0 1.5h-8.5A.75.75 0 0 1 3.75 8Z" />
              </svg>
            </a>
            <a
              href="https://github.com/sachncs/regulo"
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-11 items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-5 text-[14px] font-medium text-white backdrop-blur-md transition-colors hover:bg-white/[0.06]"
            >
              <svg
                viewBox="0 0 24 24"
                className="h-4 w-4"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2Z" />
              </svg>
              View on GitHub
            </a>
          </div>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-x-6 gap-y-3 text-[12px] text-ink-400">
            <div className="inline-flex items-center gap-2">
              <span className="h-1 w-1 rounded-full bg-emerald-400" />
              MIT licensed
            </div>
            <div className="inline-flex items-center gap-2">
              <span className="h-1 w-1 rounded-full bg-emerald-400" />
              NumPy + SciPy only
            </div>
            <div className="inline-flex items-center gap-2">
              <span className="h-1 w-1 rounded-full bg-emerald-400" />
              Python 3.10 – 3.13
            </div>
            <div className="inline-flex items-center gap-2">
              <span className="h-1 w-1 rounded-full bg-emerald-400" />
              95% test coverage
            </div>
          </div>
        </motion.div>

        <HeroVisual />
      </div>
    </section>
  );
}

function HeroVisual() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.9, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="relative mx-auto mt-20 max-w-5xl"
    >
      <div className="pointer-events-none absolute -inset-x-20 -inset-y-10 -z-10 rounded-[36px] bg-gradient-to-b from-brand-500/20 via-accent-500/10 to-transparent blur-3xl" />

      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-ink-900/60 shadow-glow">
        <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-2.5">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]/80" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]/80" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]/80" />
          </div>
          <div className="font-mono text-[11px] text-ink-400">
            regulo / runner.py
          </div>
          <div className="font-mono text-[11px] text-ink-500">adam · 500 epochs</div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.05fr_1fr]">
          <div className="border-b border-white/[0.06] p-5 font-mono text-[13px] leading-7 lg:border-b-0 lg:border-r">
            <div className="flex">
              <span className="select-none pr-4 text-right text-ink-600">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div key={i}>{i + 1}</div>
                ))}
              </span>
              <pre className="overflow-x-auto text-pretty">
                <code>
                  <span className="text-ink-500"># covariance-aware</span>
                  {"\n"}
                  <span className="text-accent-300">from</span> regulo{" "}
                  <span className="text-accent-300">import</span>{" "}
                  <span className="text-ink-100">Covridge</span>
                  {"\n"}
                  {"\n"}
                  <span className="text-ink-500"># empirical gram matrix</span>
                  {"\n"}
                  gram = <span className="text-ink-100">X</span>.T @{" "}
                  <span className="text-ink-100">X</span> / n +{" "}
                  <span className="text-brand-300">1e-3</span> * np.eye(p)
                  {"\n"}
                  {"\n"}
                  penalty = <span className="text-ink-100">Covridge</span>(
                  {"\n  "}
                  lambda1=<span className="text-brand-300">0.10</span>,
                  {"\n  "}
                  lambda2=<span className="text-brand-300">0.01</span>,
                  {"\n  "}
                  gram=gram,{"\n"}
                  )
                  {"\n"}
                  {"\n"}
                  <span className="text-ink-500"># value &amp; gradient</span>
                  {"\n"}
                  v = penalty.value(W1, 0)
                  {"\n"}
                  g = penalty.grad(W1, 0)
                </code>
              </pre>
            </div>
          </div>

          <div className="relative p-5">
            <div className="mb-4 flex items-center justify-between">
              <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-400">
                training · test mse
              </div>
              <div className="flex items-center gap-1.5 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-0.5 font-mono text-[10px] text-emerald-300">
                <span className="h-1 w-1 rounded-full bg-emerald-400" />
                converging
              </div>
            </div>

            <div className="relative h-[180px] overflow-hidden rounded-lg border border-white/[0.06] bg-gradient-to-b from-white/[0.02] to-transparent">
              <ConvergenceChart />
            </div>

            <div className="mt-4 grid grid-cols-3 gap-2">
              {[
                ["epochs", "500"],
                ["batch", "32"],
                ["seed", "0"],
              ].map(([k, v]) => (
                <div
                  key={k}
                  className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2"
                >
                  <div className="font-mono text-[10px] uppercase tracking-wider text-ink-500">
                    {k}
                  </div>
                  <div className="mt-0.5 font-mono text-[14px] text-ink-100">
                    {v}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-ink-950 to-transparent" />
    </motion.div>
  );
}

function ConvergenceChart() {
  const path =
    "M0,160 C40,150 70,130 100,118 C140,98 180,92 220,82 C260,72 300,62 340,52 C380,42 420,36 460,32 C500,28 540,26 580,24 C620,22 660,20 700,18";
  const area =
    "M0,160 C40,150 70,130 100,118 C140,98 180,92 220,82 C260,72 300,62 340,52 C380,42 420,36 460,32 C500,28 540,26 580,24 C620,22 660,20 700,18 L700,180 L0,180 Z";

  return (
    <svg
      viewBox="0 0 700 180"
      className="absolute inset-0 h-full w-full"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id="reg-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#4656F5" />
          <stop offset="1" stopColor="#3F7CD0" />
        </linearGradient>
        <linearGradient id="reg-grad-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#4656F5" stopOpacity="0.4" />
          <stop offset="1" stopColor="#4656F5" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="reg-grad-line" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#8A9FFF" />
          <stop offset="1" stopColor="#3F7CD0" />
        </linearGradient>
      </defs>

      {[0, 1, 2, 3].map((i) => (
        <line
          key={i}
          x1="0"
          y1={(180 / 4) * (i + 1)}
          x2="700"
          y2={(180 / 4) * (i + 1)}
          stroke="rgba(255,255,255,0.04)"
          strokeDasharray="2 4"
        />
      ))}

      <path d={area} fill="url(#reg-grad-fill)" />
      <path
        d={path}
        stroke="url(#reg-grad-line)"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
      />
      <circle cx="700" cy="18" r="3" fill="#8A9FFF" />
      <circle cx="700" cy="18" r="6" fill="#4656F5" opacity="0.3" />
    </svg>
  );
}
