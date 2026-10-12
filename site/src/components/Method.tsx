import { motion } from "framer-motion";

const steps = [
  {
    name: "Input",
    code: "X ∈ ℝⁿˣᵖ",
    desc: "n samples, p features.",
  },
  {
    name: "Gram",
    code: "C = XᵀX / n + δ Iₚ",
    desc: "Empirical covariance, regularized.",
  },
  {
    name: "Spectral",
    code: "C = U Λ Uᵀ",
    desc: "Eigendecomposition. The geometry of the input.",
  },
  {
    name: "Penalty",
    code: "f(W) = ‖C^½ W‖²_F + ‖W‖₁",
    desc: "Adaptive shrinkage with an explicit gradient or subgradient.",
  },
  {
    name: "MLP",
    code: "forward / backward",
    desc: "Hand-written. No autograd.",
  },
  {
    name: "Adam",
    code: "m, v → θ",
    desc: "The optimizer, the original.",
  },
];

export function Method() {
  return (
    <section id="method" className="relative py-28 sm:py-36">
      <div className="container-x">
        <div className="mx-auto max-w-2xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
            <span className="h-1 w-1 rounded-full bg-brand-400" />
            Method
          </div>
          <h2 className="mt-6 font-display text-[32px] font-semibold leading-[1.1] tracking-tight text-white sm:text-[44px] text-balance">
            The path from data to{" "}
            <span className="text-ink-400">auditable model.</span>
          </h2>
          <p className="mt-5 text-[17px] leading-relaxed text-ink-300 text-pretty">
            Input flows through the Gram matrix into a spectral penalty that
            feeds hand-written backprop into Adam. Every gradient is
            explicit. Nothing is hidden.
          </p>
        </div>

        <div className="relative mx-auto mt-16 max-w-5xl">
          <div className="absolute inset-x-0 top-1/2 hidden h-px bg-gradient-to-r from-transparent via-white/10 to-transparent lg:block" />

          <div className="grid grid-cols-2 gap-4 lg:grid-cols-6">
            {steps.map((s, i) => (
              <motion.div
                key={s.name}
                initial={{ opacity: 0, y: 14 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{
                  duration: 0.55,
                  delay: i * 0.06,
                  ease: [0.22, 1, 0.36, 1],
                }}
                className="relative"
              >
                <div className="group relative h-full rounded-xl border border-white/[0.06] bg-ink-900/40 p-4 transition-colors hover:border-white/10 hover:bg-ink-900/60">
                  <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-500">
                    step · {String(i + 1).padStart(2, "0")}
                  </div>
                  <div className="mt-3 font-display text-[16px] font-semibold text-white">
                    {s.name}
                  </div>
                  <div className="mt-3 rounded-md border border-white/[0.05] bg-ink-950/60 px-2.5 py-2 font-mono text-[12px] text-ink-100">
                    {s.code}
                  </div>
                  <p className="mt-3 text-[12.5px] leading-relaxed text-ink-400">
                    {s.desc}
                  </p>
                </div>
                {i < steps.length - 1 && (
                  <div className="absolute -right-2 top-1/2 hidden -translate-y-1/2 text-ink-600 lg:block">
                    <svg viewBox="0 0 16 16" className="h-3 w-3" fill="currentColor">
                      <path d="M8.22 4.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 8 8.22 4.28a.75.75 0 0 1 0-1.06Z" />
                    </svg>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        <div className="mx-auto mt-16 max-w-3xl text-center">
          <p className="font-mono text-[12px] leading-relaxed text-ink-500">
            A reference implementation of Qasim &amp; Javed (2024) — every
            numerical choice is exposed, every gradient is auditable, every run
            is reproducible with a single integer seed.
          </p>
        </div>
      </div>
    </section>
  );
}
