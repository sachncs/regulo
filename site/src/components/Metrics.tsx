import { motion } from "framer-motion";

const stats = [
  { value: "6", label: "Penalty families", sub: "From Void to Sparridge" },
  { value: "95%", label: "Test coverage", sub: "Enforced in CI" },
  { value: "0", label: "Framework deps", sub: "NumPy + SciPy only" },
  { value: "1", label: "Integer seed", sub: "For full reproducibility" },
];

const pillars = [
  {
    label: "Research-grade",
    title: "A faithful reproduction",
    desc: "Every formula in the paper is implemented, with explicit smooth gradients and subgradients for reproducible experiments.",
  },
  {
    label: "Readable by design",
    title: "Designed to be read",
    desc: "Small modules, typed public API, py.typed marker. The codebase reads like a textbook.",
  },
  {
    label: "Zero magic",
    title: "No autograd. No surprises.",
    desc: "Forward and backward passes are explicit. There is nothing the library does that you cannot read.",
  },
];

export function Metrics() {
  return (
    <section className="relative py-28 sm:py-36">
      <div className="container-x">
        <div className="mx-auto max-w-2xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
            <span className="h-1 w-1 rounded-full bg-brand-400" />
            Credibility
          </div>
          <h2 className="mt-6 font-display text-[32px] font-semibold leading-[1.1] tracking-tight text-white sm:text-[44px] text-balance">
            Small surface.{" "}
            <span className="text-ink-400">Explicit guarantees.</span>
          </h2>
        </div>

        <div className="mt-16 grid grid-cols-2 gap-3 lg:grid-cols-4">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{
                duration: 0.55,
                delay: i * 0.06,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="rounded-2xl border border-white/[0.06] bg-ink-900/40 p-6"
            >
              <div className="font-display text-[44px] font-semibold leading-none tracking-tightest text-white sm:text-[56px]">
                {s.value}
              </div>
              <div className="mt-3 font-display text-[15px] font-medium text-white">
                {s.label}
              </div>
              <div className="mt-1 font-mono text-[11.5px] text-ink-400">
                {s.sub}
              </div>
            </motion.div>
          ))}
        </div>

        <div className="mt-16 grid grid-cols-1 gap-4 lg:grid-cols-3">
          {pillars.map((p, i) => (
            <motion.div
              key={p.title}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{
                duration: 0.6,
                delay: 0.1 + i * 0.08,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="rounded-2xl border border-white/[0.06] bg-gradient-to-b from-white/[0.02] to-transparent p-7"
            >
              <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-400">
                {p.label}
              </div>
              <h3 className="mt-4 font-display text-[22px] font-semibold tracking-tight text-white">
                {p.title}
              </h3>
              <p className="mt-3 text-[15px] leading-relaxed text-ink-300">
                {p.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
