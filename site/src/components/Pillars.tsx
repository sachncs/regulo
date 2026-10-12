import { motion } from "framer-motion";
import { cn } from "../lib/cn";

type Pillar = {
  label: string;
  title: string;
  desc: string;
  icon: React.ReactNode;
  accent: string;
};

const pillars: Pillar[] = [
  {
    label: "01",
    title: "Auditable",
    desc: "Every penalty is a single module. Every value and gradient fits in a REPL. No hidden autograd, no framework magic.",
    accent: "from-brand-400/40 to-brand-500/0",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="h-5 w-5">
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    label: "02",
    title: "Covariance-aware",
    desc: "Covridge and Sparridge apply geometry-aware shrinkage along the eigenvectors of the empirical Gram matrix.",
    accent: "from-accent-400/40 to-accent-500/0",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="h-5 w-5">
        <circle cx="12" cy="12" r="9" />
        <ellipse cx="12" cy="12" rx="9" ry="3.5" />
        <ellipse cx="12" cy="12" rx="3.5" ry="9" />
      </svg>
    ),
  },
  {
    label: "03",
    title: "Reproducible",
    desc: "A single integer seed pins every run. NumPy and SciPy keep the numerical core transparent. CI spans Linux, macOS, and Windows.",
    accent: "from-emerald-400/40 to-emerald-500/0",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="h-5 w-5">
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" strokeLinecap="round" />
        <circle cx="12" cy="12" r="3.5" />
      </svg>
    ),
  },
];

export function Pillars() {
  return (
    <section id="product" className="relative py-28 sm:py-36">
      <div className="container-x">
        <SectionHeader
          eyebrow="Overview"
          title={
            <>
            A regularizer library built{" "}
              <span className="text-ink-400">for readable experiments.</span>
            </>
          }
          subtitle="regulo ships six penalties, two losses, an MLP, the Adam optimizer, and a Runner — explicit enough to inspect, test, and reproduce."
        />

        <div className="mt-16 grid grid-cols-1 gap-4 md:grid-cols-3">
          {pillars.map((p, i) => (
            <motion.div
              key={p.label}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.6, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }}
              className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-gradient-to-b from-white/[0.03] to-transparent p-6 transition-colors hover:border-white/10 hover:bg-white/[0.04]"
            >
              <div
                className={cn(
                  "absolute -inset-px -z-10 rounded-2xl bg-gradient-to-b opacity-0 transition-opacity duration-500 group-hover:opacity-100",
                  p.accent
                )}
              />
              <div className="flex items-center gap-3">
                <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.08] bg-white/[0.04] text-ink-200 transition-colors group-hover:text-white">
                  {p.icon}
                </span>
                <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-500">
                  {p.label}
                </span>
              </div>
              <h3 className="mt-8 font-display text-[22px] font-semibold tracking-tight text-white">
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

export function SectionHeader({
  eyebrow,
  title,
  subtitle,
  className,
}: {
  eyebrow: string;
  title: React.ReactNode;
  subtitle?: string;
  className?: string;
}) {
  return (
    <div className={cn("mx-auto max-w-2xl text-center", className)}>
      <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
        <span className="h-1 w-1 rounded-full bg-brand-400" />
        {eyebrow}
      </div>
      <h2 className="mt-6 font-display text-[32px] font-semibold leading-[1.1] tracking-tight text-white sm:text-[44px] text-balance">
        {title}
      </h2>
      {subtitle && (
        <p className="mt-5 text-[17px] leading-relaxed text-ink-300 text-pretty">
          {subtitle}
        </p>
      )}
    </div>
  );
}
