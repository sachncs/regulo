import { motion } from "framer-motion";

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden pt-32 sm:pt-40">
      <div className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[520px] grid-bg" />
      <div className="container-x">
        <div className="grid items-center gap-12 lg:grid-cols-[.9fr_1.1fr] lg:gap-20">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .55 }}>
            <div className="eyebrow font-mono text-[11px] uppercase">A small numerical library · v0.2.0</div>
            <h1 className="mt-5 max-w-xl font-display text-[48px] font-semibold leading-[.98] tracking-[-.06em] text-ink-950 sm:text-[68px]">
              Regularization you can <span className="text-brand-700">trace.</span>
            </h1>
            <p className="mt-7 max-w-xl text-[18px] leading-relaxed text-ink-300 sm:text-[20px]">
              regulo is a pure-Python reference implementation for covariance-aware regularizers. Read the formula, inspect the gradient, reproduce the run.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <a href="#install" className="inline-flex h-11 items-center gap-2 rounded-lg bg-brand-700 px-5 font-mono text-[13px] font-medium text-white transition-colors hover:bg-brand-800">pip install regulo <span aria-hidden="true">↗</span></a>
              <a href="https://github.com/sachncs/regulo" target="_blank" rel="noreferrer" className="inline-flex h-11 items-center gap-2 rounded-lg border border-ink-300 bg-white/50 px-5 text-[13px] font-medium text-ink-950 transition-colors hover:bg-white">Read the source <span aria-hidden="true">↗</span></a>
            </div>
            <div className="mt-9 grid max-w-lg grid-cols-2 gap-x-6 gap-y-3 border-t border-ink-200 pt-5 font-mono text-[11px] uppercase tracking-[.12em] text-ink-400 sm:grid-cols-4"><span>NumPy + SciPy</span><span>Python 3.10+</span><span>MIT licensed</span><span>95% covered</span></div>
          </motion.div>
          <HeroDiagram />
        </div>
      </div>
    </section>
  );
}

function HeroDiagram() {
  return (
    <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .65, delay: .1 }} className="paper-card relative overflow-hidden rounded-2xl">
      <div className="flex items-center justify-between border-b border-ink-200 px-5 py-4"><span className="font-mono text-[11px] uppercase tracking-[.16em] text-ink-400">model anatomy / 01</span><span className="rounded-full bg-teal-100 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-teal-800">auditable</span></div>
      <div className="p-5 sm:p-8">
        <div className="rounded-xl border border-ink-200 bg-[#f1eadf] p-5 sm:p-7">
          <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[.16em] text-ink-400"><span>input geometry</span><span>seed: 0</span></div>
          <svg viewBox="0 0 520 184" className="mt-5 h-auto w-full" role="img" aria-label="Data flows through a Gram matrix into a spectral penalty and an optimizer">
            <path d="M22 93h64M166 93h66M314 93h66M428 93h66" stroke="#b1aaa0" strokeWidth="1.5" strokeDasharray="4 5" />
            <g fill="#fffdf8" stroke="#1f4b75" strokeWidth="2"><rect x="8" y="59" width="78" height="68" rx="8"/><rect x="232" y="59" width="82" height="68" rx="8"/><rect x="380" y="59" width="48" height="68" rx="8"/><rect x="494" y="59" width="20" height="68" rx="8"/></g>
            <g fill="none" stroke="#2d7b75" strokeWidth="2"><ellipse cx="47" cy="93" rx="24" ry="13"/><ellipse cx="47" cy="93" rx="10" ry="24"/></g><circle cx="47" cy="93" r="3" fill="#2d7b75" />
            <text x="47" y="151" textAnchor="middle" fontFamily="monospace" fontSize="11" fill="#52656a">X</text><text x="273" y="88" textAnchor="middle" fontFamily="monospace" fontSize="13" fill="#172b35">C = XᵀX/n</text><text x="273" y="108" textAnchor="middle" fontFamily="monospace" fontSize="11" fill="#d66b4e">+ δI</text><text x="404" y="96" textAnchor="middle" fontFamily="monospace" fontSize="12" fill="#172b35">UΛUᵀ</text><text x="504" y="96" textAnchor="middle" fontFamily="monospace" fontSize="11" fill="#172b35">W</text>
            <text x="123" y="42" textAnchor="middle" fontFamily="monospace" fontSize="10" fill="#7a8582">empirical Gram</text><text x="347" y="42" textAnchor="middle" fontFamily="monospace" fontSize="10" fill="#7a8582">spectral basis</text>
          </svg>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-[1.25fr_.75fr]">
          <div className="rounded-xl bg-[#172b35] p-4 font-mono text-[12px] leading-6 text-[#e8f1ed]"><div className="text-[#9eb8b3]"># inspect the penalty</div><div><span className="text-[#f3c879]">v</span>, <span className="text-[#f3c879]">g</span> = penalty.value(W), penalty.grad(W)</div><div className="text-[#9eb8b3]"># nothing hidden behind autograd</div></div>
          <div className="rounded-xl border border-ink-200 bg-white/60 p-4"><div className="font-mono text-[10px] uppercase tracking-wider text-ink-400">contract</div><div className="mt-3 text-[15px] font-medium text-ink-950">value(W)</div><div className="text-[15px] font-medium text-ink-950">grad(W)</div><div className="mt-2 font-mono text-[11px] text-coral-700">explicit by default</div></div>
        </div>
      </div>
      <div className="border-t border-ink-200 px-5 py-3 font-mono text-[10px] uppercase tracking-[.14em] text-ink-400">X → C<sub>δ,n</sub> → UΛUᵀ → penalty → optimizer</div>
    </motion.div>
  );
}
