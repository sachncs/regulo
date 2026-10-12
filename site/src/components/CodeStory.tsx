import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Token = { t: string; c?: string };
type Line = { line: string; tokens: Token[] };
type Tab = {
  id: string;
  label: string;
  filename: string;
  language: string;
  code: Line[];
};

const tabs: Tab[] = [
  {
    id: "fit",
    label: "Train",
    filename: "runner.py",
    language: "python",
    code: [
      {
        line: "from regulo import Adam, MLP, Covridge, Runner, Square",
        tokens: [
          { t: "from" },
          { t: " regulo " },
          { t: "import", c: "kw" },
          { t: " Adam, MLP, Covridge, Runner, Square" },
        ],
      },
      {
        line: "from regulo import synth",
        tokens: [
          { t: "from" },
          { t: " regulo " },
          { t: "import", c: "kw" },
          { t: " synth" },
        ],
      },
      { line: "", tokens: [] },
      {
        line: "x, y = synth(n=200, p=20, k=10, rho=0.25, seed=0)",
        tokens: [
          { t: "x, y " },
          { t: "= " },
          { t: "synth", c: "fn" },
          { t: "(n=" },
          { t: "200", c: "num" },
          { t: ", p=" },
          { t: "20", c: "num" },
          { t: ", k=" },
          { t: "10", c: "num" },
          { t: ", rho=" },
          { t: "0.25", c: "num" },
          { t: ", seed=" },
          { t: "0", c: "num" },
          { t: ")" },
        ],
      },
      { line: "", tokens: [] },
      {
        line: "gram = x.T @ x / 200 + 1e-3 * np.eye(20)",
        tokens: [
          { t: "gram " },
          { t: "= " },
          { t: "x.T @ x / " },
          { t: "200", c: "num" },
          { t: " + " },
          { t: "1e-3", c: "num" },
          { t: " * np.eye(20)" },
        ],
      },
      { line: "", tokens: [] },
      {
        line: "runner = Runner(",
        tokens: [
          { t: "runner " },
          { t: "= " },
          { t: "Runner", c: "fn" },
          { t: "(" },
        ],
      },
      {
        line: "    MLP([20, 64, 32, 1], seed=0),",
        tokens: [
          { t: "    MLP", c: "fn" },
          { t: "([20, " },
          { t: "64", c: "num" },
          { t: ", " },
          { t: "32", c: "num" },
          { t: ", " },
          { t: "1", c: "num" },
          { t: "], seed=" },
          { t: "0", c: "num" },
          { t: ")," },
        ],
      },
      {
        line: "    Square(),",
        tokens: [{ t: "    Square", c: "fn" }, { t: "()," }],
      },
      {
        line: "    Covridge(lambda1=0.10, lambda2=0.01, gram=gram),",
        tokens: [
          { t: "    Covridge", c: "fn" },
          { t: "(lambda1=" },
          { t: "0.10", c: "num" },
          { t: ", lambda2=" },
          { t: "0.01", c: "num" },
          { t: ", gram=gram)," },
        ],
      },
      {
        line: "    Adam(lr=1e-3), batch=32, epochs=500,",
        tokens: [
          { t: "    Adam", c: "fn" },
          { t: "(lr=" },
          { t: "1e-3", c: "num" },
          { t: "), batch=" },
          { t: "32", c: "num" },
          { t: ", epochs=" },
          { t: "500", c: "num" },
          { t: "," },
        ],
      },
      { line: ")", tokens: [{ t: ")" }] },
      {
        line: "runner.fit(x, y, seed=0)",
        tokens: [
          { t: "runner", c: "fn" },
          { t: "." },
          { t: "fit", c: "fn" },
          { t: "(x, y, seed=" },
          { t: "0", c: "num" },
          { t: ")" },
        ],
      },
    ],
  },
  {
    id: "search",
    label: "Search",
    filename: "tune.py",
    language: "python",
    code: [
      {
        line: "from regulo import Adam, MLP, Ridge, Runner, search, synth, Square",
        tokens: [
          { t: "from" },
          { t: " regulo " },
          { t: "import", c: "kw" },
          { t: " Adam, MLP, Ridge, Runner, search" },
        ],
      },
      { line: "", tokens: [] },
      {
        line: "x, y = synth(n=200, p=20, k=10, rho=0.25, seed=0)",
        tokens: [
          { t: "x, y " },
          { t: "= " },
          { t: "synth", c: "fn" },
          { t: "(n=" },
          { t: "200", c: "num" },
          { t: ", p=" },
          { t: "20", c: "num" },
          { t: ", k=" },
          { t: "10", c: "num" },
          { t: ", rho=" },
          { t: "0.25", c: "num" },
          { t: ", seed=" },
          { t: "0", c: "num" },
          { t: ")" },
        ],
      },
      { line: "", tokens: [] },
      {
        line: "best, score = search(",
        tokens: [
          { t: "best, score " },
          { t: "= " },
          { t: "search", c: "fn" },
          { t: "(" },
        ],
      },
      { line: "    x, y,", tokens: [{ t: "    x, y," }] },
      {
        line: "    shape=[20, 64, 32, 1],",
        tokens: [
          { t: "    shape=[" },
          { t: "20", c: "num" },
          { t: ", " },
          { t: "64", c: "num" },
          { t: ", " },
          { t: "32", c: "num" },
          { t: ", " },
          { t: "1", c: "num" },
          { t: "]," },
        ],
      },
      {
        line: "    method=\"ridge\",",
        tokens: [
          { t: "    method=" },
          { t: "\"ridge\"", c: "str" },
          { t: "," },
        ],
      },
      {
        line: "    grid=[{\"lam\": v} for v in (1e-3, 1e-2, 1e-1, 0.5)],",
        tokens: [
          { t: "    grid=[" },
          { t: "{\"lam\": v}", c: "str" },
          { t: " " },
          { t: "for", c: "kw" },
          { t: " v " },
          { t: "in" },
          { t: " (" },
          { t: "1e-3", c: "num" },
          { t: ", " },
          { t: "1e-2", c: "num" },
          { t: ", " },
          { t: "1e-1", c: "num" },
          { t: ", " },
          { t: "0.5", c: "num" },
          { t: ")]," },
        ],
      },
      {
        line: "    loss_fn=Square(), folds=5, epochs=200, seed=0,",
        tokens: [
          { t: "    loss_fn=Square(), folds=" },
          { t: "5", c: "num" },
          { t: ", epochs=" },
          { t: "200", c: "num" },
          { t: ", seed=" },
          { t: "0", c: "num" },
          { t: "," },
        ],
      },
      { line: ")", tokens: [{ t: ")" }] },
      {
        line: "print(\"best lam:\", best, \"score:\", score)",
        tokens: [
          { t: "print", c: "fn" },
          { t: "(" },
          { t: "\"best lam:\"", c: "str" },
          { t: ", best, " },
          { t: "\"score:\"", c: "str" },
          { t: ", score)" },
        ],
      },
    ],
  },
  {
    id: "save",
    label: "Persist",
    filename: "store.py",
    language: "python",
    code: [
      {
        line: "from regulo import save, load",
        tokens: [
          { t: "from" },
          { t: " regulo " },
          { t: "import", c: "kw" },
          { t: " save, load" },
        ],
      },
      { line: "", tokens: [] },
      {
        line: "# weights + hyperparameters in a single npz",
        tokens: [{ t: "# weights + hyperparameters in a single npz" }],
      },
      {
        line: "save(\"model.npz\", runner)",
        tokens: [
          { t: "save", c: "fn" },
          { t: "(" },
          { t: "\"model.npz\"", c: "str" },
          { t: ", runner)" },
        ],
      },
      { line: "", tokens: [] },
      {
        line: "# no pickle. no surprises.",
        tokens: [{ t: "# no pickle. no surprises." }],
      },
      {
        line: "model = load(\"model.npz\")",
        tokens: [
          { t: "model " },
          { t: "= " },
          { t: "load", c: "fn" },
          { t: "(" },
          { t: "\"model.npz\"", c: "str" },
          { t: ")" },
        ],
      },
      {
        line: "yhat = model.predict(x_test)",
        tokens: [
          { t: "yhat " },
          { t: "= " },
          { t: "model", c: "fn" },
          { t: "." },
          { t: "predict", c: "fn" },
          { t: "(x_test)" },
        ],
      },
    ],
  },
];

const tokenColors: Record<string, string> = {
  kw: "text-accent-300",
  fn: "text-brand-300",
  num: "text-amber-200/90",
  str: "text-emerald-300/90",
};

export function CodeStory() {
  const [active, setActive] = useState(tabs[0].id);
  const current = tabs.find((t) => t.id === active)!;

  return (
    <section id="reproduce" className="relative py-28 sm:py-36">
      <div className="container-x">
        <div className="mx-auto max-w-2xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
            <span className="h-1 w-1 rounded-full bg-brand-400" />
            In code
          </div>
          <h2 className="mt-6 font-display text-[32px] font-semibold leading-[1.1] tracking-tight text-white sm:text-[44px] text-balance">
            Three lines of intent.{" "}
            <span className="text-ink-400">Zero magic.</span>
          </h2>
          <p className="mt-5 text-[17px] leading-relaxed text-ink-300 text-pretty">
            The public API is the smallest one we could ship that still
            expresses the full experiment.
          </p>
        </div>

        <div className="mx-auto mt-14 max-w-4xl">
          <div className="overflow-hidden rounded-2xl border border-white/[0.08] bg-ink-900/60 shadow-glow">
            <div className="hidden items-center justify-between border-b border-white/[0.06] px-4 py-3 sm:flex">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]/80" />
                <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]/80" />
                <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]/80" />
              </div>

              <div className="flex items-center gap-1 rounded-full border border-white/[0.06] bg-white/[0.03] p-1">
                {tabs.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setActive(t.id)}
                    className={`rounded-full px-3 py-1 font-mono text-[11px] uppercase tracking-wider transition-colors ${
                      active === t.id
                        ? "bg-white text-ink-950"
                        : "text-ink-300 hover:text-white"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              <div className="font-mono text-[11px] text-ink-400">
                {current.filename}
              </div>
            </div>

            <div className="border-b border-white/[0.06] p-2 sm:hidden">
              <div className="flex items-center gap-1">
                {tabs.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setActive(t.id)}
                    className={`rounded-full px-3 py-1 font-mono text-[11px] uppercase tracking-wider transition-colors ${
                      active === t.id
                        ? "bg-white text-ink-950"
                        : "text-ink-300 hover:text-white"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            <AnimatePresence mode="wait">
              <motion.pre
                key={current.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.25 }}
                className="overflow-x-auto p-6 font-mono text-[13.5px] leading-7 text-ink-200"
              >
                <code className="block">
                  {current.code.map((l, i) => (
                    <div key={i} className="flex">
                      <span className="select-none pr-4 text-right font-mono text-[12px] text-ink-600">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="whitespace-pre">
                        {l.tokens.map((tok, j) => (
                          <span
                            key={j}
                            className={tok.c ? tokenColors[tok.c] : ""}
                          >
                            {tok.t}
                          </span>
                        ))}
                      </span>
                    </div>
                  ))}
                </code>
              </motion.pre>
            </AnimatePresence>
          </div>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 font-mono text-[12px] text-ink-400">
            <span>
              <span className="text-ink-500">deps:</span> numpy ≥ 1.23, scipy ≥ 1.9
            </span>
            <span>
              <span className="text-ink-500">python:</span> 3.10 – 3.13
            </span>
            <span>
              <span className="text-ink-500">license:</span> MIT
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
