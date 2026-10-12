import { Logo } from "./Logo";
import { docsHref } from "../pages/DocsPage";

const cols = [
  {
    title: "Project",
    links: [
      { label: "Source", href: "https://github.com/sachncs/regulo" },
      { label: "Releases", href: "https://github.com/sachncs/regulo/releases" },
      { label: "Changelog", href: docsHref("changelog") },
      { label: "Roadmap", href: "https://github.com/sachncs/regulo/issues" },
    ],
  },
  {
    title: "Documentation",
    links: [
      { label: "API reference", href: docsHref("api") },
      { label: "Math", href: docsHref("math") },
      { label: "Reproducing the paper", href: docsHref("reproduce") },
      { label: "Limits", href: docsHref("limits") },
    ],
  },
  {
    title: "Community",
    links: [
      { label: "Issues", href: "https://github.com/sachncs/regulo/issues" },
      { label: "Discussions", href: "https://github.com/sachncs/regulo/discussions" },
      { label: "Contributing", href: "https://github.com/sachncs/regulo" },
      { label: "Code of conduct", href: "https://github.com/sachncs/regulo" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="relative mt-12 border-t border-white/[0.06] pt-16 pb-12">
      <div className="container-x">
        <div className="grid grid-cols-1 gap-12 lg:grid-cols-[1.2fr_1fr_1fr_1fr]">
          <div>
            <Logo />
            <p className="mt-5 max-w-sm text-[13.5px] leading-relaxed text-ink-400">
              A pure-Python NumPy and SciPy reference implementation of
              covariance-aware regularizers. Adaptive Norm-Based Regularization
              for Neural Networks.
            </p>
            <div className="mt-6 flex items-center gap-3">
              <a
                href="https://github.com/sachncs/regulo"
                target="_blank"
                rel="noreferrer"
                aria-label="GitHub"
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.06] bg-white/[0.03] text-ink-200 transition-colors hover:border-white/10 hover:text-white"
              >
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
                  <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2Z" />
                </svg>
              </a>
              <a
                href={docsHref("reproduce")}
                aria-label="Reproduction"
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.06] bg-white/[0.03] text-ink-200 transition-colors hover:border-white/10 hover:text-white"
              >
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
                  <path d="M12 2.5 2.5 7.5v9L12 21.5l9.5-5v-9L12 2.5Zm0 1.85L19.6 8.3 12 12.05 4.4 8.3 12 4.35ZM4 9.65l7 3.6v7.1l-7-3.7v-7Zm9 10.7v-7.1l7-3.6v7l-7 3.7Z" />
                </svg>
              </a>
            </div>
          </div>

          {cols.map((col) => (
            <div key={col.title}>
              <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-500">
                {col.title}
              </div>
              <ul className="mt-5 space-y-3">
                {col.links.map((l) => (
                  <li key={l.label}>
                    <a
                      href={l.href}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[13.5px] text-ink-300 transition-colors hover:text-white"
                    >
                      {l.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-16 flex flex-col items-start justify-between gap-4 border-t border-white/[0.06] pt-8 sm:flex-row sm:items-center">
          <div className="font-mono text-[11.5px] text-ink-500">
            © {new Date().getFullYear()} regulo · MIT licensed
          </div>
          <div className="flex items-center gap-5 font-mono text-[11.5px] text-ink-500">
            <span>v0.2.0</span>
            <span className="h-1 w-1 rounded-full bg-ink-700" />
            <span>NumPy + SciPy</span>
            <span className="h-1 w-1 rounded-full bg-ink-700" />
            <span>Reproducible</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
