import { useState } from "react";
import { Logo } from "./Logo";

export function Nav() {
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-ink-200 bg-[#f6f1e8]/90 backdrop-blur-md">
      <div className="container-x">
        <div
          className="flex h-16 items-center justify-between px-1"
        >
          <a href="#top" className="pl-2">
            <Logo />
          </a>
          <nav className="hidden items-center gap-8 md:flex">
            {[
              ["Product", "#product"],
              ["Penalties", "#penalties"],
              ["Method", "#method"],
              ["Reproduce", "#reproduce"],
            ].map(([label, href]) => (
              <a
                key={href}
                href={href}
                className="text-[13px] text-ink-300 transition-colors hover:text-white"
              >
                {label}
              </a>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/10 text-ink-200 md:hidden"
              aria-expanded={open}
              aria-controls="mobile-nav"
              aria-label={open ? "Close navigation" : "Open navigation"}
              onClick={() => setOpen((value) => !value)}
            >
              <span aria-hidden="true">{open ? "×" : "☰"}</span>
            </button>
            <a
              href="https://github.com/sachncs/regulo"
              target="_blank"
              rel="noreferrer"
              className="hidden items-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] text-ink-200 transition-colors hover:text-white sm:inline-flex"
            >
              <svg
                viewBox="0 0 24 24"
                className="h-4 w-4"
                fill="currentColor"
                aria-hidden="true"
              >
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2Z"
                />
              </svg>
              <span>GitHub</span>
            </a>
            <a
              href="#install"
              className="inline-flex items-center gap-1 rounded-full bg-white px-4 py-1.5 text-[13px] font-medium text-ink-950 transition-transform hover:scale-[1.02]"
            >
              Get started
              <svg
                viewBox="0 0 16 16"
                className="h-3.5 w-3.5"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M8.22 4.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 8 8.22 4.28a.75.75 0 0 1 0-1.06Z" />
                <path d="M3.75 8a.75.75 0 0 1 .75-.75h8.5a.75.75 0 0 1 0 1.5h-8.5A.75.75 0 0 1 3.75 8Z" />
              </svg>
            </a>
          </div>
        </div>
        {open && (
          <nav id="mobile-nav" className="mt-2 grid gap-1 rounded-xl border border-ink-200 bg-[#fffdf8] p-2 shadow-lg md:hidden">
            {[
              ["Product", "#product"],
              ["Penalties", "#penalties"],
              ["Method", "#method"],
              ["Reproduce", "#reproduce"],
            ].map(([label, href]) => (
              <a
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className="rounded-xl px-4 py-3 text-sm text-ink-200 hover:bg-white/[0.06] hover:text-white"
              >
                {label}
              </a>
            ))}
          </nav>
        )}
      </div>
    </header>
  );
}
