import { cn } from "../lib/cn";

export function Logo({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <span className="relative inline-flex h-7 w-7 items-center justify-center overflow-hidden rounded-[9px] bg-gradient-to-br from-brand-700 to-accent-500 shadow-[0_8px_24px_-6px_rgba(70,86,245,0.6)]">
        <img
          src={`${import.meta.env.BASE_URL}logo.svg`}
          alt=""
          className="relative h-6 w-6"
          aria-hidden="true"
        />
      </span>
      <span className="font-display text-[15px] font-semibold tracking-tight text-white">
        regulo
      </span>
    </div>
  );
}
