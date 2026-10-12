export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
        display: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: {
          50: "#F7F8FA", 100: "#EEF0F4", 200: "#DBDFE6", 300: "#B8BFCB",
          400: "#8C95A6", 500: "#5F6878", 600: "#3F4856", 700: "#2A323F",
          800: "#1A2029", 850: "#131820", 900: "#0C1018", 950: "#070A10",
        },
        brand: {
          50: "#EAF1FB", 100: "#D2E2F6", 200: "#A4C4ED", 300: "#6FA0DF",
          400: "#3F7CD0", 500: "#2563B9", 600: "#1B4F96", 700: "#163E78",
          800: "#13315F", 900: "#0E2447",
        },
        accent: {
          50: "#F1F4FF", 100: "#E1E7FF", 200: "#B7C5FF", 300: "#8A9FFF",
          400: "#6477FF", 500: "#4656F5", 600: "#3438D8", 700: "#2727AD",
          800: "#1E1E8A", 900: "#171870",
        },
      },
      backgroundImage: {
        "grid-fade": "radial-gradient(circle at 50% 0%, rgba(70,86,245,0.18), transparent 60%)",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.05), 0 30px 80px -20px rgba(70,86,245,0.35)",
      },
    },
  },
};
