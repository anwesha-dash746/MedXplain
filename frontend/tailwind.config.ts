import type { Config } from "tailwindcss";

/**
 * The palette is deliberately restrained. A clinical dashboard has one job with
 * colour — make the risk level and the explanation heat map readable at a glance
 * — so everything else stays neutral slate, and saturated colour is reserved for
 * the two things that carry meaning.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // brand / chrome
        clinical: {
          50: "#f2f7fb",
          100: "#e3edf6",
          200: "#c2d9ea",
          300: "#8fbad8",
          400: "#5596c1",
          500: "#3379a9",
          600: "#265f8c",
          700: "#204d71",
          800: "#1e415e",
          900: "#1d3850",
        },
        // risk bands, used by the gauge and the badges
        risk: {
          low: "#0f9d76",
          borderline: "#c58a12",
          elevated: "#dd6b20",
          high: "#c53030",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-1000px 0" },
          "100%": { backgroundPosition: "1000px 0" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 2s infinite linear",
        "fade-in": "fade-in 0.25s ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
