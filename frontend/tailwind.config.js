/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        background: "#09090b",
        surface: "#18181b",
        elevated: "#27272a",
        border: "#27272a",
        "border-strong": "#3f3f46",
        "text-primary": "#fafafa",
        "text-secondary": "#a1a1aa",
        "text-muted": "#71717a",
        accent: "#6366f1",
        "accent-hover": "#818cf8",
        "transit-blue": "#60a5fa",
        "stagnant-amber": "#fbbf24",
        "active-green": "#4ade80",
        "error-red": "#f87171",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "toast-in": {
          from: { transform: "translateY(16px)", opacity: "0" },
          to: { transform: "translateY(0)", opacity: "1" },
        },
        "toast-out": {
          from: { transform: "translateY(0)", opacity: "1" },
          to: { transform: "translateY(16px)", opacity: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "toast-in": "toast-in 0.2s ease-out",
        "toast-out": "toast-out 0.15s ease-in forwards",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
