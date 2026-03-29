/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0f1117",
        surface: "#1a1d23",
        border: "#2a2d35",
        "text-primary": "#e1e1e1",
        "text-secondary": "#9ca3af",
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
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
