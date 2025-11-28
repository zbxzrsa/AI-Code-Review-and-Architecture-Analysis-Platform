/** @type {import('tailwindcss').Config } */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
    "./index.html"
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#3b82f6',
          100: '#1e40af',
          500: '#1e3a8a',
          600: '#1d4ed8',
          700: '#1e40af',
          900: '#1e3a8a',
        },
        secondary: {
          50: '#f9fafb',
          100: '#f3f4f6',
          500: '#e11d48',
          600: '#f59e0b',
          700: '#f9fafb',
          900: '#f3f4f6',
        },
        success: {
          50: '#10b981',
          100: '#059669',
          500: '#047857',
          600: '#059669',
          700: '#10b981',
          900: '#059669',
        },
        warning: {
          50: '#fbbf24',
          100: '#fef3c7',
          500: '#f59e0b',
          600: '#fbbf24',
          700: '#fef3c7',
          900: '#fbbf24',
        },
        error: {
          50: '#ef4444',
          100: '#f87171',
          500: '#dc2626',
          600: '#ef4444',
          700: '#f87171',
          900: '#dc2626',
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#9ca3af',
          700: '#6b7280',
          800: '#4b5563',
          900: '#374151',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'Noto Sans', 'sans-serif'],
        mono: ['Monaco', 'Menlo', 'Monaco, Consolas', 'Liberation Mono', 'Courier New', 'monospace'],
      },
      animation: {
        keyframes: {
          'spin': {
            '0%': { transform: 'rotate(0deg)' },
            '100%': { transform: 'rotate(360deg)' },
          },
        },
      },
      plugins: [
        require('@tailwindcss/forms'),
        require('@tailwindcss/typography'),
        require('@tailwindcss/aspect-ratio'),
      ],
  },
  darkMode: 'class',
} satisfies import('tailwindcss').Config;