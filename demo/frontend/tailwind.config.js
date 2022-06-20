module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Sora', 'sans-serif'],
      },
    },
  },
  plugins: [require('@tailwindcss/line-clamp')],
}
