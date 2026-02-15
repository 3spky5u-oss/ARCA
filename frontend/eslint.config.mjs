import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

const eslintConfig = [
  {
    ignores: [".next/**", "node_modules/**", "*.config.js", "*.config.mjs"],
  },
  ...nextCoreWebVitals,
  ...nextTypescript,
  {
    rules: {
      // Relax rules for faster iteration
      "@typescript-eslint/no-unused-vars": "warn",
      "@typescript-eslint/no-explicit-any": "warn",
      "react-hooks/exhaustive-deps": "warn",
      "@next/next/no-img-element": "off",
    },
  },
];

export default eslintConfig;
