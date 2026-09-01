import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedXplain - Explainable Clinical Risk Prediction",
  description:
    "Transformer-based clinical risk prediction from unstructured notes, with word-level explanations from attention rollout, SHAP and integrated gradients.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
