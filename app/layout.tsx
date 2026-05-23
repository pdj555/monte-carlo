import type { Metadata, Viewport } from "next";
import { Ubuntu_Mono } from "next/font/google";

import "./globals.css";

const ubuntuMono = Ubuntu_Mono({
  subsets: ["latin"],
  weight: ["400", "700"],
  display: "swap",
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Monte Carlo",
  description: "Monte Carlo simulation and walk-forward validation with auditable price provenance.",
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#05080c" },
    { color: "#ffffff" },
  ],
  colorScheme: "light dark",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html className={ubuntuMono.variable} data-theme="light" lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
