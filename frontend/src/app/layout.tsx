import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Image from "next/image";
import { Toaster } from 'react-hot-toast';

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "GRINS Financial Platform",
  description: "Financial data access, analysis, and export by GRINS",
  icons: { icon: "/grins-logo.png" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} font-sans`}>
      <body className={`bg-background text-foreground antialiased`}>
        <Toaster position="top-center" reverseOrder={false} />
        {/* Example of adding a global header/navbar if needed outside dashboard */}
        {/* <header className="p-4 bg-grins-blue-night text-grins-ivory">
          <Image src="/grins-logo.png" alt="GRINS Logo" width={80} height={20} />
        </header> */}
        {children}
      </body>
    </html>
  );
}
