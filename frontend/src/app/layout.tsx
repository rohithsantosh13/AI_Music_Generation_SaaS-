import "~/styles/globals.css";

export const metadata = {
  title: "AI Music Generation",
  description: "Generate music with AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gradient-to-b from-[#2e026d] to-[#15162c] text-white">
        {children}
      </body>
    </html>
  );
}