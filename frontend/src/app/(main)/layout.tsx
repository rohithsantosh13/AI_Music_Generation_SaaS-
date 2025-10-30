import "~/styles/globals.css";

import { Providers } from "~/components/provider";
import { Toaster } from "~/components/ui/sonner";
import { SidebarInset, SidebarProvider } from "~/components/ui/sidebar";
import { AppSidebar } from "~/components/sidebar/app-sidebar";
import { SidebarTrigger } from "~/components/ui/sidebar";
import { Separator } from "~/components/ui/separator";

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
        <Providers>
          <SidebarProvider>
            <AppSidebar />
            <SidebarInset className="flex h-screen flex-col">
              <header className="flex items-center justify-between p-4">
                <div>
                  <SidebarTrigger />
                  <Separator />
                </div>
              </header>
              <main className="flex-1 overflow-y-auto">
                {children}
              </main>
            </SidebarInset>
          </SidebarProvider>
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}