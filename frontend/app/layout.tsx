import type { Metadata } from 'next'
import { Comfortaa } from 'next/font/google'
import './globals.css'
import { Toaster } from '@/components/ui/toaster'
import { Providers } from './providers'
import { DomainBranding } from '@/components/domain-branding'

const comfortaa = Comfortaa({
  subsets: ['latin'],
  variable: '--font-brand',
  weight: ['300', '400', '500', '600', '700'],
})

export const metadata: Metadata = {
  title: 'ARCA',
  description: 'From Documents to Domain Expert',
  icons: {
    icon: '/logo.png',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={comfortaa.variable}>
        <Providers>
          <DomainBranding />
          {children}
        </Providers>
        <Toaster />
      </body>
    </html>
  )
}
