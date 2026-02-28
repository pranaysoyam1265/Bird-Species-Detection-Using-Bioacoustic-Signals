import type { Metadata, Viewport } from 'next'
import { JetBrains_Mono } from 'next/font/google'
import { GeistPixelGrid } from 'geist/font/pixel'
import { ThemeProvider } from '@/components/theme-provider'

import { AuthProvider } from '@/contexts/auth-context'
import { ClientProviders } from '@/components/client-providers'

import { Toaster } from '@/components/ui/toaster'

import './globals.css'

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
})

export const metadata: Metadata = {
  title: 'BirdSense | AI-Powered Bird Species Detection from Audio',
  description:
    'BirdSense is a deep learning system that identifies 87 North American bird species from audio recordings with 96% accuracy. Upload field recordings, analyze spectrograms, and get confidence-scored species predictions in seconds.',
  keywords: [
    'bird species detection',
    'bioacoustic AI',
    'bird call identification',
    'audio species recognition',
    'spectrogram analysis',
    'ornithology AI',
    'bird song classifier',
    'deep learning birds',
    'wildlife audio analysis',
    'bird monitoring system',
    'avian bioacoustics',
    'field recording analysis',
    'bird sound recognition',
    'conservation technology',
    'PyTorch bird model',
    'North American birds',
  ],
  authors: [{ name: 'BirdSense' }],
  creator: 'BirdSense',
  publisher: 'BirdSense',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    title: 'BirdSense | AI-Powered Bird Species Detection from Audio',
    description:
      'Identify 87 bird species from audio with 96% accuracy. Deep learning bioacoustic analysis with spectrogram visualization and confidence scoring.',
    siteName: 'BirdSense',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'BirdSense | AI Bird Species Detection',
    description:
      'Deep learning system identifying 87 North American bird species from audio recordings. 96% accuracy. Spectrogram analysis. Confidence scoring.',
    creator: '@birdsense',
  },
  category: 'technology',
}

export const viewport: Viewport = {
  themeColor: '#F2F1EA',
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${jetbrainsMono.variable} ${GeistPixelGrid.variable}`} suppressHydrationWarning>
      <body className="font-mono antialiased">
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange>
          <AuthProvider>
            <ClientProviders>
              {children}
            </ClientProviders>
          </AuthProvider>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}
