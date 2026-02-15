import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Visualization - ARCA',
  description: '3D Borehole Visualization',
}

/**
 * Minimal layout for visualization routes.
 * No main app navigation - just the visualization.
 */
export default function VizLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="h-screen w-screen bg-[#212121] overflow-hidden">
      {children}
    </div>
  )
}
