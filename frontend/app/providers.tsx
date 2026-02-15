'use client';

import { DomainProvider } from '@/lib/domain-context';

export function Providers({ children }: { children: React.ReactNode }) {
  return <DomainProvider>{children}</DomainProvider>;
}
