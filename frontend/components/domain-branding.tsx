'use client';

import { useEffect } from 'react';
import { useDomain } from '@/lib/domain-context';
import { getApiBase } from '@/lib/api';

/**
 * DomainBranding - dynamically updates page title and favicon
 * based on the active domain pack configuration.
 * Renders nothing visible; side-effect only.
 */
export function DomainBranding() {
  const domain = useDomain();

  useEffect(() => {
    // Update page title from domain branding
    if (domain.app_name) {
      document.title = domain.app_name;
    }

    // Update favicon to domain logo endpoint
    let link = document.querySelector("link[rel='icon']") as HTMLLinkElement | null;
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      document.head.appendChild(link);
    }
    link.href = `${getApiBase()}/api/domain/logo`;
  }, [domain.app_name]);

  return null;
}
